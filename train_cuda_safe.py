# train_cuda_safe.py - CUDA-safe training with NaN recovery, AMP auto-tune, and resume support
import os
import argparse
import time
import gc
from datetime import datetime, timedelta

import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import psutil

from models import SteganographyAutoencoder, SteganographyLoss
from dataset import build_loaders

# ==============================
# Fine-tuning defaults (safe)
# ==============================
DEFAULT_LR = 5e-4
DEFAULT_GRAD_CLIP = 0.5
DEFAULT_ACCUM_STEPS = 2
DEFAULT_MAX_EPOCHS = 50
LR_REDUCE_ON_NAN = True
SAVE_RECOVERY_ON_NAN = True
INIT_AMP_SCALE = 2.**16
AMP_GROWTH_INTERVAL = 1000
# ==============================


def setup_cuda_environment():
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        return torch.device('cpu')
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    device = torch.device('cuda')
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f'Using GPU: {gpu_name} | Memory: {gpu_memory:.2f} GB')
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.cuda.empty_cache()
    return device


class SafeProgressTracker:
    def __init__(self, total_epochs, save_dir="progress_logs"):
        self.total_epochs = total_epochs
        self.save_dir = save_dir
        self.start_time = time.time()
        self.epoch_times = []
        self.best_metrics = {}
        os.makedirs(save_dir, exist_ok=True)
        self.log_file = os.path.join(save_dir, f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        self.log("=" * 80)
        self.log(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f"Total epochs: {total_epochs}")
        self.log("=" * 80)

    def log(self, message):
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        print(log_message, flush=True)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')
            f.flush()

    def start_epoch(self, epoch):
        self.current_epoch = epoch
        self.epoch_start_time = time.time()
        if len(self.epoch_times) > 0:
            avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
            remaining_epochs = self.total_epochs - epoch + 1
            eta_seconds = avg_epoch_time * remaining_epochs
            eta = datetime.now() + timedelta(seconds=eta_seconds)
            eta_str = f" | ETA: {eta.strftime('%H:%M:%S')}"
        else:
            eta_str = ""
        self.log(f"\n{'='*60}")
        self.log(f"Epoch {epoch}/{self.total_epochs}{eta_str}")
        self.log(f"{'='*60}")

    def end_epoch(self, metrics):
        epoch_time = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_time)
        for key, value in metrics.items():
            if key not in self.best_metrics or value < self.best_metrics[key]:
                self.best_metrics[key] = value
        total_elapsed = time.time() - self.start_time
        self.log(f"\nEpoch {self.current_epoch} Summary:")
        self.log(f"  Duration: {epoch_time:.1f}s")
        self.log(f"  Total elapsed: {total_elapsed/60:.1f}min")
        for key, value in metrics.items():
            best_indicator = "★" if value == self.best_metrics[key] else " "
            self.log(f"  {key}: {value:.6f} {best_indicator}")
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3
            gpu_reserved = torch.cuda.memory_reserved() / 1024**3
            self.log(f"  GPU Memory: {gpu_memory:.1f}GB / {gpu_reserved:.1f}GB reserved")
        cpu_percent = psutil.cpu_percent(interval=None)
        memory_percent = psutil.virtual_memory().percent
        self.log(f"  System: CPU {cpu_percent:.1f}% | RAM {memory_percent:.1f}%")


def safe_to_device(tensor, device, non_blocking=False):
    try:
        return tensor.to(device, non_blocking=non_blocking)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(f"CUDA OOM error, clearing cache and retrying...")
            torch.cuda.empty_cache()
            gc.collect()
            return tensor.to(device, non_blocking=False)
        else:
            raise e


def main():
    parser = argparse.ArgumentParser(description="CUDA-Safe Steganography Training")
    parser.add_argument('--train_dir', type=str, required=True)
    parser.add_argument('--val_dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=DEFAULT_LR)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--cosine', action='store_true')
    parser.add_argument('--tmax', type=int, default=100)
    parser.add_argument('--early_stop_patience', type=int, default=15)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=DEFAULT_ACCUM_STEPS)
    parser.add_argument('--max_grad_norm', type=float, default=DEFAULT_GRAD_CLIP)
    parser.add_argument('--save_freq', type=int, default=5)
    parser.add_argument('--vis_freq', type=int, default=10)
    parser.add_argument('--out_dir', type=str, default='cuda_safe_results')
    parser.add_argument('--ckpt_dir', type=str, default='cuda_safe_checkpoints')
    parser.add_argument('--use_cifar', action='store_true')
    parser.add_argument('--resume', type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    device = setup_cuda_environment()
    if device.type == 'cpu':
        args.device = 'cpu'

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    progress = SafeProgressTracker(args.epochs, save_dir=args.out_dir)
    progress.log(f"Using device: {device}")

    # Data
    if args.use_cifar or not os.path.exists(args.train_dir):
        from dataset import CIFARPairDataset
        progress.log("Using CIFAR-10 dataset")
        train_dataset = CIFARPairDataset(train=True, image_size=args.image_size)
        val_dataset = CIFARPairDataset(train=False, image_size=args.image_size)
        train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=(device.type == 'cuda'))
        val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=(device.type == 'cuda'))
    else:
        train_loader, val_loader = build_loaders(
            use_cifar=False,
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=min(args.num_workers, 2),
            val_split=0.1,
            pin_memory=(device.type == 'cuda')
        )

    # Model / Loss / Opt / Sched
    model = SteganographyAutoencoder(in_channels=3, hidden_dims=[64, 128, 256]).to(device)
    criterion = SteganographyLoss(alpha=1.0, beta=1.0, gamma=0.05, use_ssim=True).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.tmax) if args.cosine else None
    scaler = GradScaler(device, enabled=(device.type == 'cuda'), init_scale=INIT_AMP_SCALE, growth_interval=AMP_GROWTH_INTERVAL)

    history = {"train_loss": [], "val_loss": []}
    best_val = float('inf')
    start_epoch = 1

    # Resume
    if args.resume and os.path.exists(args.resume):
        progress.log(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scaler_state_dict" in checkpoint and scaler is not None:
            try:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
            except Exception as e:
                progress.log(f"Scaler state load failed: {e}")
        if "best_val_loss" in checkpoint:
            best_val = checkpoint["best_val_loss"]
        if "history" in checkpoint:
            history = checkpoint["history"]
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1
        progress.log(f"Resumed from epoch {start_epoch} with best_val={best_val:.6f}")

    def save_recovery(epoch, batch_idx, reason: str):
        ckpt = {
            "epoch": epoch,
            "batch_idx": batch_idx,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "best_val_loss": best_val,
            "history": history,
            "lr": optimizer.param_groups[0]['lr'],
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        }
        path = os.path.join(args.ckpt_dir, f"recovery_epoch_{epoch:03d}_batch_{batch_idx:05d}.pth")
        torch.save(ckpt, path)
        progress.log(f"✓ Recovery checkpoint saved: {path}")

    def adjust_after_nan():
        if LR_REDUCE_ON_NAN:
            for g in optimizer.param_groups:
                g['lr'] = max(g['lr'] * 0.5, 1e-8)
            progress.log(f"LR reduced to {optimizer.param_groups[0]['lr']}")
        try:
            current_scale = float(scaler.get_scale())
            new_scale = max(current_scale / 2.0, 1.0)
            if hasattr(scaler, "_scale"):
                scaler._scale = torch.tensor(new_scale, device=scaler._scale.device)
                progress.log(f"Loss scale reduced: {current_scale:.1f} → {new_scale:.1f}")
        except Exception as e:
            progress.log(f"Could not adjust AMP loss scale directly: {e}")

    def run_epoch_safe(loader, train_mode: bool, epoch_num: int):
        mode_str = "Training" if train_mode else "Validation"
        model.train() if train_mode else model.eval()
        total_loss = 0.0
        n_batches = 0
        pbar = tqdm(loader, desc=f"{mode_str} Epoch {epoch_num}", leave=False)

        accum_steps = max(1, int(args.gradient_accumulation_steps))
        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        first_logged = False

        for batch_idx, (cover, secret) in enumerate(pbar):
            cover = safe_to_device(cover, device, non_blocking=True)
            secret = safe_to_device(secret, device, non_blocking=True)

            if not first_logged:
                progress.log(f"[{mode_str}] cover range: {cover.min().item():.4f} .. {cover.max().item():.4f}")
                progress.log(f"[{mode_str}] secret range: {secret.min().item():.4f} .. {secret.max().item():.4f}")
                first_logged = True

            with autocast(device_type='cuda', enabled=(scaler is not None and device.type == 'cuda')):
                stego, secret_rec = model(cover, secret)
                losses = criterion(cover, stego, secret, secret_rec)
                loss = losses['total_loss']

            if not torch.isfinite(loss):
                progress.log(f"Non-finite loss at batch {batch_idx} ({mode_str}). Skipping batch and recovering.")
                adjust_after_nan()
                if SAVE_RECOVERY_ON_NAN and train_mode:
                    save_recovery(epoch_num, batch_idx, reason="non_finite_loss")
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                gc.collect()
                continue

            loss_to_scale = loss / accum_steps

            if train_mode:
                if scaler is not None:
                    scaler.scale(loss_to_scale).backward()
                else:
                    loss_to_scale.backward()

                do_step = ((batch_idx + 1) % accum_steps == 0) or (batch_idx + 1 == len(loader))
                if do_step:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            total_loss += float(loss.detach().item())
            n_batches += 1
            pbar.set_postfix({"loss": f"{(total_loss / n_batches):.4f}"})

            if device.type == 'cuda' and (batch_idx % 20 == 0):
                torch.cuda.empty_cache()

        pbar.close()
        return total_loss / max(1, n_batches)

    # Main loop
    epochs_no_improve = 0
    for epoch in range(start_epoch, args.epochs + 1):
        progress.start_epoch(epoch)

        train_loss = run_epoch_safe(train_loader, True, epoch)
        with torch.no_grad():
            val_loss = run_epoch_safe(val_loader, False, epoch)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if scheduler is not None:
            scheduler.step()

        metrics = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": optimizer.param_groups[0]['lr']
        }
        progress.end_epoch(metrics)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val
            }, os.path.join(args.ckpt_dir, "best_model.pth"))
            progress.log("  ★ New best model saved!")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= args.early_stop_patience:
            progress.log(f"Early stopping after {epochs_no_improve} epochs without improvement")
            break

        if epoch % args.save_freq == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict() if scaler else None,
                "best_val_loss": best_val,
                "history": history
            }, os.path.join(args.ckpt_dir, f"checkpoint_epoch_{epoch:03d}.pth"))
            progress.log(f"✓ Checkpoint saved for epoch {epoch}")

        if device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

    progress.log("Training session ended")


if __name__ == "__main__":
    main()
