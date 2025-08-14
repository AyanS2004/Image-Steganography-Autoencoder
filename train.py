# train.py - Enhanced with comprehensive progress tracking and modern PyTorch APIs
import os
import argparse
import math
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime, timedelta
import psutil
import gc

from models import SteganographyAutoencoder, SteganographyLoss
from dataset import CustomImageDataset, denormalize_image, build_loaders


class ProgressTracker:
    """Comprehensive progress tracking and monitoring"""
    
    def __init__(self, total_epochs, save_dir="progress_logs"):
        self.total_epochs = total_epochs
        self.save_dir = save_dir
        self.start_time = time.time()
        self.epoch_times = []
        self.best_metrics = {}
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize log file
        self.log_file = os.path.join(save_dir, f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        self.log("=" * 80)
        self.log(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f"Total epochs: {total_epochs}")
        self.log("=" * 80)
    
    def log(self, message):
        """Log message to both console and file"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')
    
    def start_epoch(self, epoch):
        """Start tracking an epoch"""
        self.current_epoch = epoch
        self.epoch_start_time = time.time()
        
        # Calculate ETA
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
        """End epoch tracking and log metrics"""
        epoch_time = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_time)
        
        # Update best metrics
        for key, value in metrics.items():
            if key not in self.best_metrics or value < self.best_metrics[key]:
                self.best_metrics[key] = value
        
        # Calculate total elapsed time
        total_elapsed = time.time() - self.start_time
        
        # Log detailed metrics
        self.log(f"\nEpoch {self.current_epoch} Summary:")
        self.log(f"  Duration: {epoch_time:.1f}s")
        self.log(f"  Total elapsed: {total_elapsed/60:.1f}min")
        
        for key, value in metrics.items():
            best_indicator = "★" if value == self.best_metrics[key] else " "
            self.log(f"  {key}: {value:.6f} {best_indicator}")
        
        # Memory usage
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3
            gpu_reserved = torch.cuda.memory_reserved() / 1024**3
            self.log(f"  GPU Memory: {gpu_memory:.1f}GB / {gpu_reserved:.1f}GB reserved")
        
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        self.log(f"  System: CPU {cpu_percent:.1f}% | RAM {memory_percent:.1f}%")
    
    def save_progress_plot(self, history, output_path):
        """Save detailed progress plots"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Loss curves
            axes[0,0].plot(history["train_loss"], label="Train", alpha=0.8)
            axes[0,0].plot(history["val_loss"], label="Val", alpha=0.8)
            axes[0,0].set_xlabel("Epoch")
            axes[0,0].set_ylabel("Loss")
            axes[0,0].set_title("Training & Validation Loss")
            axes[0,0].legend()
            axes[0,0].grid(True, alpha=0.3)
            
            # Recent progress (last 20 epochs)
            recent = min(20, len(history["train_loss"]))
            if recent > 1:
                axes[0,1].plot(history["train_loss"][-recent:], label="Train", alpha=0.8)
                axes[0,1].plot(history["val_loss"][-recent:], label="Val", alpha=0.8)
                axes[0,1].set_xlabel(f"Last {recent} Epochs")
                axes[0,1].set_ylabel("Loss")
                axes[0,1].set_title("Recent Progress")
                axes[0,1].legend()
                axes[0,1].grid(True, alpha=0.3)
            
            # Training speed
            if len(self.epoch_times) > 0:
                axes[1,0].plot(self.epoch_times, marker='o', alpha=0.7)
                axes[1,0].set_xlabel("Epoch")
                axes[1,0].set_ylabel("Time (seconds)")
                axes[1,0].set_title("Training Speed")
                axes[1,0].grid(True, alpha=0.3)
                
                # Add average line
                avg_time = sum(self.epoch_times) / len(self.epoch_times)
                axes[1,0].axhline(y=avg_time, color='r', linestyle='--', 
                                 label=f'Avg: {avg_time:.1f}s')
                axes[1,0].legend()
            
            # Improvement rate
            if len(history["val_loss"]) > 1:
                improvements = []
                for i in range(1, len(history["val_loss"])):
                    improvement = history["val_loss"][i-1] - history["val_loss"][i]
                    improvements.append(improvement)
                
                axes[1,1].plot(improvements, marker='o', alpha=0.7)
                axes[1,1].set_xlabel("Epoch")
                axes[1,1].set_ylabel("Loss Improvement")
                axes[1,1].set_title("Validation Loss Improvement")
                axes[1,1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
                axes[1,1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.log(f"Progress plot saved: {output_path}")
            
        except Exception as e:
            self.log(f"Failed to save progress plot: {e}")


def get_system_info():
    """Get system information for logging"""
    info = {
        "Python": f"{torch.__version__}",
        "PyTorch": torch.__version__,
        "CUDA Available": torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        info["GPU"] = torch.cuda.get_device_name()
        info["GPU Memory"] = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB"
    
    info["CPU Cores"] = psutil.cpu_count()
    info["RAM"] = f"{psutil.virtual_memory().total / 1024**3:.1f}GB"
    
    return info


# ---------- Utility: tiny CIFAR fallback dataset (pairs cover/secret) ----------
class CIFARPairDataset(Dataset):
    def __init__(self, root='./data', train=True, image_size=128):
        self.ds = torchvision.datasets.CIFAR10(
            root=root, train=train, download=True,
            transform=transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5),
                                     (0.5, 0.5, 0.5))
            ])
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        cover_img, _ = self.ds[idx]
        # pick a different index for secret
        secret_idx = idx
        while secret_idx == idx:
            secret_idx = torch.randint(0, len(self.ds), (1,)).item()
        secret_img, _ = self.ds[secret_idx]
        return cover_img, secret_img


def visualize_samples(cover, secret, stego, secret_rec, out_path, max_rows=4):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n = min(max_rows, cover.size(0))
    cover = denormalize_image(cover[:n].cpu())
    secret = denormalize_image(secret[:n].cpu())
    stego = denormalize_image(stego[:n].cpu())
    secret_rec = denormalize_image(secret_rec[:n].cpu())

    fig, axes = plt.subplots(n, 4, figsize=(16, 4*n))
    if n == 1:
        axes = [axes]  # make iterable

    for i in range(n):
        axes[i][0].imshow(cover[i].permute(1, 2, 0))
        axes[i][0].set_title('Cover')
        axes[i][0].axis('off')

        axes[i][1].imshow(secret[i].permute(1, 2, 0))
        axes[i][1].set_title('Secret')
        axes[i][1].axis('off')

        axes[i][2].imshow(stego[i].permute(1, 2, 0))
        axes[i][2].set_title('Stego')
        axes[i][2].axis('off')

        axes[i][3].imshow(secret_rec[i].permute(1, 2, 0))
        axes[i][3].set_title('Recovered')
        axes[i][3].axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_checkpoint(path, model, optimizer, epoch, scaler, best_val, history, progress_tracker=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint_data = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "best_val_loss": best_val,
        "history": history,
        "timestamp": datetime.now().isoformat()
    }
    
    if progress_tracker:
        checkpoint_data["epoch_times"] = progress_tracker.epoch_times
        checkpoint_data["best_metrics"] = progress_tracker.best_metrics
    
    torch.save(checkpoint_data, path)
    
    if progress_tracker:
        progress_tracker.log(f"✓ Checkpoint saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Train Steganography Autoencoder with Progress Tracking")

    # dataset paths (COCO style)
    parser.add_argument('--train_dir', type=str, default=r'D:\COCO\train',
                        help='Directory with training images')
    parser.add_argument('--val_dir', type=str, default=r'D:\COCO\val',
                        help='Directory with validation images')

    # training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=4)

    # scheduling / stopping
    parser.add_argument('--cosine', action='store_true', help='Use CosineAnnealingLR')
    parser.add_argument('--tmax', type=int, default=100, help='T_max for cosine scheduler (epochs)')
    parser.add_argument('--early_stop_patience', type=int, default=15)

    # housekeeping
    parser.add_argument('--save_freq', type=int, default=5)
    parser.add_argument('--vis_freq', type=int, default=5)
    parser.add_argument('--progress_freq', type=int, default=10, help='Save progress plots every N epochs')
    parser.add_argument('--out_dir', type=str, default='optimized_results')
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints')

    # testing options
    parser.add_argument('--use_cifar', action='store_true', help='Use CIFAR-10 for quick testing')

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Initialize progress tracker
    progress_tracker = ProgressTracker(args.epochs, save_dir=args.out_dir)
    
    # Log system information
    progress_tracker.log("System Information:")
    system_info = get_system_info()
    for key, value in system_info.items():
        progress_tracker.log(f"  {key}: {value}")
    
    progress_tracker.log(f"Using device: {device}")

    # Log training configuration
    progress_tracker.log("\nTraining Configuration:")
    for arg, value in vars(args).items():
        progress_tracker.log(f"  {arg}: {value}")

    try:
        # loaders
        if args.use_cifar:
            progress_tracker.log("Using CIFAR-10 dataset for testing")
            train_dataset = CIFARPairDataset(train=True, image_size=args.image_size)
            val_dataset = CIFARPairDataset(train=False, image_size=args.image_size)
            
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                    shuffle=True, num_workers=args.num_workers, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                                  shuffle=False, num_workers=args.num_workers, pin_memory=True)
        else:
            train_loader, val_loader = build_loaders(
                use_cifar=False,
                train_dir=args.train_dir,
                val_dir=args.val_dir,
                image_size=args.image_size,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                val_split=0.1
            )

        progress_tracker.log(f"Training batches: {len(train_loader)}")
        progress_tracker.log(f"Validation batches: {len(val_loader)}")

    except Exception as e:
        progress_tracker.log(f"Error loading data: {e}")
        progress_tracker.log("Falling back to CIFAR-10 dataset")
        train_dataset = CIFARPairDataset(train=True, image_size=args.image_size)
        val_dataset = CIFARPairDataset(train=False, image_size=args.image_size)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                shuffle=True, num_workers=args.num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                              shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # model + criterion
    model = SteganographyAutoencoder(in_channels=3, hidden_dims=[64, 128, 256, 512]).to(device)
    criterion = SteganographyLoss(alpha=1.0, beta=1.0, gamma=0.1, use_ssim=True)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = None
    if args.cosine:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.tmax)

    # Use modern GradScaler API
    scaler = GradScaler(device, enabled=(device.type == 'cuda'))

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    progress_tracker.log(f"Model: {total_params:,} total params | {trainable_params:,} trainable")

    history = {"train_loss": [], "val_loss": []}
    best_val = float('inf')
    best_path = os.path.join(args.ckpt_dir, "best_model.pth")

    def run_one_epoch(loader, train_mode: bool, epoch_num: int):
        if train_mode:
            model.train()
            desc = f"Training Epoch {epoch_num}"
        else:
            model.eval()
            desc = f"Validation Epoch {epoch_num}"

        total = 0.0
        n_batches = 0
        detailed_losses = {"cover_loss": 0, "secret_loss": 0, "perceptual_loss": 0}
        
        # Create progress bar
        pbar = tqdm(loader, desc=desc, leave=False, 
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        for batch_idx, (cover, secret) in enumerate(pbar):
            cover = cover.to(device, non_blocking=True)
            secret = secret.to(device, non_blocking=True)

            if train_mode:
                optimizer.zero_grad(set_to_none=True)

            # Use modern autocast API
            with autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                stego, secret_rec = model(cover, secret)
                losses = criterion(cover, stego, secret, secret_rec)
                loss = losses['total_loss']

            if train_mode:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total += loss.detach().item()
            n_batches += 1
            
            # Accumulate detailed losses
            for key in detailed_losses:
                if key in losses:
                    detailed_losses[key] += losses[key].detach().item()
            
            # Update progress bar
            current_avg = total / n_batches
            pbar.set_postfix({"loss": f"{current_avg:.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.6f}"})
            
            # Memory cleanup every 50 batches
            if batch_idx % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        pbar.close()
        
        # Average the detailed losses
        for key in detailed_losses:
            detailed_losses[key] /= max(1, n_batches)
        
        return total / max(1, n_batches), detailed_losses

    epochs_no_improve = 0

    try:
        for epoch in range(1, args.epochs + 1):
            progress_tracker.start_epoch(epoch)
            
            # Training
            train_loss, train_detailed = run_one_epoch(train_loader, train_mode=True, epoch_num=epoch)
            
            # Validation
            val_loss, val_detailed = run_one_epoch(val_loader, train_mode=False, epoch_num=epoch)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            if scheduler is not None:
                scheduler.step()

            # Prepare metrics for logging
            metrics = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": optimizer.param_groups[0]['lr']
            }
            
            # Add detailed loss components
            for key, value in train_detailed.items():
                metrics[f"train_{key}"] = value
            for key, value in val_detailed.items():
                metrics[f"val_{key}"] = value

            progress_tracker.end_epoch(metrics)

            # checkpointing
            if epoch % args.save_freq == 0:
                save_checkpoint(os.path.join(args.ckpt_dir, f"checkpoint_epoch_{epoch:03d}.pth"),
                              model, optimizer, epoch, scaler, best_val, history, progress_tracker)

            # best model
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                save_checkpoint(best_path, model, optimizer, epoch, scaler, best_val, history, progress_tracker)
                progress_tracker.log("  ★ New best model saved!")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            # visualization
            if epoch % args.vis_freq == 0:
                progress_tracker.log("Generating visualization...")
                model.eval()
                try:
                    with torch.no_grad():
                        cover_vis, secret_vis = next(iter(val_loader))
                        cover_vis = cover_vis.to(device)[:4]
                        secret_vis = secret_vis.to(device)[:4]
                        stego_vis, secret_rec_vis = model(cover_vis, secret_vis)
                    vis_path = os.path.join(args.out_dir, f"epoch_{epoch:03d}.png")
                    visualize_samples(cover_vis, secret_vis, stego_vis, secret_rec_vis, vis_path, max_rows=4)
                    progress_tracker.log(f"  ✓ Visualization saved: {vis_path}")
                except StopIteration:
                    progress_tracker.log("  ⚠ Could not generate visualization (no validation data)")

            # Save progress plots
            if epoch % args.progress_freq == 0:
                plot_path = os.path.join(args.out_dir, f"progress_epoch_{epoch:03d}.png")
                progress_tracker.save_progress_plot(history, plot_path)

            # early stopping
            if epochs_no_improve >= args.early_stop_patience:
                progress_tracker.log(f"⚠ Early stopping triggered! No improvement for {epochs_no_improve} epochs")
                break

            # Memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    except KeyboardInterrupt:
        progress_tracker.log("\n⚠ Training interrupted by user!")
        progress_tracker.log("Saving emergency checkpoint...")
        emergency_path = os.path.join(args.ckpt_dir, "emergency_checkpoint.pth")
        save_checkpoint(emergency_path, model, optimizer, epoch, scaler, best_val, history, progress_tracker)

    finally:
        # Final progress plot
        final_plot_path = os.path.join(args.out_dir, "final_progress.png")
        progress_tracker.save_progress_plot(history, final_plot_path)

        # Save final history
        with open(os.path.join(args.out_dir, "history.json"), "w") as f:
            json.dump({
                "history": history,
                "best_metrics": progress_tracker.best_metrics,
                "epoch_times": progress_tracker.epoch_times,
                "total_training_time": time.time() - progress_tracker.start_time
            }, f, indent=2)

        # Final summary
        total_time = time.time() - progress_tracker.start_time
        progress_tracker.log(f"\n{'='*80}")
        progress_tracker.log("TRAINING COMPLETE!")
        progress_tracker.log(f"{'='*80}")
        progress_tracker.log(f"  Duration: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
        progress_tracker.log(f"  Final epoch: {epoch}")
        progress_tracker.log(f"  Best validation loss: {best_val:.6f}")
        
        if len(progress_tracker.epoch_times) > 0:
            avg_epoch_time = sum(progress_tracker.epoch_times) / len(progress_tracker.epoch_times)
            progress_tracker.log(f"  Average epoch time: {avg_epoch_time:.1f}s")
        
        progress_tracker.log(f"  Best model: {best_path}")
        progress_tracker.log(f"  Results directory: {args.out_dir}")
        progress_tracker.log(f"  Training log: {progress_tracker.log_file}")
        progress_tracker.log(f"{'='*80}")


if __name__ == "__main__":
    main()