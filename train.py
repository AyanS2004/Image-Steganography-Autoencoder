# train.py
import os
import argparse
import math
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from models import SteganographyAutoencoder, SteganographyLoss
from dataset import CustomImageDataset, denormalize_image

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


def build_loaders(
    use_cifar: bool,
    train_dir: str,
    val_dir: str | None,
    image_size: int,
    batch_size: int,
    num_workers: int,
    val_split: float = 0.1
):
    """
    Returns train_loader, val_loader
    - If use_cifar=True: uses CIFAR-10 pair dataset (80/20 split)
    - Else: uses CustomImageDataset from dataset.py with explicit val_dir if provided,
            otherwise creates a split from train_dir
    """
    if use_cifar:
        full_train = CIFARPairDataset(train=True, image_size=image_size)
        full_val = CIFARPairDataset(train=False, image_size=image_size)
        train_loader = DataLoader(full_train, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(full_val, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True)
        return train_loader, val_loader

    # Custom COCO-style directories
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Train directory not found: {train_dir}")

    train_dataset = CustomImageDataset(image_dir=train_dir, image_size=image_size)

    if val_dir and os.path.isdir(val_dir):
        val_dataset = CustomImageDataset(image_dir=val_dir, image_size=image_size)
    else:
        # split from train
        n_total = len(train_dataset)
        n_val = max(1, int(n_total * val_split))
        n_train = n_total - n_val
        train_dataset, val_dataset = random_split(
            train_dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42)
        )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


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


def save_checkpoint(path, model, optimizer, epoch, scaler, best_val, history):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "best_val_loss": best_val,
        "history": history
    }, path)


def main():
    parser = argparse.ArgumentParser(description="Train Steganography Autoencoder (AMP + EarlyStop)")
    # data
    parser.add_argument('--use_cifar', action='store_true', help='Use CIFAR-10 instead of custom directory')
    parser.add_argument('--train_dir', type=str, default=r'D:\COCO\train',
                        help='Directory with training images (ignored if --use_cifar)')
    parser.add_argument('--val_dir', type=str, default=r'D:\COCO\val',
                        help='Directory with validation images (optional; if missing, split from train)')
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
    parser.add_argument('--out_dir', type=str, default='optimized_results')
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints')

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # loaders
    train_loader, val_loader = build_loaders(
        use_cifar=args.use_cifar,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # model + criterion
    model = SteganographyAutoencoder(in_channels=3, hidden_dims=[64, 128, 256, 512]).to(device)
    criterion = SteganographyLoss(alpha=1.0, beta=1.0, gamma=0.1, use_ssim=True)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = None
    if args.cosine:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.tmax)

    scaler = GradScaler(enabled=(device.type == 'cuda'))

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,} | Trainable: {trainable_params:,}")

    history = {"train_loss": [], "val_loss": []}
    best_val = float('inf')
    best_path = os.path.join(args.ckpt_dir, "best_model.pth")
    start_time = time.time()

    def run_one_epoch(loader, train_mode: bool):
        if train_mode:
            model.train()
        else:
            model.eval()

        total = 0.0
        n_batches = 0

        for cover, secret in loader:
            cover = cover.to(device, non_blocking=True)
            secret = secret.to(device, non_blocking=True)

            if train_mode:
                optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=(device.type == 'cuda')):
                stego, secret_rec = model(cover, secret)
                losses = criterion(cover, stego, secret, secret_rec)
                loss = losses['total_loss']

            if train_mode:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total += loss.detach().item()
            n_batches += 1

        return total / max(1, n_batches)

    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = run_one_epoch(train_loader, train_mode=True)
        val_loss = run_one_epoch(val_loader, train_mode=False)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if scheduler is not None:
            scheduler.step()

        print(f"[{epoch:03d}/{args.epochs}] "
              f"train: {train_loss:.4f} | val: {val_loss:.4f} | "
              f"lr: {optimizer.param_groups[0]['lr']:.6f}")

        # checkpointing
        if epoch % args.save_freq == 0:
            save_checkpoint(os.path.join(args.ckpt_dir, f"checkpoint_epoch_{epoch}.pth"),
                            model, optimizer, epoch, scaler, best_val, history)

        # best model
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            save_checkpoint(best_path, model, optimizer, epoch, scaler, best_val, history)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # visualization
        if epoch % args.vis_freq == 0:
            model.eval()
            try:
                with torch.no_grad():
                    cover_vis, secret_vis = next(iter(val_loader))
                    cover_vis = cover_vis.to(device)[:4]
                    secret_vis = secret_vis.to(device)[:4]
                    stego_vis, secret_rec_vis = model(cover_vis, secret_vis)
                vis_path = os.path.join(args.out_dir, f"epoch_{epoch:03d}.png")
                visualize_samples(cover_vis, secret_vis, stego_vis, secret_rec_vis, vis_path, max_rows=4)
            except StopIteration:
                pass

        # early stopping
        if epochs_no_improve >= args.early_stop_patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {epochs_no_improve} epochs).")
            break

    # save training curves
    plt.figure(figsize=(7,5))
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training / Validation Loss")
    plt.legend()
    plt.tight_layout()
    curve_path = os.path.join(args.out_dir, "loss_curve.png")
    plt.savefig(curve_path, dpi=150)
    plt.close()

    # save final history
    with open(os.path.join(args.out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    elapsed = time.time() - start_time
    print(f"Done. Best val loss: {best_val:.6f}. Elapsed: {elapsed/60:.1f} min.")
    print(f"Best checkpoint: {best_path}")
    print(f"Results: {args.out_dir}")

if __name__ == "__main__":
    main()
