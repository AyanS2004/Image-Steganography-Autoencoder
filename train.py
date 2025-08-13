import torch
import torch.nn as nn
import argparse
import os

from models import SteganographyAutoencoder
from dataset import create_dataloader
from trainer import SteganographyTrainer


def main():
    parser = argparse.ArgumentParser(description='Train Steganography Autoencoder')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--image_size', type=int, default=128, help='Image size')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--save_freq', type=int, default=5, help='Save checkpoint frequency')
    parser.add_argument('--visualize_freq', type=int, default=5, help='Visualization frequency')
    parser.add_argument('--use_cifar', action='store_true', help='Use CIFAR-10 dataset')
    parser.add_argument('--custom_image_dir', type=str, default=None, help='Custom image directory')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = create_dataloader(
        batch_size=args.batch_size,
        image_size=args.image_size,
        use_cifar=args.use_cifar,
        custom_image_dir=args.custom_image_dir
    )
    
    # Create model
    print("Creating model...")
    model = SteganographyAutoencoder(
        in_channels=3,
        hidden_dims=[64, 128, 256, 512]
    )
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    print("Creating trainer...")
    trainer = SteganographyTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=args.lr
    )
    
    # Start training
    print("Starting training...")
    trainer.train(
        num_epochs=args.epochs,
        save_freq=args.save_freq,
        visualize_freq=args.visualize_freq
    )
    
    print("Training completed!")


if __name__ == '__main__':
    main()

