#!/usr/bin/env python3
"""
Demo script for Digital Watermarking project.
This script provides an easy way to test the steganography system with sample images.
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from models import SteganographyAutoencoder
from dataset import denormalize_image, SteganographyDataLoader
from trainer import SteganographyTrainer
import requests
from io import BytesIO


def download_sample_images():
    """Download sample images for demonstration"""
    sample_dir = "sample_images"
    os.makedirs(sample_dir, exist_ok=True)
    
    # Sample image URLs (using placeholder images)
    sample_urls = {
        "cover1.jpg": "https://picsum.photos/256/256?random=1",
        "cover2.jpg": "https://picsum.photos/256/256?random=2", 
        "secret1.jpg": "https://picsum.photos/256/256?random=3",
        "secret2.jpg": "https://picsum.photos/256/256?random=4"
    }
    
    print("Downloading sample images...")
    for filename, url in sample_urls.items():
        filepath = os.path.join(sample_dir, filename)
        if not os.path.exists(filepath):
            try:
                response = requests.get(url)
                response.raise_for_status()
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                print(f"Downloaded {filename}")
            except Exception as e:
                print(f"Failed to download {filename}: {e}")
        else:
            print(f"{filename} already exists")
    
    return sample_dir


def create_synthetic_images(image_size=128):
    """Create synthetic images for testing when internet is not available"""
    sample_dir = "sample_images"
    os.makedirs(sample_dir, exist_ok=True)
    
    print("Creating synthetic sample images...")
    
    # Create different patterns
    patterns = {
        "cover1.jpg": lambda: np.random.rand(image_size, image_size, 3) * 0.8 + 0.1,
        "cover2.jpg": lambda: create_gradient_image(image_size),
        "secret1.jpg": lambda: create_checkerboard_image(image_size),
        "secret2.jpg": lambda: create_circular_pattern(image_size)
    }
    
    for filename, pattern_func in patterns.items():
        filepath = os.path.join(sample_dir, filename)
        if not os.path.exists(filepath):
            img_array = (pattern_func() * 255).astype(np.uint8)
            img = Image.fromarray(img_array)
            img.save(filepath)
            print(f"Created {filename}")
    
    return sample_dir


def create_gradient_image(size):
    """Create a gradient image"""
    img = np.zeros((size, size, 3))
    for i in range(size):
        for j in range(size):
            img[i, j, 0] = i / size  # Red gradient
            img[i, j, 1] = j / size  # Green gradient
            img[i, j, 2] = (i + j) / (2 * size)  # Blue gradient
    return img


def create_checkerboard_image(size, square_size=16):
    """Create a checkerboard pattern"""
    img = np.zeros((size, size, 3))
    for i in range(size):
        for j in range(size):
            if ((i // square_size) + (j // square_size)) % 2 == 0:
                img[i, j] = [1, 0, 0]  # Red squares
            else:
                img[i, j] = [0, 0, 1]  # Blue squares
    return img


def create_circular_pattern(size):
    """Create a circular pattern"""
    img = np.zeros((size, size, 3))
    center = size // 2
    for i in range(size):
        for j in range(size):
            distance = np.sqrt((i - center)**2 + (j - center)**2)
            normalized_dist = distance / (size / 2)
            img[i, j] = [normalized_dist, 1 - normalized_dist, 0.5]
    return np.clip(img, 0, 1)


def load_image(image_path, image_size=128):
    """Load and preprocess an image"""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image


def quick_train_demo(device='cuda', epochs=10, image_size=64):
    """Quick training demonstration with small model and fewer epochs"""
    print("\n=== Quick Training Demo ===")
    print(f"Training for {epochs} epochs with image size {image_size}x{image_size}")
    
    # Create data loader with CIFAR-10
    data_loader = SteganographyDataLoader(
        batch_size=16,
        image_size=image_size,
        use_cifar=True
    )
    train_loader, val_loader = data_loader.get_loaders()
    
    # Create smaller model for quick demo
    model = SteganographyAutoencoder(
        in_channels=3,
        hidden_dims=[32, 64, 128]  # Smaller model
    )
    
    # Create trainer
    trainer = SteganographyTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=0.001
    )
    
    # Train for a few epochs
    trainer.train(
        num_epochs=epochs,
        save_freq=5,
        visualize_freq=2
    )
    
    print("Quick training demo completed!")
    return model


def inference_demo(model=None, device='cuda', image_size=128):
    """Demonstration of inference with sample images"""
    print("\n=== Inference Demo ===")
    
    # Try to get sample images
    try:
        sample_dir = download_sample_images()
    except:
        print("Could not download images, creating synthetic ones...")
        sample_dir = create_synthetic_images(image_size)
    
    # If no model provided, try to load from checkpoint or create new one
    if model is None:
        checkpoint_path = "checkpoints/checkpoint_epoch_5.pth"
        if os.path.exists(checkpoint_path):
            print("Loading model from checkpoint...")
            model = SteganographyAutoencoder(in_channels=3, hidden_dims=[64, 128, 256, 512])
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("No checkpoint found. Creating untrained model for demo...")
            model = SteganographyAutoencoder(in_channels=3, hidden_dims=[64, 128, 256, 512])
    
    model.to(device)
    model.eval()
    
    # Load sample images
    cover_path = os.path.join(sample_dir, "cover1.jpg")
    secret_path = os.path.join(sample_dir, "secret1.jpg")
    
    if not (os.path.exists(cover_path) and os.path.exists(secret_path)):
        print("Sample images not found!")
        return
    
    cover = load_image(cover_path, image_size).to(device)
    secret = load_image(secret_path, image_size).to(device)
    
    print(f"Loaded cover image: {cover_path}")
    print(f"Loaded secret image: {secret_path}")
    
    # Perform steganography
    with torch.no_grad():
        stego, secret_recovered = model(cover, secret)
    
    # Visualize results
    visualize_demo_results(cover, secret, stego, secret_recovered)
    
    # Calculate metrics
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim
    
    cover_np = denormalize_image(cover.squeeze(0)).permute(1, 2, 0).cpu().numpy()
    stego_np = denormalize_image(stego.squeeze(0)).permute(1, 2, 0).cpu().numpy()
    secret_np = denormalize_image(secret.squeeze(0)).permute(1, 2, 0).cpu().numpy()
    recovered_np = denormalize_image(secret_recovered.squeeze(0)).permute(1, 2, 0).cpu().numpy()
    
    cover_psnr = psnr(cover_np, stego_np, data_range=1.0)
    
    # Calculate SSIM with proper window size for small images
    try:
        height, width = cover_np.shape[:2]
        min_dim = min(height, width)
        if min_dim < 7:
            win_size = 3
        elif min_dim < 11:
            win_size = 5
        else:
            win_size = 7
        if win_size % 2 == 0:
            win_size -= 1
        cover_ssim = ssim(cover_np, stego_np, win_size=win_size, channel_axis=2, data_range=1.0)
    except Exception as e:
        print(f"Cover SSIM calculation failed: {e}")
        cover_ssim = np.corrcoef(cover_np.flatten(), stego_np.flatten())[0, 1]
        if np.isnan(cover_ssim):
            cover_ssim = 0.0
    
    secret_psnr = psnr(secret_np, recovered_np, data_range=1.0)
    
    # Calculate SSIM with proper window size for small images
    try:
        height, width = secret_np.shape[:2]
        min_dim = min(height, width)
        if min_dim < 7:
            win_size = 3
        elif min_dim < 11:
            win_size = 5
        else:
            win_size = 7
        if win_size % 2 == 0:
            win_size -= 1
        secret_ssim = ssim(secret_np, recovered_np, win_size=win_size, channel_axis=2, data_range=1.0)
    except Exception as e:
        print(f"Secret SSIM calculation failed: {e}")
        secret_ssim = np.corrcoef(secret_np.flatten(), recovered_np.flatten())[0, 1]
        if np.isnan(secret_ssim):
            secret_ssim = 0.0
    
    print(f"\nMetrics:")
    print(f"Cover preservation - PSNR: {cover_psnr:.2f} dB, SSIM: {cover_ssim:.4f}")
    print(f"Secret recovery - PSNR: {secret_psnr:.2f} dB, SSIM: {secret_ssim:.4f}")


def visualize_demo_results(cover, secret, stego, secret_recovered):
    """Visualize steganography results"""
    # Convert to displayable format
    cover_img = denormalize_image(cover.squeeze(0)).permute(1, 2, 0).cpu().numpy()
    secret_img = denormalize_image(secret.squeeze(0)).permute(1, 2, 0).cpu().numpy()
    stego_img = denormalize_image(stego.squeeze(0)).permute(1, 2, 0).cpu().numpy()
    recovered_img = denormalize_image(secret_recovered.squeeze(0)).permute(1, 2, 0).cpu().numpy()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    axes[0, 0].imshow(cover_img)
    axes[0, 0].set_title('Cover Image', fontsize=14)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(secret_img)
    axes[0, 1].set_title('Secret Image', fontsize=14)
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(stego_img)
    axes[1, 0].set_title('Stego Image\n(Cover with hidden Secret)', fontsize=14)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(recovered_img)
    axes[1, 1].set_title('Recovered Secret Image', fontsize=14)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    # Save the result
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/demo_result.png', dpi=150, bbox_inches='tight')
    print(f"Demo visualization saved to results/demo_result.png")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Digital Watermarking Demo')
    parser.add_argument('--mode', type=str, choices=['train', 'inference', 'both'], 
                       default='both', help='Demo mode')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs for demo')
    parser.add_argument('--image_size', type=int, default=64, help='Image size for demo')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = None
    
    print("🎭 Digital Watermarking Demo")
    print("=" * 50)
    print("This demo showcases image-in-image steganography using deep learning.")
    print("We will hide a secret image inside a cover image using autoencoders.\n")
    
    if args.mode in ['train', 'both']:
        try:
            model = quick_train_demo(device, args.epochs, args.image_size)
        except Exception as e:
            print(f"Training demo failed: {e}")
            print("Continuing with inference demo...")
    
    if args.mode in ['inference', 'both']:
        try:
            inference_demo(model, device, args.image_size)
        except Exception as e:
            print(f"Inference demo failed: {e}")
    
    print("\n" + "=" * 50)
    print("Demo completed! Check the 'results' and 'visualizations' folders for outputs.")
    print("\nTo run full training:")
    print("  python train.py --epochs 50 --batch_size 32 --use_cifar")
    print("\nTo run inference on custom images:")
    print("  python inference.py --checkpoint checkpoints/best_model.pth --cover_image your_cover.jpg --secret_image your_secret.jpg")


if __name__ == '__main__':
    main()
