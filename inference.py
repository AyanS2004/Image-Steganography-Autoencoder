import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

from models import SteganographyAutoencoder
from dataset import denormalize_image


def load_image(image_path, image_size=128):
    """Load and preprocess a single image"""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image


def save_image(tensor, filepath):
    """Save tensor as image"""
    # Denormalize and convert to PIL
    image = denormalize_image(tensor.squeeze(0))
    image = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    image = Image.fromarray(image)
    image.save(filepath)


def visualize_results(cover, secret, stego, secret_recovered, save_path=None):
    """Visualize the steganography results"""
    # Denormalize images
    cover = denormalize_image(cover.squeeze(0))
    secret = denormalize_image(secret.squeeze(0))
    stego = denormalize_image(stego.squeeze(0))
    secret_recovered = denormalize_image(secret_recovered.squeeze(0))
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(cover.permute(1, 2, 0))
    axes[0].set_title('Cover Image')
    axes[0].axis('off')
    
    axes[1].imshow(secret.permute(1, 2, 0))
    axes[1].set_title('Secret Image')
    axes[1].axis('off')
    
    axes[2].imshow(stego.permute(1, 2, 0))
    axes[2].set_title('Stego Image')
    axes[2].axis('off')
    
    axes[3].imshow(secret_recovered.permute(1, 2, 0))
    axes[3].set_title('Recovered Secret')
    axes[3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()


def calculate_metrics(original, reconstructed):
    """Calculate PSNR and SSIM between original and reconstructed images"""
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim
    
    # Convert to numpy arrays
    orig_np = denormalize_image(original.squeeze(0)).permute(1, 2, 0).numpy()
    recon_np = denormalize_image(reconstructed.squeeze(0)).permute(1, 2, 0).numpy()
    
    # Calculate metrics
    psnr_val = psnr(orig_np, recon_np, data_range=1.0)
    ssim_val = ssim(orig_np, recon_np, multichannel=True, data_range=1.0)
    
    return psnr_val, ssim_val


def main():
    parser = argparse.ArgumentParser(description='Steganography Inference')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--cover_image', type=str, required=True, help='Path to cover image')
    parser.add_argument('--secret_image', type=str, required=True, help='Path to secret image')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    parser.add_argument('--image_size', type=int, default=128, help='Image size')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--visualize', action='store_true', help='Show visualization')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print("Loading model...")
    model = SteganographyAutoencoder(
        in_channels=3,
        hidden_dims=[64, 128, 256, 512]
    )
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load images
    print("Loading images...")
    cover = load_image(args.cover_image, args.image_size).to(device)
    secret = load_image(args.secret_image, args.image_size).to(device)
    
    # Perform steganography
    print("Performing steganography...")
    with torch.no_grad():
        stego, secret_recovered = model(cover, secret)
    
    # Save results
    print("Saving results...")
    save_image(stego, os.path.join(args.output_dir, 'stego_image.png'))
    save_image(secret_recovered, os.path.join(args.output_dir, 'recovered_secret.png'))
    
    # Calculate metrics
    print("Calculating metrics...")
    cover_psnr, cover_ssim = calculate_metrics(cover, stego)
    secret_psnr, secret_ssim = calculate_metrics(secret, secret_recovered)
    
    print(f"\nResults:")
    print(f"Cover Image - PSNR: {cover_psnr:.2f} dB, SSIM: {cover_ssim:.4f}")
    print(f"Secret Image - PSNR: {secret_psnr:.2f} dB, SSIM: {secret_ssim:.4f}")
    
    # Save visualization
    if args.visualize:
        visualize_results(
            cover, secret, stego, secret_recovered,
            save_path=os.path.join(args.output_dir, 'visualization.png')
        )
    
    print(f"\nResults saved to {args.output_dir}/")


if __name__ == '__main__':
    main()

