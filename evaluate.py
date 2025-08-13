#!/usr/bin/env python3
"""
Comprehensive evaluation script for Digital Watermarking system.
Provides detailed metrics, robustness testing, and benchmarking capabilities.
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import json
from tqdm import tqdm
import csv
from datetime import datetime

from models import SteganographyAutoencoder
from dataset import denormalize_image, create_dataloader
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import cv2


class SteganographyEvaluator:
    """Comprehensive evaluator for steganography models"""
    
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.results = {}
        
    def evaluate_basic_metrics(self, test_loader, num_samples=100):
        """Evaluate basic quality metrics (PSNR, SSIM)"""
        print("Evaluating basic metrics...")
        
        self.model.eval()
        psnr_cover = []
        ssim_cover = []
        psnr_secret = []
        ssim_secret = []
        
        with torch.no_grad():
            count = 0
            for cover, secret in tqdm(test_loader, desc="Basic metrics"):
                if count >= num_samples:
                    break
                    
                cover, secret = cover.to(self.device), secret.to(self.device)
                stego, secret_recovered = self.model(cover, secret)
                
                # Convert to numpy for metrics calculation
                for i in range(cover.size(0)):
                    if count >= num_samples:
                        break
                        
                    cover_np = denormalize_image(cover[i]).permute(1, 2, 0).cpu().numpy()
                    stego_np = denormalize_image(stego[i]).permute(1, 2, 0).cpu().numpy()
                    secret_np = denormalize_image(secret[i]).permute(1, 2, 0).cpu().numpy()
                    recovered_np = denormalize_image(secret_recovered[i]).permute(1, 2, 0).cpu().numpy()
                    
                    # Calculate metrics
                    psnr_cover.append(psnr(cover_np, stego_np, data_range=1.0))
                    ssim_cover.append(ssim(cover_np, stego_np, multichannel=True, data_range=1.0))
                    psnr_secret.append(psnr(secret_np, recovered_np, data_range=1.0))
                    ssim_secret.append(ssim(secret_np, recovered_np, multichannel=True, data_range=1.0))
                    
                    count += 1
        
        self.results['basic_metrics'] = {
            'cover_psnr_mean': float(np.mean(psnr_cover)),
            'cover_psnr_std': float(np.std(psnr_cover)),
            'cover_ssim_mean': float(np.mean(ssim_cover)),
            'cover_ssim_std': float(np.std(ssim_cover)),
            'secret_psnr_mean': float(np.mean(psnr_secret)),
            'secret_psnr_std': float(np.std(psnr_secret)),
            'secret_ssim_mean': float(np.mean(ssim_secret)),
            'secret_ssim_std': float(np.std(ssim_secret))
        }
        
        print(f"Cover Image Quality - PSNR: {np.mean(psnr_cover):.2f}±{np.std(psnr_cover):.2f} dB")
        print(f"Cover Image Quality - SSIM: {np.mean(ssim_cover):.4f}±{np.std(ssim_cover):.4f}")
        print(f"Secret Recovery - PSNR: {np.mean(psnr_secret):.2f}±{np.std(psnr_secret):.2f} dB")
        print(f"Secret Recovery - SSIM: {np.mean(ssim_secret):.4f}±{np.std(ssim_secret):.4f}")
        
    def evaluate_robustness(self, test_loader, num_samples=50):
        """Evaluate robustness against various attacks"""
        print("\nEvaluating robustness against attacks...")
        
        attacks = {
            'jpeg_compression_75': self._jpeg_compression_attack,
            'jpeg_compression_50': lambda x: self._jpeg_compression_attack(x, quality=50),
            'gaussian_noise': self._gaussian_noise_attack,
            'gaussian_blur': self._gaussian_blur_attack,
            'rotation': self._rotation_attack,
            'scaling': self._scaling_attack,
            'brightness': self._brightness_attack,
            'contrast': self._contrast_attack
        }
        
        robustness_results = {}
        
        for attack_name, attack_func in attacks.items():
            print(f"Testing {attack_name}...")
            psnr_vals = []
            ssim_vals = []
            
            with torch.no_grad():
                count = 0
                for cover, secret in test_loader:
                    if count >= num_samples:
                        break
                        
                    cover, secret = cover.to(self.device), secret.to(self.device)
                    stego, secret_recovered = self.model(cover, secret)
                    
                    for i in range(cover.size(0)):
                        if count >= num_samples:
                            break
                            
                        # Apply attack to stego image
                        stego_attacked = attack_func(stego[i:i+1])
                        secret_recovered_attacked = self.model.decode(stego_attacked)
                        
                        # Calculate metrics
                        secret_np = denormalize_image(secret[i]).permute(1, 2, 0).cpu().numpy()
                        recovered_np = denormalize_image(secret_recovered_attacked[0]).permute(1, 2, 0).cpu().numpy()
                        
                        psnr_vals.append(psnr(secret_np, recovered_np, data_range=1.0))
                        ssim_vals.append(ssim(secret_np, recovered_np, multichannel=True, data_range=1.0))
                        
                        count += 1
            
            robustness_results[attack_name] = {
                'psnr_mean': float(np.mean(psnr_vals)),
                'psnr_std': float(np.std(psnr_vals)),
                'ssim_mean': float(np.mean(ssim_vals)),
                'ssim_std': float(np.std(ssim_vals))
            }
            
            print(f"  {attack_name} - PSNR: {np.mean(psnr_vals):.2f}±{np.std(psnr_vals):.2f} dB, "
                  f"SSIM: {np.mean(ssim_vals):.4f}±{np.std(ssim_vals):.4f}")
        
        self.results['robustness'] = robustness_results
        
    def _jpeg_compression_attack(self, stego_tensor, quality=75):
        """Apply JPEG compression attack"""
        stego_img = denormalize_image(stego_tensor.squeeze(0)).permute(1, 2, 0).cpu().numpy()
        stego_img = (stego_img * 255).astype(np.uint8)
        
        # Convert to PIL and apply JPEG compression
        pil_img = Image.fromarray(stego_img)
        from io import BytesIO
        buffer = BytesIO()
        pil_img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        compressed_img = Image.open(buffer)
        
        # Convert back to tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        return transform(compressed_img).unsqueeze(0).to(self.device)
    
    def _gaussian_noise_attack(self, stego_tensor, std=0.05):
        """Add Gaussian noise"""
        noise = torch.randn_like(stego_tensor) * std
        return torch.clamp(stego_tensor + noise, -1, 1)
    
    def _gaussian_blur_attack(self, stego_tensor, kernel_size=3):
        """Apply Gaussian blur"""
        # Convert to PIL for blur
        stego_img = denormalize_image(stego_tensor.squeeze(0)).permute(1, 2, 0).cpu().numpy()
        stego_img = (stego_img * 255).astype(np.uint8)
        pil_img = Image.fromarray(stego_img)
        blurred_img = pil_img.filter(ImageFilter.GaussianBlur(radius=kernel_size//2))
        
        # Convert back to tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        return transform(blurred_img).unsqueeze(0).to(self.device)
    
    def _rotation_attack(self, stego_tensor, angle=5):
        """Apply rotation"""
        stego_img = denormalize_image(stego_tensor.squeeze(0)).permute(1, 2, 0).cpu().numpy()
        stego_img = (stego_img * 255).astype(np.uint8)
        
        # Apply rotation using OpenCV
        center = (stego_img.shape[1] // 2, stego_img.shape[0] // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(stego_img, M, (stego_img.shape[1], stego_img.shape[0]))
        
        # Convert back to tensor
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        return transform(rotated).unsqueeze(0).to(self.device)
    
    def _scaling_attack(self, stego_tensor, scale=0.9):
        """Apply scaling (resize and crop/pad)"""
        _, _, h, w = stego_tensor.shape
        new_size = int(h * scale)
        
        # Resize
        scaled = F.interpolate(stego_tensor, size=(new_size, new_size), mode='bilinear', align_corners=False)
        
        # Pad or crop back to original size
        if scale < 1.0:
            # Pad
            pad = (h - new_size) // 2
            scaled = F.pad(scaled, (pad, pad, pad, pad))
        else:
            # Crop
            crop = (new_size - h) // 2
            scaled = scaled[:, :, crop:crop+h, crop:crop+w]
        
        return scaled
    
    def _brightness_attack(self, stego_tensor, factor=0.2):
        """Apply brightness change"""
        return torch.clamp(stego_tensor + factor, -1, 1)
    
    def _contrast_attack(self, stego_tensor, factor=0.8):
        """Apply contrast change"""
        return torch.clamp(stego_tensor * factor, -1, 1)
    
    def evaluate_capacity(self, test_loader, num_samples=20):
        """Evaluate embedding capacity (bits per pixel)"""
        print("\nEvaluating embedding capacity...")
        
        total_bits = 0
        total_pixels = 0
        
        with torch.no_grad():
            count = 0
            for cover, secret in test_loader:
                if count >= num_samples:
                    break
                    
                cover, secret = cover.to(self.device), secret.to(self.device)
                
                # Calculate theoretical capacity (3 channels * 8 bits per channel)
                batch_size, channels, height, width = secret.shape
                bits_per_image = channels * height * width * 8
                total_bits += bits_per_image * batch_size
                total_pixels += height * width * batch_size
                
                count += batch_size
        
        bits_per_pixel = total_bits / total_pixels
        
        self.results['capacity'] = {
            'bits_per_pixel': float(bits_per_pixel),
            'total_samples': count
        }
        
        print(f"Theoretical embedding capacity: {bits_per_pixel:.2f} bits per pixel")
    
    def evaluate_visual_quality(self, test_loader, save_samples=True):
        """Evaluate visual quality with sample outputs"""
        print("\nEvaluating visual quality...")
        
        self.model.eval()
        
        if save_samples:
            os.makedirs('evaluation_results/visual_samples', exist_ok=True)
            
            with torch.no_grad():
                cover, secret = next(iter(test_loader))
                cover, secret = cover[:4].to(self.device), secret[:4].to(self.device)
                stego, secret_recovered = self.model(cover, secret)
                
                # Create comparison figure
                fig, axes = plt.subplots(4, 4, figsize=(16, 16))
                
                for i in range(4):
                    # Original images
                    cover_img = denormalize_image(cover[i]).permute(1, 2, 0).cpu().numpy()
                    secret_img = denormalize_image(secret[i]).permute(1, 2, 0).cpu().numpy()
                    stego_img = denormalize_image(stego[i]).permute(1, 2, 0).cpu().numpy()
                    recovered_img = denormalize_image(secret_recovered[i]).permute(1, 2, 0).cpu().numpy()
                    
                    axes[i, 0].imshow(cover_img)
                    axes[i, 0].set_title(f'Cover {i+1}')
                    axes[i, 0].axis('off')
                    
                    axes[i, 1].imshow(secret_img)
                    axes[i, 1].set_title(f'Secret {i+1}')
                    axes[i, 1].axis('off')
                    
                    axes[i, 2].imshow(stego_img)
                    axes[i, 2].set_title(f'Stego {i+1}')
                    axes[i, 2].axis('off')
                    
                    axes[i, 3].imshow(recovered_img)
                    axes[i, 3].set_title(f'Recovered {i+1}')
                    axes[i, 3].axis('off')
                
                plt.tight_layout()
                plt.savefig('evaluation_results/visual_samples/quality_comparison.png', 
                           dpi=150, bbox_inches='tight')
                plt.close()
                
                print("Visual quality samples saved to evaluation_results/visual_samples/")
    
    def generate_report(self, output_dir='evaluation_results'):
        """Generate comprehensive evaluation report"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save JSON report
        report_data = {
            'evaluation_date': datetime.now().isoformat(),
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'results': self.results
        }
        
        with open(os.path.join(output_dir, 'evaluation_report.json'), 'w') as f:
            json.dump(report_data, f, indent=2)
        
        # Generate markdown report
        md_report = self._generate_markdown_report()
        with open(os.path.join(output_dir, 'evaluation_report.md'), 'w') as f:
            f.write(md_report)
        
        # Save CSV with metrics
        self._save_csv_report(output_dir)
        
        print(f"\nEvaluation report saved to {output_dir}/")
    
    def _generate_markdown_report(self):
        """Generate markdown evaluation report"""
        report = f"""# Digital Watermarking Evaluation Report

**Evaluation Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Model Parameters:** {sum(p.numel() for p in self.model.parameters()):,}

## Basic Quality Metrics

"""
        
        if 'basic_metrics' in self.results:
            metrics = self.results['basic_metrics']
            report += f"""
### Cover Image Preservation
- **PSNR:** {metrics['cover_psnr_mean']:.2f} ± {metrics['cover_psnr_std']:.2f} dB
- **SSIM:** {metrics['cover_ssim_mean']:.4f} ± {metrics['cover_ssim_std']:.4f}

### Secret Image Recovery  
- **PSNR:** {metrics['secret_psnr_mean']:.2f} ± {metrics['secret_psnr_std']:.2f} dB
- **SSIM:** {metrics['secret_ssim_mean']:.4f} ± {metrics['secret_ssim_std']:.4f}

"""
        
        if 'robustness' in self.results:
            report += "## Robustness Against Attacks\n\n"
            report += "| Attack | PSNR (dB) | SSIM |\n"
            report += "|--------|-----------|------|\n"
            
            for attack_name, metrics in self.results['robustness'].items():
                report += f"| {attack_name.replace('_', ' ').title()} | "
                report += f"{metrics['psnr_mean']:.2f} ± {metrics['psnr_std']:.2f} | "
                report += f"{metrics['ssim_mean']:.4f} ± {metrics['ssim_std']:.4f} |\n"
        
        if 'capacity' in self.results:
            capacity = self.results['capacity']
            report += f"""
## Embedding Capacity

- **Bits per Pixel:** {capacity['bits_per_pixel']:.2f}
- **Samples Evaluated:** {capacity['total_samples']}

"""
        
        return report
    
    def _save_csv_report(self, output_dir):
        """Save detailed metrics in CSV format"""
        csv_path = os.path.join(output_dir, 'detailed_metrics.csv')
        
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write basic metrics
            if 'basic_metrics' in self.results:
                writer.writerow(['Metric Category', 'Metric', 'Mean', 'Std'])
                metrics = self.results['basic_metrics']
                writer.writerow(['Cover Quality', 'PSNR (dB)', metrics['cover_psnr_mean'], metrics['cover_psnr_std']])
                writer.writerow(['Cover Quality', 'SSIM', metrics['cover_ssim_mean'], metrics['cover_ssim_std']])
                writer.writerow(['Secret Recovery', 'PSNR (dB)', metrics['secret_psnr_mean'], metrics['secret_psnr_std']])
                writer.writerow(['Secret Recovery', 'SSIM', metrics['secret_ssim_mean'], metrics['secret_ssim_std']])
                
            # Write robustness metrics
            if 'robustness' in self.results:
                writer.writerow([])
                writer.writerow(['Attack', 'PSNR Mean', 'PSNR Std', 'SSIM Mean', 'SSIM Std'])
                for attack_name, metrics in self.results['robustness'].items():
                    writer.writerow([
                        attack_name,
                        metrics['psnr_mean'],
                        metrics['psnr_std'],
                        metrics['ssim_mean'],
                        metrics['ssim_std']
                    ])


def main():
    parser = argparse.ArgumentParser(description='Evaluate Steganography Model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--test_data', type=str, choices=['cifar', 'custom'], default='cifar', 
                       help='Test dataset to use')
    parser.add_argument('--custom_image_dir', type=str, default=None, 
                       help='Custom image directory (if test_data=custom)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--image_size', type=int, default=128, help='Image size')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples for evaluation')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='Output directory')
    parser.add_argument('--skip_robustness', action='store_true', help='Skip robustness testing')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
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
    
    print(f"Model loaded from {args.checkpoint}")
    
    # Create test data loader
    print("Creating test data loader...")
    _, test_loader = create_dataloader(
        batch_size=args.batch_size,
        image_size=args.image_size,
        use_cifar=(args.test_data == 'cifar'),
        custom_image_dir=args.custom_image_dir
    )
    
    # Create evaluator
    evaluator = SteganographyEvaluator(model, device)
    
    # Run evaluation
    print("\n" + "="*60)
    print("STARTING COMPREHENSIVE EVALUATION")
    print("="*60)
    
    # Basic metrics
    evaluator.evaluate_basic_metrics(test_loader, args.num_samples)
    
    # Capacity evaluation
    evaluator.evaluate_capacity(test_loader, min(20, args.num_samples))
    
    # Visual quality
    evaluator.evaluate_visual_quality(test_loader, save_samples=True)
    
    # Robustness testing
    if not args.skip_robustness:
        evaluator.evaluate_robustness(test_loader, min(50, args.num_samples))
    
    # Generate report
    evaluator.generate_report(args.output_dir)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETED")
    print("="*60)
    print(f"Results saved to: {args.output_dir}")
    

if __name__ == '__main__':
    main()
