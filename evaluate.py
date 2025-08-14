#!/usr/bin/env python3
"""
Evaluation script for our trained SteganographyAutoencoder
Uses CIFAR test set and the best model checkpoint from training
"""

import torch
from torch.utils.data import DataLoader
from datetime import datetime

from models import SteganographyAutoencoder
from dataset import CIFARPairDataset, denormalize_image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import json

# ===============================
# Evaluator Class
# ===============================
class SteganographyEvaluator:
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.results = {}

    def evaluate_basic_metrics(self, test_loader, num_samples=100):
        print("Evaluating basic metrics...")
        self.model.eval()
        psnr_cover, ssim_cover = [], []
        psnr_secret, ssim_secret = [], []

        with torch.no_grad():
            count = 0
            for cover, secret in tqdm(test_loader, desc="Basic metrics"):
                if count >= num_samples:
                    break
                cover, secret = cover.to(self.device), secret.to(self.device)
                stego, secret_rec = self.model(cover, secret)
                for i in range(cover.size(0)):
                    if count >= num_samples:
                        break
                    c_np = denormalize_image(cover[i]).permute(1, 2, 0).cpu().numpy()
                    s_np = denormalize_image(secret[i]).permute(1, 2, 0).cpu().numpy()
                    st_np = denormalize_image(stego[i]).permute(1, 2, 0).cpu().numpy()
                    sr_np = denormalize_image(secret_rec[i]).permute(1, 2, 0).cpu().numpy()

                    psnr_cover.append(psnr(c_np, st_np, data_range=1.0))
                    ssim_cover.append(ssim(c_np, st_np, channel_axis=2, data_range=1.0))
                    psnr_secret.append(psnr(s_np, sr_np, data_range=1.0))
                    ssim_secret.append(ssim(s_np, sr_np, channel_axis=2, data_range=1.0))
                    count += 1

        self.results['basic_metrics'] = {
            'cover_psnr_mean': float(np.mean(psnr_cover)),
            'cover_psnr_std': float(np.std(psnr_cover)),
            'cover_ssim_mean': float(np.mean(ssim_cover)),
            'cover_ssim_std': float(np.std(ssim_cover)),
            'secret_psnr_mean': float(np.mean(psnr_secret)),
            'secret_psnr_std': float(np.std(psnr_secret)),
            'secret_ssim_mean': float(np.mean(ssim_secret)),
            'secret_ssim_std': float(np.std(ssim_secret)),
        }

        print(f"Cover Image - PSNR: {np.mean(psnr_cover):.2f} ± {np.std(psnr_cover):.2f} dB")
        print(f"Cover Image - SSIM: {np.mean(ssim_cover):.4f} ± {np.std(ssim_cover):.4f}")
        print(f"Secret Image - PSNR: {np.mean(psnr_secret):.2f} ± {np.std(psnr_secret):.2f} dB")
        print(f"Secret Image - SSIM: {np.mean(ssim_secret):.4f} ± {np.std(ssim_secret):.4f}")

    def evaluate_capacity(self, test_loader, num_samples=20):
        print("\nEvaluating capacity...")
        total_bits, total_pixels = 0, 0
        with torch.no_grad():
            count = 0
            for _, secret in test_loader:
                if count >= num_samples:
                    break
                batch_size, channels, height, width = secret.shape
                total_bits += channels * height * width * 8 * batch_size
                total_pixels += height * width * batch_size
                count += batch_size
        bpp = total_bits / total_pixels
        self.results['capacity'] = {
            'bits_per_pixel': float(bpp),
            'total_samples': count
        }
        print(f"Embedding capacity: {bpp:.2f} bits per pixel")

    def evaluate_visual_quality(self, test_loader, save_dir='evaluation_results/visual_samples'):
        print("\nSaving visual quality samples...")
        os.makedirs(save_dir, exist_ok=True)
        self.model.eval()
        with torch.no_grad():
            cover, secret = next(iter(test_loader))
            cover, secret = cover[:4].to(self.device), secret[:4].to(self.device)
            stego, secret_rec = self.model(cover, secret)
            fig, axes = plt.subplots(4, 4, figsize=(16, 16))
            for i in range(4):
                imgs = [
                    denormalize_image(cover[i]).permute(1, 2, 0).cpu().numpy(),
                    denormalize_image(secret[i]).permute(1, 2, 0).cpu().numpy(),
                    denormalize_image(stego[i]).permute(1, 2, 0).cpu().numpy(),
                    denormalize_image(secret_rec[i]).permute(1, 2, 0).cpu().numpy()
                ]
                titles = ['Cover', 'Secret', 'Stego', 'Recovered']
                for j in range(4):
                    axes[i, j].imshow(imgs[j])
                    axes[i, j].set_title(titles[j])
                    axes[i, j].axis('off')
            plt.tight_layout()
            out_path = os.path.join(save_dir, 'quality_comparison.png')
            plt.savefig(out_path, dpi=150)
            plt.close()
            print(f"Saved visual comparison to {out_path}")

    def generate_report(self, output_dir='evaluation_results'):
        os.makedirs(output_dir, exist_ok=True)
        report = {
            'evaluation_date': datetime.now().isoformat(),
            'model_params': sum(p.numel() for p in self.model.parameters()),
            'results': self.results
        }
        with open(os.path.join(output_dir, 'evaluation_report.json'), 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {output_dir}/evaluation_report.json")


# ===============================
# Main
# ===============================
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model (match training architecture)
    model = SteganographyAutoencoder(in_channels=3, hidden_dims=[64, 128, 256])
    checkpoint_path = "cuda_safe_checkpoints/best_model.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()
    print(f"Loaded model from {checkpoint_path}")

    # Create CIFAR test loader (match training image size)
    test_dataset = CIFARPairDataset(train=False, image_size=64)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    # Run evaluation
    evaluator = SteganographyEvaluator(model, device)
    evaluator.evaluate_basic_metrics(test_loader, num_samples=100)
    evaluator.evaluate_capacity(test_loader, num_samples=20)
    evaluator.evaluate_visual_quality(test_loader)
    evaluator.generate_report()
    print("\nEVALUATION COMPLETE.")
