#!/usr/bin/env python3
"""
Batch processing script for Digital Watermarking system.
Handles multiple image pairs for embedding and extraction operations.
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import json
from tqdm import tqdm
import glob
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from models import SteganographyAutoencoder
from dataset import denormalize_image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


class BatchProcessor:
    """Batch processor for steganography operations"""
    
    def __init__(self, model_path, device='cuda', image_size=128):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Load model
        self.model = SteganographyAutoencoder(
            in_channels=3,
            hidden_dims=[64, 128, 256, 512]
        )
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Thread lock for GPU operations
        self.gpu_lock = threading.Lock()
        
        print(f"Model loaded on {self.device}")
        print(f"Image processing size: {image_size}x{image_size}")
    
    def load_image(self, image_path):
        """Load and preprocess a single image"""
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0)
            return image_tensor, True
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return None, False
    
    def save_image(self, tensor, filepath, original_size=None):
        """Save tensor as image, optionally resize to original dimensions"""
        image = denormalize_image(tensor.squeeze(0))
        image = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(image)
        
        if original_size:
            pil_image = pil_image.resize(original_size, Image.LANCZOS)
        
        pil_image.save(filepath)
    
    def process_single_pair(self, cover_path, secret_path, output_dir, save_individual=True):
        """Process a single cover-secret image pair"""
        try:
            # Load images
            cover_tensor, cover_ok = self.load_image(cover_path)
            secret_tensor, secret_ok = self.load_image(secret_path)
            
            if not (cover_ok and secret_ok):
                return None
            
            # Get original sizes for later restoration
            cover_original = Image.open(cover_path)
            secret_original = Image.open(secret_path)
            cover_size = cover_original.size
            secret_size = secret_original.size
            
            # Perform steganography with GPU lock
            with self.gpu_lock:
                cover_tensor = cover_tensor.to(self.device)
                secret_tensor = secret_tensor.to(self.device)
                
                with torch.no_grad():
                    stego, secret_recovered = self.model(cover_tensor, secret_tensor)
                
                # Move back to CPU
                stego = stego.cpu()
                secret_recovered = secret_recovered.cpu()
                cover_tensor = cover_tensor.cpu()
                secret_tensor = secret_tensor.cpu()
            
            # Calculate metrics
            cover_np = denormalize_image(cover_tensor.squeeze(0)).permute(1, 2, 0).numpy()
            stego_np = denormalize_image(stego.squeeze(0)).permute(1, 2, 0).numpy()
            secret_np = denormalize_image(secret_tensor.squeeze(0)).permute(1, 2, 0).numpy()
            recovered_np = denormalize_image(secret_recovered.squeeze(0)).permute(1, 2, 0).numpy()
            
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
            
            # Generate output filenames
            cover_name = os.path.splitext(os.path.basename(cover_path))[0]
            secret_name = os.path.splitext(os.path.basename(secret_path))[0]
            pair_name = f"{cover_name}_X_{secret_name}"
            
            results = {
                'cover_path': cover_path,
                'secret_path': secret_path,
                'pair_name': pair_name,
                'cover_psnr': float(cover_psnr),
                'cover_ssim': float(cover_ssim),
                'secret_psnr': float(secret_psnr),
                'secret_ssim': float(secret_ssim),
                'cover_size': cover_size,
                'secret_size': secret_size
            }
            
            if save_individual:
                # Save stego image
                stego_path = os.path.join(output_dir, f"stego_{pair_name}.png")
                self.save_image(stego, stego_path, cover_size)
                results['stego_path'] = stego_path
                
                # Save recovered secret
                recovered_path = os.path.join(output_dir, f"recovered_{pair_name}.png")
                self.save_image(secret_recovered, recovered_path, secret_size)
                results['recovered_path'] = recovered_path
            
            return results
            
        except Exception as e:
            print(f"Error processing pair {cover_path} + {secret_path}: {e}")
            return None
    
    def batch_embed(self, cover_dir, secret_dir, output_dir, max_workers=4, pair_strategy='sequential'):
        """
        Batch embedding operation
        
        Args:
            cover_dir: Directory containing cover images
            secret_dir: Directory containing secret images
            output_dir: Output directory for results
            max_workers: Number of parallel workers
            pair_strategy: 'sequential', 'random', or 'all_combinations'
        """
        print(f"Starting batch embedding...")
        print(f"Cover directory: {cover_dir}")
        print(f"Secret directory: {secret_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Pairing strategy: {pair_strategy}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Get image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        cover_files = []
        secret_files = []
        
        for ext in image_extensions:
            cover_files.extend(glob.glob(os.path.join(cover_dir, ext)))
            cover_files.extend(glob.glob(os.path.join(cover_dir, ext.upper())))
            secret_files.extend(glob.glob(os.path.join(secret_dir, ext)))
            secret_files.extend(glob.glob(os.path.join(secret_dir, ext.upper())))
        
        cover_files.sort()
        secret_files.sort()
        
        print(f"Found {len(cover_files)} cover images and {len(secret_files)} secret images")
        
        # Create image pairs based on strategy
        pairs = self._create_image_pairs(cover_files, secret_files, pair_strategy)
        print(f"Created {len(pairs)} image pairs")
        
        # Process pairs
        results = []
        failed_pairs = []
        
        if max_workers == 1:
            # Sequential processing
            for cover_path, secret_path in tqdm(pairs, desc="Processing pairs"):
                result = self.process_single_pair(cover_path, secret_path, output_dir)
                if result:
                    results.append(result)
                else:
                    failed_pairs.append((cover_path, secret_path))
        else:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self.process_single_pair, cover_path, secret_path, output_dir): (cover_path, secret_path)
                    for cover_path, secret_path in pairs
                }
                
                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing pairs"):
                    cover_path, secret_path = futures[future]
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                        else:
                            failed_pairs.append((cover_path, secret_path))
                    except Exception as e:
                        print(f"Error in future for {cover_path} + {secret_path}: {e}")
                        failed_pairs.append((cover_path, secret_path))
        
        # Save batch report
        self._save_batch_report(results, failed_pairs, output_dir)
        
        print(f"\nBatch processing completed!")
        print(f"Successfully processed: {len(results)} pairs")
        print(f"Failed: {len(failed_pairs)} pairs")
        print(f"Results saved to: {output_dir}")
        
        return results, failed_pairs
    
    def batch_extract(self, stego_dir, output_dir, max_workers=4):
        """Batch extraction of secret images from stego images"""
        print(f"Starting batch extraction...")
        print(f"Stego directory: {stego_dir}")
        print(f"Output directory: {output_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Get stego image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        stego_files = []
        
        for ext in image_extensions:
            stego_files.extend(glob.glob(os.path.join(stego_dir, ext)))
            stego_files.extend(glob.glob(os.path.join(stego_dir, ext.upper())))
        
        stego_files.sort()
        print(f"Found {len(stego_files)} stego images")
        
        # Process files
        results = []
        failed_files = []
        
        def extract_single(stego_path):
            try:
                stego_tensor, stego_ok = self.load_image(stego_path)
                if not stego_ok:
                    return None
                
                # Get original size
                stego_original = Image.open(stego_path)
                stego_size = stego_original.size
                
                # Extract secret
                with self.gpu_lock:
                    stego_tensor = stego_tensor.to(self.device)
                    
                    with torch.no_grad():
                        secret_recovered = self.model.decode(stego_tensor)
                    
                    secret_recovered = secret_recovered.cpu()
                
                # Save extracted secret
                filename = os.path.splitext(os.path.basename(stego_path))[0]
                output_path = os.path.join(output_dir, f"extracted_{filename}.png")
                self.save_image(secret_recovered, output_path, stego_size)
                
                return {
                    'stego_path': stego_path,
                    'extracted_path': output_path,
                    'original_size': stego_size
                }
                
            except Exception as e:
                print(f"Error extracting from {stego_path}: {e}")
                return None
        
        if max_workers == 1:
            # Sequential processing
            for stego_path in tqdm(stego_files, desc="Extracting secrets"):
                result = extract_single(stego_path)
                if result:
                    results.append(result)
                else:
                    failed_files.append(stego_path)
        else:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(extract_single, stego_path): stego_path for stego_path in stego_files}
                
                for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting secrets"):
                    stego_path = futures[future]
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                        else:
                            failed_files.append(stego_path)
                    except Exception as e:
                        print(f"Error in future for {stego_path}: {e}")
                        failed_files.append(stego_path)
        
        # Save extraction report
        extraction_report = {
            'extraction_date': datetime.now().isoformat(),
            'total_processed': len(stego_files),
            'successful_extractions': len(results),
            'failed_extractions': len(failed_files),
            'results': results,
            'failed_files': failed_files
        }
        
        with open(os.path.join(output_dir, 'extraction_report.json'), 'w') as f:
            json.dump(extraction_report, f, indent=2)
        
        print(f"\nBatch extraction completed!")
        print(f"Successfully extracted: {len(results)} images")
        print(f"Failed: {len(failed_files)} images")
        print(f"Results saved to: {output_dir}")
        
        return results, failed_files
    
    def _create_image_pairs(self, cover_files, secret_files, strategy):
        """Create image pairs based on the specified strategy"""
        pairs = []
        
        if strategy == 'sequential':
            # Pair images sequentially
            min_len = min(len(cover_files), len(secret_files))
            pairs = [(cover_files[i], secret_files[i]) for i in range(min_len)]
            
        elif strategy == 'random':
            # Random pairing (same number of pairs as minimum length)
            import random
            min_len = min(len(cover_files), len(secret_files))
            cover_sample = random.sample(cover_files, min_len)
            secret_sample = random.sample(secret_files, min_len)
            pairs = list(zip(cover_sample, secret_sample))
            
        elif strategy == 'all_combinations':
            # All possible combinations
            for cover in cover_files:
                for secret in secret_files:
                    pairs.append((cover, secret))
        
        return pairs
    
    def _save_batch_report(self, results, failed_pairs, output_dir):
        """Save comprehensive batch processing report"""
        # Calculate statistics
        if results:
            cover_psnr_values = [r['cover_psnr'] for r in results]
            cover_ssim_values = [r['cover_ssim'] for r in results]
            secret_psnr_values = [r['secret_psnr'] for r in results]
            secret_ssim_values = [r['secret_ssim'] for r in results]
            
            stats = {
                'cover_psnr_mean': float(np.mean(cover_psnr_values)),
                'cover_psnr_std': float(np.std(cover_psnr_values)),
                'cover_ssim_mean': float(np.mean(cover_ssim_values)),
                'cover_ssim_std': float(np.std(cover_ssim_values)),
                'secret_psnr_mean': float(np.mean(secret_psnr_values)),
                'secret_psnr_std': float(np.std(secret_psnr_values)),
                'secret_ssim_mean': float(np.mean(secret_ssim_values)),
                'secret_ssim_std': float(np.std(secret_ssim_values))
            }
        else:
            stats = {}
        
        # Create report
        report = {
            'processing_date': datetime.now().isoformat(),
            'total_pairs': len(results) + len(failed_pairs),
            'successful_pairs': len(results),
            'failed_pairs': len(failed_pairs),
            'statistics': stats,
            'results': results,
            'failed_pairs': [{'cover': pair[0], 'secret': pair[1]} for pair in failed_pairs]
        }
        
        # Save JSON report
        with open(os.path.join(output_dir, 'batch_report.json'), 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save CSV summary
        import csv
        csv_path = os.path.join(output_dir, 'batch_summary.csv')
        with open(csv_path, 'w', newline='') as csvfile:
            if results:
                fieldnames = ['pair_name', 'cover_psnr', 'cover_ssim', 'secret_psnr', 'secret_ssim']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for result in results:
                    writer.writerow({
                        'pair_name': result['pair_name'],
                        'cover_psnr': result['cover_psnr'],
                        'cover_ssim': result['cover_ssim'],
                        'secret_psnr': result['secret_psnr'],
                        'secret_ssim': result['secret_ssim']
                    })
        
        # Create visualization
        if results:
            self._create_batch_visualization(results, output_dir)
    
    def _create_batch_visualization(self, results, output_dir):
        """Create visualization of batch processing results"""
        cover_psnr = [r['cover_psnr'] for r in results]
        cover_ssim = [r['cover_ssim'] for r in results]
        secret_psnr = [r['secret_psnr'] for r in results]
        secret_ssim = [r['secret_ssim'] for r in results]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Cover PSNR histogram
        axes[0, 0].hist(cover_psnr, bins=20, alpha=0.7, color='blue')
        axes[0, 0].set_title(f'Cover PSNR Distribution\nMean: {np.mean(cover_psnr):.2f} dB')
        axes[0, 0].set_xlabel('PSNR (dB)')
        axes[0, 0].set_ylabel('Frequency')
        
        # Cover SSIM histogram
        axes[0, 1].hist(cover_ssim, bins=20, alpha=0.7, color='green')
        axes[0, 1].set_title(f'Cover SSIM Distribution\nMean: {np.mean(cover_ssim):.4f}')
        axes[0, 1].set_xlabel('SSIM')
        axes[0, 1].set_ylabel('Frequency')
        
        # Secret PSNR histogram
        axes[1, 0].hist(secret_psnr, bins=20, alpha=0.7, color='red')
        axes[1, 0].set_title(f'Secret PSNR Distribution\nMean: {np.mean(secret_psnr):.2f} dB')
        axes[1, 0].set_xlabel('PSNR (dB)')
        axes[1, 0].set_ylabel('Frequency')
        
        # Secret SSIM histogram
        axes[1, 1].hist(secret_ssim, bins=20, alpha=0.7, color='orange')
        axes[1, 1].set_title(f'Secret SSIM Distribution\nMean: {np.mean(secret_ssim):.4f}')
        axes[1, 1].set_xlabel('SSIM')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'batch_statistics.png'), dpi=150, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Batch Processing for Digital Watermarking')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--mode', type=str, choices=['embed', 'extract'], required=True, 
                       help='Processing mode')
    
    # Embedding arguments
    parser.add_argument('--cover_dir', type=str, help='Directory with cover images (for embed mode)')
    parser.add_argument('--secret_dir', type=str, help='Directory with secret images (for embed mode)')
    parser.add_argument('--pair_strategy', type=str, choices=['sequential', 'random', 'all_combinations'],
                       default='sequential', help='Image pairing strategy')
    
    # Extraction arguments
    parser.add_argument('--stego_dir', type=str, help='Directory with stego images (for extract mode)')
    
    # Common arguments
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--image_size', type=int, default=128, help='Processing image size')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--max_workers', type=int, default=4, help='Number of parallel workers')
    
    args = parser.parse_args()
    
    # Create batch processor
    processor = BatchProcessor(args.checkpoint, args.device, args.image_size)
    
    if args.mode == 'embed':
        if not args.cover_dir or not args.secret_dir:
            parser.error("embed mode requires --cover_dir and --secret_dir")
        
        results, failed = processor.batch_embed(
            args.cover_dir,
            args.secret_dir,
            args.output_dir,
            args.max_workers,
            args.pair_strategy
        )
        
    elif args.mode == 'extract':
        if not args.stego_dir:
            parser.error("extract mode requires --stego_dir")
        
        results, failed = processor.batch_extract(
            args.stego_dir,
            args.output_dir,
            args.max_workers
        )


if __name__ == '__main__':
    main()
