#!/usr/bin/env python3
"""
Advanced optimization and features for Digital Watermarking system.
Includes model optimization, advanced training techniques, and performance enhancements.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
from datetime import datetime

from models import SteganographyAutoencoder, SteganographyLoss
from model_variants import get_model_variant
from dataset import create_dataloader


class AdvancedTrainer:
    """Enhanced trainer with advanced optimization techniques"""
    
    def __init__(self, model, train_loader, val_loader, device='cuda', config=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config or self._default_config()
        
        # Advanced loss function with perceptual loss
        self.criterion = AdvancedSteganographyLoss(
            alpha=self.config['loss_weights']['alpha'],
            beta=self.config['loss_weights']['beta'],
            gamma=self.config['loss_weights']['gamma'],
            perceptual_weight=self.config['loss_weights']['perceptual']
        )
        
        # Advanced optimizer with different LR for encoder/decoder
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Training state
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Gradient scaling for mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
        
        # Create directories
        os.makedirs('checkpoints', exist_ok=True)
        os.makedirs('optimized_results', exist_ok=True)
    
    def _default_config(self):
        """Default configuration for advanced training"""
        return {
            'optimizer': {
                'type': 'AdamW',
                'lr': 0.001,
                'weight_decay': 0.01,
                'betas': [0.9, 0.999]
            },
            'scheduler': {
                'type': 'CosineAnnealingLR',
                'T_max': 100,
                'eta_min': 1e-6
            },
            'loss_weights': {
                'alpha': 1.0,      # Cover loss weight
                'beta': 1.0,       # Secret loss weight  
                'gamma': 0.1,      # SSIM loss weight
                'perceptual': 0.1  # Perceptual loss weight
            },
            'training': {
                'mixed_precision': True,
                'gradient_clipping': 1.0,
                'early_stopping_patience': 15,
                'warmup_epochs': 5
            },
            'augmentation': {
                'enabled': True,
                'noise_std': 0.01,
                'brightness_range': 0.1,
                'contrast_range': 0.1
            }
        }
    
    def _setup_optimizer(self):
        """Setup advanced optimizer with parameter groups"""
        encoder_params = list(self.model.encoder.parameters())
        decoder_params = list(self.model.decoder.parameters())
        
        if self.config['optimizer']['type'] == 'AdamW':
            optimizer = optim.AdamW([
                {'params': encoder_params, 'lr': self.config['optimizer']['lr']},
                {'params': decoder_params, 'lr': self.config['optimizer']['lr'] * 0.8}  # Slightly lower LR for decoder
            ], 
            weight_decay=self.config['optimizer']['weight_decay'],
            betas=self.config['optimizer']['betas'])
        else:
            optimizer = optim.Adam([
                {'params': encoder_params, 'lr': self.config['optimizer']['lr']},
                {'params': decoder_params, 'lr': self.config['optimizer']['lr'] * 0.8}
            ])
        
        return optimizer
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        if self.config['scheduler']['type'] == 'CosineAnnealingLR':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['scheduler']['T_max'],
                eta_min=self.config['scheduler']['eta_min']
            )
        elif self.config['scheduler']['type'] == 'ReduceLROnPlateau':
            return ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
        else:
            return None
    
    def train_epoch(self, epoch):
        """Enhanced training epoch with advanced techniques"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        # Warmup learning rate
        if epoch < self.config['training']['warmup_epochs']:
            warmup_factor = (epoch + 1) / self.config['training']['warmup_epochs']
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * warmup_factor
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}')
        
        for batch_idx, (cover, secret) in enumerate(progress_bar):
            cover, secret = cover.to(self.device), secret.to(self.device)
            
            # Data augmentation
            if self.config['augmentation']['enabled']:
                cover, secret = self._augment_data(cover, secret)
            
            self.optimizer.zero_grad()
            
            # Mixed precision training
            if self.scaler is not None and self.config['training']['mixed_precision']:
                with torch.cuda.amp.autocast():
                    stego, secret_recovered = self.model(cover, secret)
                    loss_dict = self.criterion(cover, stego, secret, secret_recovered)
                    loss = loss_dict['total_loss']
                
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config['training']['gradient_clipping'] > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['training']['gradient_clipping']
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                stego, secret_recovered = self.model(cover, secret)
                loss_dict = self.criterion(cover, stego, secret, secret_recovered)
                loss = loss_dict['total_loss']
                
                loss.backward()
                
                # Gradient clipping
                if self.config['training']['gradient_clipping'] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['training']['gradient_clipping']
                    )
                
                self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg': f'{total_loss/num_batches:.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        return total_loss / num_batches
    
    def _augment_data(self, cover, secret):
        """Apply data augmentation"""
        if self.training:
            # Add noise
            if self.config['augmentation']['noise_std'] > 0:
                noise = torch.randn_like(cover) * self.config['augmentation']['noise_std']
                cover = cover + noise
                secret = secret + noise
            
            # Brightness adjustment
            if self.config['augmentation']['brightness_range'] > 0:
                brightness_factor = 1 + (torch.rand(1) - 0.5) * 2 * self.config['augmentation']['brightness_range']
                cover = cover * brightness_factor
                secret = secret * brightness_factor
            
            # Contrast adjustment  
            if self.config['augmentation']['contrast_range'] > 0:
                contrast_factor = 1 + (torch.rand(1) - 0.5) * 2 * self.config['augmentation']['contrast_range']
                cover = (cover - cover.mean()) * contrast_factor + cover.mean()
                secret = (secret - secret.mean()) * contrast_factor + secret.mean()
            
            # Clamp to valid range
            cover = torch.clamp(cover, -1, 1)
            secret = torch.clamp(secret, -1, 1)
        
        return cover, secret
    
    def validate_epoch(self, epoch):
        """Enhanced validation with detailed metrics"""
        self.model.eval()
        total_loss = 0
        metrics = {'cover_psnr': [], 'cover_ssim': [], 'secret_psnr': [], 'secret_ssim': []}
        
        with torch.no_grad():
            for cover, secret in tqdm(self.val_loader, desc='Validation'):
                cover, secret = cover.to(self.device), secret.to(self.device)
                
                stego, secret_recovered = self.model(cover, secret)
                loss_dict = self.criterion(cover, stego, secret, secret_recovered)
                total_loss += loss_dict['total_loss'].item()
                
                # Calculate detailed metrics
                batch_metrics = self._calculate_batch_metrics(cover, stego, secret, secret_recovered)
                for key, values in batch_metrics.items():
                    metrics[key].extend(values)
        
        avg_loss = total_loss / len(self.val_loader)
        
        # Calculate average metrics
        avg_metrics = {key: np.mean(values) for key, values in metrics.items()}
        
        print(f'Validation - Loss: {avg_loss:.4f}, '
              f'Cover PSNR: {avg_metrics["cover_psnr"]:.2f}, '
              f'Secret PSNR: {avg_metrics["secret_psnr"]:.2f}')
        
        return avg_loss, avg_metrics
    
    def _calculate_batch_metrics(self, cover, stego, secret, secret_recovered):
        """Calculate PSNR and SSIM for a batch"""
        from skimage.metrics import peak_signal_noise_ratio as psnr
        from skimage.metrics import structural_similarity as ssim
        from dataset import denormalize_image
        
        metrics = {'cover_psnr': [], 'cover_ssim': [], 'secret_psnr': [], 'secret_ssim': []}
        
        for i in range(cover.size(0)):
            # Convert to numpy
            cover_np = denormalize_image(cover[i]).permute(1, 2, 0).cpu().numpy()
            stego_np = denormalize_image(stego[i]).permute(1, 2, 0).cpu().numpy()
            secret_np = denormalize_image(secret[i]).permute(1, 2, 0).cpu().numpy()
            recovered_np = denormalize_image(secret_recovered[i]).permute(1, 2, 0).cpu().numpy()
            
            # Calculate metrics
            metrics['cover_psnr'].append(psnr(cover_np, stego_np, data_range=1.0))
            metrics['cover_ssim'].append(ssim(cover_np, stego_np, multichannel=True, data_range=1.0))
            metrics['secret_psnr'].append(psnr(secret_np, recovered_np, data_range=1.0))
            metrics['secret_ssim'].append(ssim(secret_np, recovered_np, multichannel=True, data_range=1.0))
        
        return metrics
    
    def train(self, num_epochs):
        """Enhanced training loop with advanced features"""
        print(f"Starting advanced training for {num_epochs} epochs...")
        print(f"Configuration: {json.dumps(self.config, indent=2)}")
        
        training_history = {
            'train_losses': [],
            'val_losses': [],
            'val_metrics': [],
            'learning_rates': []
        }
        
        for epoch in range(num_epochs):
            # Training
            train_loss = self.train_epoch(epoch)
            
            # Validation
            val_loss, val_metrics = self.validate_epoch(epoch)
            
            # Update learning rate scheduler
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Record history
            training_history['train_losses'].append(train_loss)
            training_history['val_losses'].append(val_loss)
            training_history['val_metrics'].append(val_metrics)
            training_history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                # Save best model
                self.save_checkpoint(epoch, is_best=True)
            else:
                self.patience_counter += 1
            
            # Check early stopping
            if self.patience_counter >= self.config['training']['early_stopping_patience']:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
            
            # Regular checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch)
        
        # Save training history
        self.save_training_history(training_history)
        
        print("Advanced training completed!")
        return training_history
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        filename = 'best_model.pth' if is_best else f'checkpoint_epoch_{epoch+1}.pth'
        torch.save(checkpoint, os.path.join('checkpoints', filename))
    
    def save_training_history(self, history):
        """Save training history and create plots"""
        # Save as JSON
        with open('optimized_results/training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        # Create plots
        self._plot_training_curves(history)
    
    def _plot_training_curves(self, history):
        """Create comprehensive training plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(history['train_losses'], label='Train Loss')
        axes[0, 0].plot(history['val_losses'], label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Learning rate
        axes[0, 1].plot(history['learning_rates'])
        axes[0, 1].set_title('Learning Rate Schedule')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True)
        
        # PSNR metrics
        cover_psnr = [m['cover_psnr'] for m in history['val_metrics']]
        secret_psnr = [m['secret_psnr'] for m in history['val_metrics']]
        axes[1, 0].plot(cover_psnr, label='Cover PSNR')
        axes[1, 0].plot(secret_psnr, label='Secret PSNR')
        axes[1, 0].set_title('PSNR Metrics')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('PSNR (dB)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # SSIM metrics
        cover_ssim = [m['cover_ssim'] for m in history['val_metrics']]
        secret_ssim = [m['secret_ssim'] for m in history['val_metrics']]
        axes[1, 1].plot(cover_ssim, label='Cover SSIM')
        axes[1, 1].plot(secret_ssim, label='Secret SSIM')
        axes[1, 1].set_title('SSIM Metrics')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('SSIM')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('optimized_results/training_curves.png', dpi=150, bbox_inches='tight')
        plt.close()


class AdvancedSteganographyLoss(nn.Module):
    """Enhanced loss function with perceptual loss"""
    
    def __init__(self, alpha=1.0, beta=1.0, gamma=0.1, perceptual_weight=0.1):
        super(AdvancedSteganographyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta  
        self.gamma = gamma
        self.perceptual_weight = perceptual_weight
        
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
        # VGG for perceptual loss
        if perceptual_weight > 0:
            self.vgg = self._load_vgg()
        
    def _load_vgg(self):
        """Load VGG network for perceptual loss"""
        import torchvision.models as models
        vgg = models.vgg16(pretrained=True).features[:16]  # Up to relu3_3
        vgg.eval()
        for param in vgg.parameters():
            param.requires_grad = False
        return vgg
    
    def _perceptual_loss(self, x, y):
        """Calculate perceptual loss using VGG features"""
        if self.perceptual_weight == 0:
            return 0
        
        # Normalize to ImageNet statistics
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        
        x_norm = (x * 0.5 + 0.5 - mean) / std
        y_norm = (y * 0.5 + 0.5 - mean) / std
        
        x_features = self.vgg(x_norm)
        y_features = self.vgg(y_norm)
        
        return self.mse_loss(x_features, y_features)
    
    def forward(self, cover, stego, secret, secret_recovered):
        # Basic losses
        cover_loss = self.mse_loss(cover, stego)
        secret_loss = self.mse_loss(secret, secret_recovered)
        
        # SSIM loss
        ssim_loss = 0
        if self.gamma > 0:
            ssim_loss = self._ssim_loss(cover, stego) + self._ssim_loss(secret, secret_recovered)
        
        # Perceptual loss
        perceptual_loss = 0
        if self.perceptual_weight > 0:
            perceptual_loss = (self._perceptual_loss(cover, stego) + 
                             self._perceptual_loss(secret, secret_recovered))
        
        # Total loss
        total_loss = (self.alpha * cover_loss + 
                     self.beta * secret_loss + 
                     self.gamma * ssim_loss +
                     self.perceptual_weight * perceptual_loss)
        
        return {
            'total_loss': total_loss,
            'cover_loss': cover_loss,
            'secret_loss': secret_loss,
            'ssim_loss': ssim_loss,
            'perceptual_loss': perceptual_loss
        }
    
    def _ssim_loss(self, x, y, window_size=11):
        """SSIM loss implementation"""
        def gaussian_window(size, sigma):
            coords = torch.arange(size, dtype=torch.float32)
            coords -= size // 2
            g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
            g /= g.sum()
            return g.view(1, 1, -1) * g.view(1, -1, 1)
        
        window = gaussian_window(window_size, 1.5).to(x.device)
        
        mu1 = F.conv2d(x, window, padding=window_size//2, groups=x.size(1))
        mu2 = F.conv2d(y, window, padding=window_size//2, groups=y.size(1))
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(x * x, window, padding=window_size//2, groups=x.size(1)) - mu1_sq
        sigma2_sq = F.conv2d(y * y, window, padding=window_size//2, groups=y.size(1)) - mu2_sq
        sigma12 = F.conv2d(x * y, window, padding=window_size//2, groups=x.size(1)) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return 1 - ssim_map.mean()


def optimize_model(model_path, optimization_config=None):
    """Main optimization function"""
    import argparse
    
    # Load configuration
    config = optimization_config or {}
    
    # Load data
    train_loader, val_loader = create_dataloader(
        batch_size=config.get('batch_size', 32),
        image_size=config.get('image_size', 128),
        use_cifar=config.get('use_cifar', True)
    )
    
    # Load or create model
    if model_path and os.path.exists(model_path):
        model = SteganographyAutoencoder()
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {model_path}")
    else:
        model = SteganographyAutoencoder()
        print("Created new model for optimization")
    
    # Create advanced trainer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = AdvancedTrainer(model, train_loader, val_loader, device, config)
    
    # Start training
    history = trainer.train(config.get('epochs', 100))
    
    return model, history


def main():
    """Main function for optimization script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced Optimization for Digital Watermarking')
    parser.add_argument('--model_path', type=str, help='Path to existing model checkpoint')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--config', type=str, help='Path to configuration JSON file')
    parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision training')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'optimizer': {'lr': args.lr},
            'training': {'mixed_precision': args.mixed_precision}
        }
    
    print("🚀 Starting Advanced Optimization")
    print("=" * 50)
    
    # Run optimization
    model, history = optimize_model(args.model_path, config)
    
    print("\n" + "=" * 50)
    print("Optimization completed successfully!")
    print("Check 'optimized_results/' for training curves and metrics.")


if __name__ == '__main__':
    main()
