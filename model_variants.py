#!/usr/bin/env python3
"""
Alternative model architectures for different use cases.
Includes lightweight, high-capacity, and robust variants.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models import SteganographyLoss


class LightweightEncoder(nn.Module):
    """Lightweight encoder for fast inference on mobile/edge devices"""
    
    def __init__(self, in_channels=6, out_channels=3, hidden_dims=[32, 64, 128]):
        super(LightweightEncoder, self).__init__()
        
        self.hidden_dims = hidden_dims
        
        # Depthwise separable convolutions for efficiency
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dims[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.ReLU(inplace=True)
        )
        
        # Encoder with depthwise separable convolutions
        self.encoder_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.encoder_layers.append(self._depthwise_separable_conv(
                hidden_dims[i], hidden_dims[i+1], stride=2
            ))
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1, 0, -1):
            self.decoder_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i-1], 
                                     kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(hidden_dims[i-1]),
                    nn.ReLU(inplace=True)
                )
            )
        
        self.final_conv = nn.Conv2d(hidden_dims[0], out_channels, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()
        
    def _depthwise_separable_conv(self, in_channels, out_channels, stride=1):
        """Depthwise separable convolution for efficiency"""
        return nn.Sequential(
            # Depthwise convolution
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, 
                     padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            
            # Pointwise convolution
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, cover, secret):
        x = torch.cat([cover, secret], dim=1)
        x = self.initial_conv(x)
        
        # Store features for skip connections
        features = [x]
        for layer in self.encoder_layers:
            x = layer(x)
            features.append(x)
        
        # Decoder with skip connections
        for i, layer in enumerate(self.decoder_layers):
            x = layer(x)
            if i < len(features) - 1:
                x = x + features[-(i+2)]
        
        return self.tanh(self.final_conv(x))


class LightweightDecoder(nn.Module):
    """Lightweight decoder for fast secret extraction"""
    
    def __init__(self, in_channels=3, out_channels=3, hidden_dims=[32, 64, 128]):
        super(LightweightDecoder, self).__init__()
        
        self.hidden_dims = hidden_dims
        
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dims[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.ReLU(inplace=True)
        )
        
        # Encoder with depthwise separable convolutions
        self.encoder_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.encoder_layers.append(self._depthwise_separable_conv(
                hidden_dims[i], hidden_dims[i+1], stride=2
            ))
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1, 0, -1):
            self.decoder_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i-1], 
                                     kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(hidden_dims[i-1]),
                    nn.ReLU(inplace=True)
                )
            )
        
        self.final_conv = nn.Conv2d(hidden_dims[0], out_channels, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()
        
    def _depthwise_separable_conv(self, in_channels, out_channels, stride=1):
        """Depthwise separable convolution for efficiency"""
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, 
                     padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, stego):
        x = self.initial_conv(stego)
        
        features = [x]
        for layer in self.encoder_layers:
            x = layer(x)
            features.append(x)
        
        for i, layer in enumerate(self.decoder_layers):
            x = layer(x)
            if i < len(features) - 1:
                x = x + features[-(i+2)]
        
        return self.tanh(self.final_conv(x))


class HighCapacityEncoder(nn.Module):
    """High-capacity encoder for hiding larger secrets or multiple images"""
    
    def __init__(self, in_channels=6, out_channels=3, hidden_dims=[128, 256, 512, 1024]):
        super(HighCapacityEncoder, self).__init__()
        
        self.hidden_dims = hidden_dims
        
        # Enhanced initial processing
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dims[0], kernel_size=7, padding=3),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dims[0], hidden_dims[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks for better feature learning
        self.encoder_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.encoder_layers.append(
                nn.Sequential(
                    self._residual_block(hidden_dims[i], hidden_dims[i]),
                    nn.Conv2d(hidden_dims[i], hidden_dims[i+1], kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(hidden_dims[i+1]),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Attention mechanism for better feature integration
        self.attention = nn.MultiheadAttention(hidden_dims[-1], num_heads=8, batch_first=True)
        
        # Decoder layers with residual connections
        self.decoder_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1, 0, -1):
            self.decoder_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i-1], 
                                     kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(hidden_dims[i-1]),
                    nn.ReLU(inplace=True),
                    self._residual_block(hidden_dims[i-1], hidden_dims[i-1])
                )
            )
        
        # Enhanced final processing
        self.final_layers = nn.Sequential(
            nn.Conv2d(hidden_dims[0], hidden_dims[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dims[0], out_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
    def _residual_block(self, channels, out_channels):
        """Residual block for better gradient flow"""
        return nn.Sequential(
            nn.Conv2d(channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, cover, secret):
        x = torch.cat([cover, secret], dim=1)
        x = self.initial_conv(x)
        
        # Store features for skip connections
        features = [x]
        for layer in self.encoder_layers:
            x = layer(x)
            features.append(x)
        
        # Apply attention at the bottleneck
        b, c, h, w = x.shape
        x_flat = x.view(b, c, h*w).transpose(1, 2)
        x_att, _ = self.attention(x_flat, x_flat, x_flat)
        x = x_att.transpose(1, 2).view(b, c, h, w) + x
        
        # Decoder with skip connections and residual learning
        for i, layer in enumerate(self.decoder_layers):
            x = layer(x)
            if i < len(features) - 1:
                skip = features[-(i+2)]
                x = x + skip
        
        return self.final_layers(x)


class RobustEncoder(nn.Module):
    """Robust encoder designed to resist common attacks"""
    
    def __init__(self, in_channels=6, out_channels=3, hidden_dims=[64, 128, 256, 512]):
        super(RobustEncoder, self).__init__()
        
        self.hidden_dims = hidden_dims
        
        # Noise-resistant initial processing
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dims[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)  # Add dropout for robustness
        )
        
        # Encoder with noise injection during training
        self.encoder_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.encoder_layers.append(
                nn.Sequential(
                    nn.Conv2d(hidden_dims[i], hidden_dims[i+1], kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(hidden_dims[i+1]),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(0.1)
                )
            )
        
        # Robust decoder
        self.decoder_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1, 0, -1):
            self.decoder_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i-1], 
                                     kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(hidden_dims[i-1]),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(0.1)
                )
            )
        
        # Final robust processing
        self.final_conv = nn.Sequential(
            nn.Conv2d(hidden_dims[0], hidden_dims[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dims[0], out_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, cover, secret):
        x = torch.cat([cover, secret], dim=1)
        
        # Add noise during training for robustness
        if self.training:
            noise = torch.randn_like(x) * 0.01
            x = x + noise
        
        x = self.initial_conv(x)
        
        features = [x]
        for layer in self.encoder_layers:
            x = layer(x)
            features.append(x)
        
        for i, layer in enumerate(self.decoder_layers):
            x = layer(x)
            if i < len(features) - 1:
                x = x + features[-(i+2)]
        
        return self.final_conv(x)


class LightweightSteganography(nn.Module):
    """Complete lightweight model for mobile deployment"""
    
    def __init__(self, in_channels=3, hidden_dims=[32, 64, 128]):
        super(LightweightSteganography, self).__init__()
        
        self.encoder = LightweightEncoder(
            in_channels=in_channels*2, 
            out_channels=in_channels, 
            hidden_dims=hidden_dims
        )
        self.decoder = LightweightDecoder(
            in_channels=in_channels, 
            out_channels=in_channels, 
            hidden_dims=hidden_dims
        )
        
    def forward(self, cover, secret):
        stego = self.encoder(cover, secret)
        secret_recovered = self.decoder(stego)
        return stego, secret_recovered
    
    def encode(self, cover, secret):
        return self.encoder(cover, secret)
    
    def decode(self, stego):
        return self.decoder(stego)


class HighCapacitySteganography(nn.Module):
    """High-capacity model for complex steganography tasks"""
    
    def __init__(self, in_channels=3, hidden_dims=[128, 256, 512, 1024]):
        super(HighCapacitySteganography, self).__init__()
        
        self.encoder = HighCapacityEncoder(
            in_channels=in_channels*2, 
            out_channels=in_channels, 
            hidden_dims=hidden_dims
        )
        # Use same decoder architecture but with larger capacity
        from models import Decoder
        self.decoder = Decoder(
            in_channels=in_channels, 
            out_channels=in_channels, 
            hidden_dims=hidden_dims
        )
        
    def forward(self, cover, secret):
        stego = self.encoder(cover, secret)
        secret_recovered = self.decoder(stego)
        return stego, secret_recovered
    
    def encode(self, cover, secret):
        return self.encoder(cover, secret)
    
    def decode(self, stego):
        return self.decoder(stego)


class RobustSteganography(nn.Module):
    """Robust model designed to resist attacks"""
    
    def __init__(self, in_channels=3, hidden_dims=[64, 128, 256, 512]):
        super(RobustSteganography, self).__init__()
        
        self.encoder = RobustEncoder(
            in_channels=in_channels*2, 
            out_channels=in_channels, 
            hidden_dims=hidden_dims
        )
        # Use robust decoder (same as encoder architecture)
        self.decoder = RobustEncoder(
            in_channels=in_channels, 
            out_channels=in_channels, 
            hidden_dims=hidden_dims
        )
        
    def forward(self, cover, secret):
        stego = self.encoder(cover, secret)
        secret_recovered = self.decoder(stego)
        return stego, secret_recovered
    
    def encode(self, cover, secret):
        return self.encoder(cover, secret)
    
    def decode(self, stego):
        return self.decoder(stego)


class AdaptiveSteganography(nn.Module):
    """Adaptive model that can switch between different modes"""
    
    def __init__(self, in_channels=3):
        super(AdaptiveSteganography, self).__init__()
        
        # Multiple encoders for different scenarios
        self.lightweight_model = LightweightSteganography(in_channels, [32, 64, 128])
        self.standard_model = None  # Will be loaded from original model
        self.robust_model = RobustSteganography(in_channels, [64, 128, 256, 512])
        
        self.mode = 'standard'  # 'lightweight', 'standard', 'robust'
        
    def set_mode(self, mode):
        """Set the operating mode"""
        if mode in ['lightweight', 'standard', 'robust']:
            self.mode = mode
        else:
            raise ValueError("Mode must be 'lightweight', 'standard', or 'robust'")
    
    def forward(self, cover, secret):
        if self.mode == 'lightweight':
            return self.lightweight_model(cover, secret)
        elif self.mode == 'robust':
            return self.robust_model(cover, secret)
        else:  # standard
            if self.standard_model is None:
                # Fallback to lightweight if standard not loaded
                return self.lightweight_model(cover, secret)
            return self.standard_model(cover, secret)
    
    def encode(self, cover, secret):
        stego, _ = self.forward(cover, secret)
        return stego
    
    def decode(self, stego):
        if self.mode == 'lightweight':
            return self.lightweight_model.decode(stego)
        elif self.mode == 'robust':
            return self.robust_model.decode(stego)
        else:
            if self.standard_model is None:
                return self.lightweight_model.decode(stego)
            return self.standard_model.decode(stego)


def get_model_variant(variant_name, **kwargs):
    """Factory function to create model variants"""
    
    variants = {
        'lightweight': LightweightSteganography,
        'high_capacity': HighCapacitySteganography,
        'robust': RobustSteganography,
        'adaptive': AdaptiveSteganography
    }
    
    if variant_name not in variants:
        raise ValueError(f"Unknown variant '{variant_name}'. Available: {list(variants.keys())}")
    
    return variants[variant_name](**kwargs)


def compare_model_sizes():
    """Compare the parameter counts of different model variants"""
    
    models = {
        'Lightweight': LightweightSteganography(in_channels=3, hidden_dims=[32, 64, 128]),
        'Standard': None,  # Would need to import from models.py
        'High Capacity': HighCapacitySteganography(in_channels=3, hidden_dims=[128, 256, 512, 1024]),
        'Robust': RobustSteganography(in_channels=3, hidden_dims=[64, 128, 256, 512])
    }
    
    print("Model Comparison:")
    print("-" * 50)
    
    for name, model in models.items():
        if model is not None:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"{name}:")
            print(f"  Total parameters: {total_params:,}")
            print(f"  Trainable parameters: {trainable_params:,}")
            print(f"  Model size (MB): {total_params * 4 / 1024 / 1024:.2f}")
            print()


if __name__ == '__main__':
    # Example usage and comparison
    print("Digital Watermarking Model Variants")
    print("=" * 50)
    
    # Compare model sizes
    compare_model_sizes()
    
    # Test lightweight model
    print("Testing Lightweight Model:")
    lightweight_model = get_model_variant('lightweight', in_channels=3, hidden_dims=[32, 64, 128])
    
    # Create dummy input
    cover = torch.randn(1, 3, 128, 128)
    secret = torch.randn(1, 3, 128, 128)
    
    with torch.no_grad():
        stego, recovered = lightweight_model(cover, secret)
        
    print(f"Input shapes: Cover {cover.shape}, Secret {secret.shape}")
    print(f"Output shapes: Stego {stego.shape}, Recovered {recovered.shape}")
    
    # Test adaptive model
    print("\nTesting Adaptive Model:")
    adaptive_model = get_model_variant('adaptive', in_channels=3)
    
    for mode in ['lightweight', 'robust']:
        adaptive_model.set_mode(mode)
        with torch.no_grad():
            stego, recovered = adaptive_model(cover, secret)
        print(f"Mode '{mode}': Stego {stego.shape}, Recovered {recovered.shape}")
