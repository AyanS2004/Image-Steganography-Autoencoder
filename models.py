# models.py - Steganography models with updated torchvision API
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import VGG19_Weights
import math


class SteganographyAutoencoder(nn.Module):
    """
    Steganography Autoencoder that embeds a secret image into a cover image
    and can recover both the steganographic image and the secret image.
    """
    
    def __init__(self, in_channels=3, hidden_dims=[64, 128, 256, 512]):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        
        # Encoder for cover image
        self.cover_encoder = self._build_encoder(in_channels, hidden_dims)
        
        # Encoder for secret image  
        self.secret_encoder = self._build_encoder(in_channels, hidden_dims)
        
        # Fusion layer to combine cover and secret features
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_dims[-1] * 2, hidden_dims[-1], 3, padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dims[-1], hidden_dims[-1], 3, padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.ReLU(inplace=True)
        )
        
        # Decoder to generate steganographic image
        self.stego_decoder = self._build_decoder(hidden_dims, in_channels)
        
        # Decoder to recover secret image from steganographic image
        self.secret_decoder = self._build_decoder(hidden_dims, in_channels)
        
        self._initialize_weights()
    
    def _build_encoder(self, in_channels, hidden_dims):
        """Build encoder layers"""
        layers = []
        
        # Initial layer
        layers.extend([
            nn.Conv2d(in_channels, hidden_dims[0], 7, padding=3),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.ReLU(inplace=True)
        ])
        
        # Downsampling layers
        for i in range(len(hidden_dims) - 1):
            layers.extend([
                nn.Conv2d(hidden_dims[i], hidden_dims[i], 3, padding=1),
                nn.BatchNorm2d(hidden_dims[i]),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dims[i], hidden_dims[i+1], 3, stride=2, padding=1),
                nn.BatchNorm2d(hidden_dims[i+1]),
                nn.ReLU(inplace=True)
            ])
        
        # Final encoder layer
        layers.extend([
            nn.Conv2d(hidden_dims[-1], hidden_dims[-1], 3, padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.ReLU(inplace=True)
        ])
        
        return nn.Sequential(*layers)
    
    def _build_decoder(self, hidden_dims, out_channels):
        """Build decoder layers"""
        layers = []
        reversed_dims = hidden_dims[::-1]
        
        # Upsampling layers
        for i in range(len(reversed_dims) - 1):
            layers.extend([
                nn.ConvTranspose2d(reversed_dims[i], reversed_dims[i], 3, padding=1),
                nn.BatchNorm2d(reversed_dims[i]),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(reversed_dims[i], reversed_dims[i+1], 3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(reversed_dims[i+1]),
                nn.ReLU(inplace=True)
            ])
        
        # Final output layer
        layers.extend([
            nn.Conv2d(reversed_dims[-1], reversed_dims[-1], 3, padding=1),
            nn.BatchNorm2d(reversed_dims[-1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(reversed_dims[-1], out_channels, 7, padding=3),
            nn.Tanh()  # Output in [-1, 1] range
        ])
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, cover, secret):
        """
        Forward pass
        Args:
            cover: Cover image tensor [B, C, H, W]
            secret: Secret image tensor [B, C, H, W]
        Returns:
            stego: Steganographic image [B, C, H, W]
            secret_recovered: Recovered secret image [B, C, H, W]
        """
        # Encode both images
        cover_features = self.cover_encoder(cover)
        secret_features = self.secret_encoder(secret)
        
        # Fuse features
        fused_features = torch.cat([cover_features, secret_features], dim=1)
        fused_features = self.fusion(fused_features)
        
        # Generate steganographic image
        stego = self.stego_decoder(fused_features)
        
        # Recover secret from steganographic image
        stego_features = self.cover_encoder(stego)
        secret_recovered = self.secret_decoder(stego_features)
        
        return stego, secret_recovered


class PerceptualLoss(nn.Module):
    """VGG-based perceptual loss for better visual quality"""
    
    def __init__(self, feature_layers=None, use_input_norm=True):
        super().__init__()
        
        if feature_layers is None:
            feature_layers = [3, 8, 15, 22]  # relu1_2, relu2_2, relu3_3, relu4_3
        
        self.feature_layers = set(feature_layers)
        
        # Load VGG19 with modern weights API
        vgg = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        self.features = vgg.features
        
        # Freeze VGG parameters
        for param in self.features.parameters():
            param.requires_grad = False
        
        self.use_input_norm = use_input_norm
        if use_input_norm:
            # ImageNet normalization
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def _normalize_inputs(self, x):
        """Normalize input from [-1,1] to ImageNet stats"""
        if self.use_input_norm:
            # Convert from [-1,1] to [0,1]
            x = (x + 1) / 2
            # Apply ImageNet normalization
            x = (x - self.mean) / self.std
        return x
    
    def _extract_features(self, x):
        """Extract features from specified layers"""
        features = []
        x = self._normalize_inputs(x)
        
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.feature_layers:
                features.append(x)
        
        return features
    
    def forward(self, pred, target):
        """Compute perceptual loss"""
        pred_features = self._extract_features(pred)
        target_features = self._extract_features(target)
        
        loss = 0
        for pred_feat, target_feat in zip(pred_features, target_features):
            loss += F.mse_loss(pred_feat, target_feat)
        
        return loss / len(pred_features)


class SSIMLoss(nn.Module):
    """Structural Similarity Index Loss"""
    
    def __init__(self, window_size=11, sigma=1.5, channel=3):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.channel = channel
        
        # Create Gaussian window
        gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) 
                             for x in range(window_size)])
        gauss = gauss / gauss.sum()
        
        # Create 2D window
        window = gauss.unsqueeze(1).mm(gauss.unsqueeze(0))
        window = window.unsqueeze(0).unsqueeze(0)
        window = window.expand(channel, 1, window_size, window_size)
        
        self.register_buffer('window', window)
        self.padding = window_size // 2
    
    def _ssim(self, img1, img2):
        """Compute SSIM between two images"""
        mu1 = F.conv2d(img1, self.window, padding=self.padding, groups=self.channel)
        mu2 = F.conv2d(img2, self.window, padding=self.padding, groups=self.channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1*img1, self.window, padding=self.padding, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, self.window, padding=self.padding, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, self.window, padding=self.padding, groups=self.channel) - mu1_mu2
        
        C1 = 0.01**2
        C2 = 0.03**2
        
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        
        return ssim_map.mean()
    
    def forward(self, pred, target):
        """Compute SSIM loss (1 - SSIM)"""
        return 1 - self._ssim(pred, target)


class SteganographyLoss(nn.Module):
    """Combined loss function for steganography training"""
    
    def __init__(self, alpha=1.0, beta=1.0, gamma=0.1, use_ssim=True, use_perceptual=True):
        super().__init__()
        
        self.alpha = alpha  # Weight for cover similarity
        self.beta = beta    # Weight for secret recovery
        self.gamma = gamma  # Weight for perceptual/SSIM loss
        
        self.mse_loss = nn.MSELoss()
        
        if use_perceptual:
            self.perceptual_loss = PerceptualLoss()
        else:
            self.perceptual_loss = None
            
        if use_ssim:
            self.ssim_loss = SSIMLoss()
        else:
            self.ssim_loss = None
    
    def forward(self, cover, stego, secret, secret_recovered):
        """
        Compute combined steganography loss
        
        Args:
            cover: Original cover image
            stego: Generated steganographic image  
            secret: Original secret image
            secret_recovered: Recovered secret image
            
        Returns:
            dict: Dictionary containing individual loss components and total loss
        """
        losses = {}
        
        # Cover similarity loss (stego should look like cover)
        cover_loss = self.mse_loss(stego, cover)
        losses['cover_loss'] = cover_loss
        
        # Secret recovery loss
        secret_loss = self.mse_loss(secret_recovered, secret)
        losses['secret_loss'] = secret_loss
        
        # Perceptual loss for better visual quality
        perceptual_loss = 0
        if self.perceptual_loss is not None:
            perceptual_loss += self.perceptual_loss(stego, cover)
            perceptual_loss += self.perceptual_loss(secret_recovered, secret)
            perceptual_loss /= 2
        
        # SSIM loss for structural similarity
        if self.ssim_loss is not None:
            ssim_loss = self.ssim_loss(stego, cover) + self.ssim_loss(secret_recovered, secret)
            ssim_loss /= 2
            perceptual_loss += ssim_loss
        
        losses['perceptual_loss'] = perceptual_loss
        
        # Total loss
        total_loss = (self.alpha * cover_loss + 
                     self.beta * secret_loss + 
                     self.gamma * perceptual_loss)
        
        losses['total_loss'] = total_loss
        
        return losses


def test_model():
    """Test the steganography model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = SteganographyAutoencoder().to(device)
    criterion = SteganographyLoss()
    
    # Test with random data
    batch_size = 2
    cover = torch.randn(batch_size, 3, 128, 128).to(device)
    secret = torch.randn(batch_size, 3, 128, 128).to(device)
    
    # Forward pass
    with torch.no_grad():
        stego, secret_recovered = model(cover, secret)
        losses = criterion(cover, stego, secret, secret_recovered)
    
    print(f"Model output shapes:")
    print(f"  Stego: {stego.shape}")
    print(f"  Secret recovered: {secret_recovered.shape}")
    print(f"Loss components:")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.6f}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")


if __name__ == "__main__":
    test_model()