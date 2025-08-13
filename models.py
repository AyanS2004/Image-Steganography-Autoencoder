import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    Encoder network that takes cover and secret images and produces a stego image.
    Uses convolutional layers to embed the secret image into the cover image.
    """
    
    def __init__(self, in_channels=6, out_channels=3, hidden_dims=[64, 128, 256, 512]):
        super(Encoder, self).__init__()
        
        self.hidden_dims = hidden_dims
        
        # Initial convolution to process concatenated cover + secret images
        self.initial_conv = nn.Conv2d(in_channels, hidden_dims[0], kernel_size=3, padding=1)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.encoder_layers.append(
                nn.Sequential(
                    nn.Conv2d(hidden_dims[i], hidden_dims[i+1], kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(hidden_dims[i+1]),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Decoder layers (upsampling to original size)
        self.decoder_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1, 0, -1):
            self.decoder_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i-1], kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(hidden_dims[i-1]),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Final convolution to output stego image
        self.final_conv = nn.Conv2d(hidden_dims[0], out_channels, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()
        
    def forward(self, cover, secret):
        # Concatenate cover and secret images along channel dimension
        x = torch.cat([cover, secret], dim=1)
        
        # Initial convolution
        x = F.relu(self.initial_conv(x))
        
        # Encoder path
        encoder_features = [x]
        for layer in self.encoder_layers:
            x = layer(x)
            encoder_features.append(x)
        
        # Decoder path with skip connections
        for i, layer in enumerate(self.decoder_layers):
            x = layer(x)
            # Add skip connection
            if i < len(encoder_features) - 1:
                x = x + encoder_features[-(i+2)]
        
        # Final convolution and tanh activation
        stego = self.tanh(self.final_conv(x))
        
        return stego


class Decoder(nn.Module):
    """
    Decoder network that takes a stego image and reconstructs the secret image.
    """
    
    def __init__(self, in_channels=3, out_channels=3, hidden_dims=[64, 128, 256, 512]):
        super(Decoder, self).__init__()
        
        self.hidden_dims = hidden_dims
        
        # Initial convolution
        self.initial_conv = nn.Conv2d(in_channels, hidden_dims[0], kernel_size=3, padding=1)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.encoder_layers.append(
                nn.Sequential(
                    nn.Conv2d(hidden_dims[i], hidden_dims[i+1], kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(hidden_dims[i+1]),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1, 0, -1):
            self.decoder_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i-1], kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(hidden_dims[i-1]),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Final convolution
        self.final_conv = nn.Conv2d(hidden_dims[0], out_channels, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()
        
    def forward(self, stego):
        # Initial convolution
        x = F.relu(self.initial_conv(stego))
        
        # Encoder path
        encoder_features = [x]
        for layer in self.encoder_layers:
            x = layer(x)
            encoder_features.append(x)
        
        # Decoder path with skip connections
        for i, layer in enumerate(self.decoder_layers):
            x = layer(x)
            # Add skip connection
            if i < len(encoder_features) - 1:
                x = x + encoder_features[-(i+2)]
        
        # Final convolution and tanh activation
        secret_recovered = self.tanh(self.final_conv(x))
        
        return secret_recovered


class SteganographyAutoencoder(nn.Module):
    """
    Complete autoencoder for image-in-image steganography.
    Combines encoder and decoder into a single model.
    """
    
    def __init__(self, in_channels=3, hidden_dims=[64, 128, 256, 512]):
        super(SteganographyAutoencoder, self).__init__()
        
        self.encoder = Encoder(in_channels=in_channels*2, out_channels=in_channels, hidden_dims=hidden_dims)
        self.decoder = Decoder(in_channels=in_channels, out_channels=in_channels, hidden_dims=hidden_dims)
        
    def forward(self, cover, secret):
        # Encode: cover + secret -> stego
        stego = self.encoder(cover, secret)
        
        # Decode: stego -> recovered secret
        secret_recovered = self.decoder(stego)
        
        return stego, secret_recovered
    
    def encode(self, cover, secret):
        """Only perform encoding"""
        return self.encoder(cover, secret)
    
    def decode(self, stego):
        """Only perform decoding"""
        return self.decoder(stego)


class SteganographyLoss(nn.Module):
    """
    Custom loss function for steganography training.
    Combines cover loss and secret loss with optional SSIM loss.
    """
    
    def __init__(self, alpha=1.0, beta=1.0, gamma=0.1, use_ssim=True):
        super(SteganographyLoss, self).__init__()
        self.alpha = alpha  # Weight for cover loss
        self.beta = beta    # Weight for secret loss
        self.gamma = gamma  # Weight for SSIM loss
        self.use_ssim = use_ssim
        
        # MSE loss
        self.mse_loss = nn.MSELoss()
        
        # SSIM loss (if enabled)
        if self.use_ssim:
            self.ssim_loss = self._ssim_loss
        
    def _ssim_loss(self, x, y, window_size=11):
        """Calculate SSIM loss between two images"""
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
    
    def forward(self, cover, stego, secret, secret_recovered):
        # Cover loss: MSE between cover and stego
        cover_loss = self.mse_loss(cover, stego)
        
        # Secret loss: MSE between original and recovered secret
        secret_loss = self.mse_loss(secret, secret_recovered)
        
        # SSIM loss (optional)
        ssim_loss = 0
        if self.use_ssim:
            ssim_loss = self.ssim_loss(cover, stego) + self.ssim_loss(secret, secret_recovered)
        
        # Total loss
        total_loss = self.alpha * cover_loss + self.beta * secret_loss + self.gamma * ssim_loss
        
        return {
            'total_loss': total_loss,
            'cover_loss': cover_loss,
            'secret_loss': secret_loss,
            'ssim_loss': ssim_loss
        }

