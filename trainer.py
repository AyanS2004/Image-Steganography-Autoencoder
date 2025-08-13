import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
from models import SteganographyLoss
from dataset import denormalize_image


class SteganographyTrainer:
    def __init__(self, model, train_loader, val_loader, device='cuda', lr=0.001):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.criterion = SteganographyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.train_losses = []
        self.val_losses = []
        
        os.makedirs('checkpoints', exist_ok=True)
        os.makedirs('visualizations', exist_ok=True)
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        
        for cover, secret in tqdm(self.train_loader, desc=f'Epoch {epoch+1}'):
            cover, secret = cover.to(self.device), secret.to(self.device)
            
            self.optimizer.zero_grad()
            stego, secret_recovered = self.model(cover, secret)
            
            loss_dict = self.criterion(cover, stego, secret, secret_recovered)
            loss = loss_dict['total_loss']
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def validate_epoch(self, epoch):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for cover, secret in self.val_loader:
                cover, secret = cover.to(self.device), secret.to(self.device)
                stego, secret_recovered = self.model(cover, secret)
                
                loss_dict = self.criterion(cover, stego, secret, secret_recovered)
                total_loss += loss_dict['total_loss'].item()
        
        return total_loss / len(self.val_loader)
    
    def visualize_results(self, epoch, num_samples=4):
        self.model.eval()
        
        cover, secret = next(iter(self.val_loader))
        cover, secret = cover[:num_samples].to(self.device), secret[:num_samples].to(self.device)
        
        with torch.no_grad():
            stego, secret_recovered = self.model(cover, secret)
        
        cover = denormalize_image(cover.cpu())
        secret = denormalize_image(secret.cpu())
        stego = denormalize_image(stego.cpu())
        secret_recovered = denormalize_image(secret_recovered.cpu())
        
        fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
        
        for i in range(num_samples):
            axes[i, 0].imshow(cover[i].permute(1, 2, 0))
            axes[i, 0].set_title('Cover Image')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(secret[i].permute(1, 2, 0))
            axes[i, 1].set_title('Secret Image')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(stego[i].permute(1, 2, 0))
            axes[i, 2].set_title('Stego Image')
            axes[i, 2].axis('off')
            
            axes[i, 3].imshow(secret_recovered[i].permute(1, 2, 0))
            axes[i, 3].set_title('Recovered Secret')
            axes[i, 3].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'visualizations/epoch_{epoch+1}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_checkpoint(self, epoch):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        torch.save(checkpoint, f'checkpoints/checkpoint_epoch_{epoch+1}.pth')
    
    def train(self, num_epochs, save_freq=5, visualize_freq=5):
        print(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate_epoch(epoch)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'  Train Loss: {train_loss:.4f}')
            print(f'  Val Loss: {val_loss:.4f}')
            
            if (epoch + 1) % save_freq == 0:
                self.save_checkpoint(epoch)
            
            if (epoch + 1) % visualize_freq == 0:
                self.visualize_results(epoch)
        
        print("Training completed!")
