import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
import random


class SteganographyDataset(Dataset):
    """
    Dataset for image-in-image steganography.
    Creates pairs of cover and secret images from CIFAR-10 dataset.
    """
    
    def __init__(self, root='./data', train=True, image_size=128, transform=None):
        self.root = root
        self.train = train
        self.image_size = image_size
        self.transform = transform
        
        # Load CIFAR-10 dataset
        self.cifar_dataset = torchvision.datasets.CIFAR10(
            root=root, 
            train=train, 
            download=True,
            transform=transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        )
        
        # Create indices for random pairing
        self.indices = list(range(len(self.cifar_dataset)))
        
    def __len__(self):
        return len(self.cifar_dataset)
    
    def __getitem__(self, idx):
        # Get cover image
        cover_img, _ = self.cifar_dataset[idx]
        
        # Randomly select a different image as secret
        secret_idx = random.choice([i for i in self.indices if i != idx])
        secret_img, _ = self.cifar_dataset[secret_idx]
        
        return cover_img, secret_img


class CustomImageDataset(Dataset):
    """
    Custom dataset for loading images from a directory.
    Useful for testing with custom images.
    """
    
    def __init__(self, image_dir, image_size=128, transform=None):
        self.image_dir = image_dir
        self.image_size = image_size
        self.transform = transform
        
        # Get list of image files
        self.image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            self.image_files.extend(
                [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
                 if f.lower().endswith(ext.replace('*', ''))]
            )
        
        if not self.image_files:
            raise ValueError(f"No image files found in {image_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            # Default transforms
            transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            image = transform(image)
        
        return image


class SteganographyDataLoader:
    """
    Data loader wrapper for steganography training and testing.
    """
    
    def __init__(self, batch_size=32, image_size=128, num_workers=4, 
                 custom_image_dir=None, use_cifar=True):
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers
        
        if use_cifar:
            # Use CIFAR-10 dataset
            self.train_dataset = SteganographyDataset(
                root='./data', 
                train=True, 
                image_size=image_size
            )
            self.test_dataset = SteganographyDataset(
                root='./data', 
                train=False, 
                image_size=image_size
            )
        else:
            # Use custom images
            if custom_image_dir is None:
                raise ValueError("custom_image_dir must be provided when use_cifar=False")
            
            self.train_dataset = CustomImageDataset(
                image_dir=custom_image_dir,
                image_size=image_size
            )
            self.test_dataset = self.train_dataset  # Use same dataset for testing
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    def get_loaders(self):
        """Return train and test data loaders"""
        return self.train_loader, self.test_loader
    
    def get_sample_batch(self):
        """Get a sample batch for testing"""
        for batch in self.train_loader:
            return batch
        return None


def create_dataloader(batch_size=32, image_size=128, num_workers=4, 
                     custom_image_dir=None, use_cifar=True):
    """
    Convenience function to create data loaders.
    """
    loader = SteganographyDataLoader(
        batch_size=batch_size,
        image_size=image_size,
        num_workers=num_workers,
        custom_image_dir=custom_image_dir,
        use_cifar=use_cifar
    )
    return loader.get_loaders()


def denormalize_image(tensor):
    """
    Denormalize image tensor from [-1, 1] to [0, 1] range.
    """
    return (tensor + 1) / 2


def normalize_image(tensor):
    """
    Normalize image tensor from [0, 1] to [-1, 1] range.
    """
    return 2 * tensor - 1
