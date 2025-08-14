# dataset.py - Complete dataset module with build_loaders function
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import random


def denormalize_image(tensor):
    """Convert normalized tensor back to [0,1] range for visualization"""
    # Assuming normalization was done with mean=0.5, std=0.5
    return tensor * 0.5 + 0.5


class CustomImageDataset(Dataset):
    """Dataset for loading image pairs from directories"""
    
    def __init__(self, image_dirs, image_size=128, transform=None):
        """
        Args:
            image_dirs: List of directories containing images, or single directory
            image_size: Size to resize images to
            transform: Optional transform to apply
        """
        if isinstance(image_dirs, str):
            image_dirs = [image_dirs]
        
        self.image_paths = []
        self.image_size = image_size
        
        # Collect all image paths
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        for image_dir in image_dirs:
            if os.path.exists(image_dir):
                for filename in os.listdir(image_dir):
                    if any(filename.lower().endswith(ext) for ext in valid_extensions):
                        self.image_paths.append(os.path.join(image_dir, filename))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No valid images found in directories: {image_dirs}")
        
        # Default transform if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            self.transform = transform
        
        print(f"Found {len(self.image_paths)} images in dataset")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """Return a pair of images: cover and secret"""
        try:
            # Load cover image
            cover_path = self.image_paths[idx]
            cover_image = Image.open(cover_path).convert('RGB')
            cover_tensor = self.transform(cover_image)
            
            # Load secret image (randomly select different image)
            secret_idx = random.randint(0, len(self.image_paths) - 1)
            while secret_idx == idx and len(self.image_paths) > 1:
                secret_idx = random.randint(0, len(self.image_paths) - 1)
            
            secret_path = self.image_paths[secret_idx]
            secret_image = Image.open(secret_path).convert('RGB')
            secret_tensor = self.transform(secret_image)
            
            return cover_tensor, secret_tensor
            
        except Exception as e:
            print(f"Error loading image at index {idx}: {e}")
            # Return a random valid pair as fallback
            return self.__getitem__(random.randint(0, len(self.image_paths) - 1))


class CIFARPairDataset(Dataset):
    """CIFAR-10 dataset that returns pairs of images"""
    
    def __init__(self, root='./data', train=True, image_size=128):
        self.dataset = torchvision.datasets.CIFAR10(
            root=root, 
            train=train, 
            download=True,
            transform=transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        )
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """Return a pair of images: cover and secret"""
        cover_img, _ = self.dataset[idx]
        
        # Select different image for secret
        secret_idx = random.randint(0, len(self.dataset) - 1)
        while secret_idx == idx and len(self.dataset) > 1:
            secret_idx = random.randint(0, len(self.dataset) - 1)
        
        secret_img, _ = self.dataset[secret_idx]
        
        return cover_img, secret_img


def build_loaders(use_cifar=False, train_dir=None, val_dir=None, image_size=128, 
                  batch_size=32, num_workers=4, val_split=0.1, pin_memory=True):
    """
    Build training and validation data loaders
    
    Args:
        use_cifar: If True, use CIFAR-10 dataset
        train_dir: Directory containing training images
        val_dir: Directory containing validation images  
        image_size: Size to resize images to
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        val_split: Fraction of data to use for validation (if val_dir not provided)
        pin_memory: Whether to pin memory for faster GPU transfer
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    
    if use_cifar:
        print("Using CIFAR-10 dataset")
        train_dataset = CIFARPairDataset(train=True, image_size=image_size)
        val_dataset = CIFARPairDataset(train=False, image_size=image_size)
        
    else:
        # Use custom image directories
        if train_dir is None:
            raise ValueError("train_dir must be provided when not using CIFAR")
        
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),  # Data augmentation for training
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        if val_dir and os.path.exists(val_dir):
            # Use separate validation directory
            print(f"Using separate train/val directories: {train_dir} / {val_dir}")
            train_dataset = CustomImageDataset([train_dir], image_size, transform=transform)
            val_dataset = CustomImageDataset([val_dir], image_size, transform=val_transform)
            
        else:
            # Split training directory
            print(f"Splitting training directory with {val_split:.1%} for validation")
            full_dataset = CustomImageDataset([train_dir], image_size, transform=val_transform)
            
            # Calculate split sizes
            total_size = len(full_dataset)
            val_size = int(total_size * val_split)
            train_size = total_size - val_size
            
            # Split dataset
            train_dataset, val_dataset = random_split(
                full_dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(42)  # For reproducible splits
            )
            
            # Apply different transforms to train set
            train_dataset.dataset.transform = transform
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # Drop last incomplete batch
        persistent_workers=num_workers > 0  # Keep workers alive between epochs
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0
    )
    
    print(f"Created data loaders:")
    print(f"  Training: {len(train_loader)} batches ({len(train_dataset)} samples)")
    print(f"  Validation: {len(val_loader)} batches ({len(val_dataset)} samples)")
    
    return train_loader, val_loader


# Test function
def test_loaders():
    """Test the data loaders with sample data"""
    print("Testing CIFAR-10 loaders...")
    train_loader, val_loader = build_loaders(
        use_cifar=True,
        image_size=64,
        batch_size=8,
        num_workers=0
    )
    
    # Test loading a batch
    cover_batch, secret_batch = next(iter(train_loader))
    print(f"Cover batch shape: {cover_batch.shape}")
    print(f"Secret batch shape: {secret_batch.shape}")
    print("CIFAR-10 loaders working correctly!")


if __name__ == "__main__":
    test_loaders()