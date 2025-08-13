import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import random


class CustomImageDataset(Dataset):
    """
    Dataset for image-in-image steganography using custom images (e.g., COCO train2017).
    Loads images from a directory and returns random cover-secret pairs.
    """
    def __init__(self, image_dir, image_size=128):
        self.image_dir = image_dir
        self.image_files = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        if not self.image_files:
            raise ValueError(f"No images found in {image_dir}")

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Cover image
        cover_path = os.path.join(self.image_dir, self.image_files[idx])
        cover_img = Image.open(cover_path).convert('RGB')

        # Secret image (different from cover)
        secret_idx = idx
        while secret_idx == idx:
            secret_idx = random.randint(0, len(self.image_files) - 1)
        secret_path = os.path.join(self.image_dir, self.image_files[secret_idx])
        secret_img = Image.open(secret_path).convert('RGB')

        cover_img = self.transform(cover_img)
        secret_img = self.transform(secret_img)
        return cover_img, secret_img


def create_dataloader(batch_size=32, image_size=128, num_workers=4,
                      custom_image_dir=None):
    """
    Creates train and test data loaders using only custom images.
    """
    if custom_image_dir is None:
        raise ValueError("custom_image_dir must be provided")

    dataset = CustomImageDataset(image_dir=custom_image_dir, image_size=image_size)

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def denormalize_image(tensor):
    """Denormalize from [-1, 1] to [0, 1]."""
    return (tensor + 1) / 2


def normalize_image(tensor):
    """Normalize from [0, 1] to [-1, 1]."""
    return 2 * tensor - 1
