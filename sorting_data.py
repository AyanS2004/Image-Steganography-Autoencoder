import os
import random
import shutil
from pathlib import Path

def create_coco_subset(
    coco_train_dir,
    coco_val_dir,
    output_train_dir,
    output_val_dir,
    num_train=20000,
    num_val=2000,
    extensions=(".jpg", ".jpeg", ".png")
):
    """
    Create a random subset of COCO train/val images.

    coco_train_dir: Path to original COCO train folder
    coco_val_dir: Path to original COCO val folder
    output_train_dir: Destination folder for subset train
    output_val_dir: Destination folder for subset val
    num_train: Number of training images to copy
    num_val: Number of validation images to copy
    extensions: Allowed image extensions
    """
    def copy_random_files(src_dir, dest_dir, num_files):
        Path(dest_dir).mkdir(parents=True, exist_ok=True)
        files = [f for f in os.listdir(src_dir) if f.lower().endswith(extensions)]
        if num_files > len(files):
            raise ValueError(f"Requested {num_files} files, but only {len(files)} available in {src_dir}")

        selected_files = random.sample(files, num_files)
        for i, file in enumerate(selected_files, 1):
            shutil.copy2(os.path.join(src_dir, file), os.path.join(dest_dir, file))
            if i % 500 == 0:
                print(f"Copied {i}/{num_files} files to {dest_dir}")

    print(f"Creating training subset of {num_train} images...")
    copy_random_files(coco_train_dir, output_train_dir, num_train)

    print(f"Creating validation subset of {num_val} images...")
    copy_random_files(coco_val_dir, output_val_dir, num_val)

    print("Subset creation complete!")

if __name__ == "__main__":
    # Paths - change these to match your system
    coco_train_dir = r"D:\COCO\train"
    coco_val_dir = r"D:\COCO\val"
    output_train_dir = r"D:\COCO\train_subset"
    output_val_dir = r"D:\COCO\val_subset"

    create_coco_subset(
        coco_train_dir=coco_train_dir,
        coco_val_dir=coco_val_dir,
        output_train_dir=output_train_dir,
        output_val_dir=output_val_dir,
        num_train=20000,  # adjust if needed
        num_val=2000
    )
