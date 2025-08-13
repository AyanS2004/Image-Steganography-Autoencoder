import os, random, shutil

src_dir = r"D:\train2017"
train_dir = r"D:\COCO\train"
val_dir = r"D:\COCO\val"

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

files = [f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.png'))]
random.shuffle(files)

split_ratio = 0.9  # 90% train, 10% val
split_index = int(len(files) * split_ratio)

train_files = files[:split_index]
val_files = files[split_index:]

for f in train_files:
    shutil.copy(os.path.join(src_dir, f), os.path.join(train_dir, f))

for f in val_files:
    shutil.copy(os.path.join(src_dir, f), os.path.join(val_dir, f))

print(f"Train: {len(train_files)}, Val: {len(val_files)}")
