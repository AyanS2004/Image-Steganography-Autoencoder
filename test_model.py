# test_model_inference.py
import torch
from PIL import Image
from torchvision import transforms
import random
import os
from skimage.exposure import match_histograms
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from models import SteganographyAutoencoder  # Import from your models.py

# ===== CONFIG =====
MODEL_PATH = "cuda_safe_checkpoints/best_model.pth"
TEST_IMAGES_DIR = "test_images"
OUTPUT_DIR = "test_results"
IMAGE_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ==================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load model
model = SteganographyAutoencoder().to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Transform to match training preprocessing
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

# Denormalize for saving
def denorm(tensor):
    return torch.clamp(tensor * 0.5 + 0.5, 0, 1)

# Pick 2 random images
images = random.sample(os.listdir(TEST_IMAGES_DIR), 2)
cover_path = os.path.join(TEST_IMAGES_DIR, images[0])
secret_path = os.path.join(TEST_IMAGES_DIR, images[1])

cover_img = transform(Image.open(cover_path).convert("RGB")).unsqueeze(0).to(DEVICE)
secret_img = transform(Image.open(secret_path).convert("RGB")).unsqueeze(0).to(DEVICE)

# Forward pass
with torch.no_grad():
    stego, secret_recovered = model(cover_img, secret_img)

# Save images
to_pil = transforms.ToPILImage()
to_pil(denorm(cover_img.squeeze().cpu())).save(os.path.join(OUTPUT_DIR, "figure_5_1_cover.png"))
to_pil(denorm(secret_img.squeeze().cpu())).save(os.path.join(OUTPUT_DIR, "figure_5_2_secret.png"))
to_pil(denorm(stego.squeeze().cpu())).save(os.path.join(OUTPUT_DIR, "figure_5_3_stego.png"))
to_pil(denorm(secret_recovered.squeeze().cpu())).save(os.path.join(OUTPUT_DIR, "figure_5_5_decoded_secret.png"))

# Color-corrected secret
decoded_img = denorm(secret_recovered.squeeze().cpu()).permute(1, 2, 0).numpy()
original_img = denorm(secret_img.squeeze().cpu()).permute(1, 2, 0).numpy()
matched = match_histograms(decoded_img, original_img, channel_axis=-1)
Image.fromarray((matched * 255).astype(np.uint8)).save(
    os.path.join(OUTPUT_DIR, "figure_5_5b_decoded_secret_color_corrected.png")
)

# Heatmap (absolute difference)
cover_np = denorm(cover_img.squeeze().cpu()).permute(1, 2, 0).numpy()
stego_np = denorm(stego.squeeze().cpu()).permute(1, 2, 0).numpy()
diff = np.abs(cover_np - stego_np).mean(axis=2)  # average over RGB

plt.imshow(diff, cmap='hot')
plt.axis('off')
plt.colorbar(label="Difference Intensity")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "figure_5_4_heatmap.png"), dpi=300)
plt.close()

# PSNR & SSIM metrics
psnr_value = psnr(cover_np, stego_np, data_range=1)
ssim_value = ssim(cover_np, stego_np, channel_axis=2, data_range=1)

metrics_table = f"PSNR: {psnr_value:.2f} dB\nSSIM: {ssim_value:.4f}\n"
with open(os.path.join(OUTPUT_DIR, "figure_5_7_metrics.txt"), "w") as f:
    f.write(metrics_table)

print(f"Done! Results saved in {OUTPUT_DIR}")
print(metrics_table)
print(f"Cover: {images[0]}")
print(f"Secret: {images[1]}")
