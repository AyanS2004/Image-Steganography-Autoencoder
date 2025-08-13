# Digital Watermarking - Quick Start Guide

## 🚀 Getting Started

### 1. Run the Demo
```bash
python demo.py --mode both --epochs 10
```

### 2. Train on CIFAR-10
```bash
python train.py --epochs 50 --batch_size 32 --use_cifar
```

### 3. Start Web Interface
```bash
python web_app.py
```
Then open http://127.0.0.1:5000 in your browser.

### 4. Evaluate Model
```bash
python evaluate.py --checkpoint checkpoints/best_model.pth
```

### 5. Batch Processing
```bash
# Embed secrets in multiple images
python batch_process.py --mode embed --checkpoint checkpoints/best_model.pth \
  --cover_dir /path/to/covers --secret_dir /path/to/secrets --output_dir results

# Extract secrets from stego images  
python batch_process.py --mode extract --checkpoint checkpoints/best_model.pth \
  --stego_dir /path/to/stegos --output_dir results
```

## 📊 Model Variants

- **Lightweight**: Fast inference for mobile devices
- **High Capacity**: Better quality for complex images
- **Robust**: Resistant to compression and noise
- **Adaptive**: Switch between modes dynamically

## 🔧 Advanced Features

- Mixed precision training
- Perceptual loss functions
- Robustness testing
- Comprehensive evaluation
- Web-based interface
- Batch processing

## 📁 Project Structure

```
DigitalWatermarking/
├── models.py              # Core model architectures
├── model_variants.py      # Alternative model designs
├── dataset.py             # Data loading and preprocessing
├── trainer.py             # Basic training loop
├── train.py              # Main training script
├── inference.py          # Single image inference
├── demo.py               # Interactive demonstration
├── evaluate.py           # Comprehensive evaluation
├── batch_process.py      # Batch operations
├── web_app.py            # Web interface
├── optimization.py       # Advanced training techniques
├── config.json           # Training configuration
└── web_config.json       # Web app configuration
```

## 🎯 Next Steps

1. Train your first model: `python demo.py`
2. Explore the web interface: `python web_app.py`
3. Try different model variants in `model_variants.py`
4. Optimize performance with `optimization.py`
5. Test robustness with `evaluate.py`

Happy experimenting! 🎭
