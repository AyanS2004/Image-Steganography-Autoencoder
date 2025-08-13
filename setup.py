#!/usr/bin/env python3
"""
Setup script for Digital Watermarking project.
Automates installation and initial setup.
"""

import os
import subprocess
import sys
import json
from pathlib import Path


def run_command(command, description=""):
    """Run a command and handle errors"""
    print(f"{'🔄 ' + description if description else '🔄 Running command'}: {command}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(f"✅ {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(f"   {e.stderr.strip()}")
        return False


def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def create_directories():
    """Create necessary project directories"""
    directories = [
        'checkpoints',
        'visualizations', 
        'results',
        'data',
        'web_uploads',
        'web_results',
        'optimized_results',
        'evaluation_results',
        'templates'
    ]
    
    print("📁 Creating project directories...")
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"   ✅ {directory}/")
    
    return True


def install_dependencies():
    """Install required Python packages"""
    print("📦 Installing dependencies...")
    
    # Check if requirements.txt exists
    if not os.path.exists('requirements.txt'):
        print("❌ requirements.txt not found!")
        return False
    
    # Install packages
    success = run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing Python packages"
    )
    
    if success:
        print("✅ All dependencies installed successfully!")
    
    return success


def check_gpu_availability():
    """Check for GPU availability"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"🎮 GPU Available: {gpu_name} ({gpu_count} device(s))")
            return True
        else:
            print("💻 No GPU detected. Will use CPU for training.")
            return False
    except ImportError:
        print("⚠️  Could not check GPU availability (PyTorch not installed yet)")
        return False


def create_config_files():
    """Create default configuration files"""
    print("⚙️  Creating configuration files...")
    
    # Training configuration
    train_config = {
        "model": {
            "variant": "standard",
            "hidden_dims": [64, 128, 256, 512],
            "image_size": 128
        },
        "training": {
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.001,
            "use_cifar": True,
            "mixed_precision": True,
            "early_stopping_patience": 15
        },
        "optimization": {
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "weight_decay": 0.01,
            "gradient_clipping": 1.0
        },
        "loss_weights": {
            "alpha": 1.0,
            "beta": 1.0, 
            "gamma": 0.1,
            "perceptual": 0.1
        }
    }
    
    with open('config.json', 'w', encoding='utf-8') as f:
        json.dump(train_config, f, indent=2)
    
    print("   ✅ config.json")
    
    # Web app configuration
    web_config = {
        "server": {
            "host": "127.0.0.1",
            "port": 5000,
            "debug": False
        },
        "model": {
            "checkpoint_path": "checkpoints/best_model.pth",
            "device": "cuda",
            "image_size": 128
        },
        "upload": {
            "max_file_size": "16MB",
            "allowed_extensions": ["png", "jpg", "jpeg", "bmp", "tiff"]
        }
    }
    
    with open('web_config.json', 'w', encoding='utf-8') as f:
        json.dump(web_config, f, indent=2)
    
    print("   ✅ web_config.json")
    
    return True


def create_launch_scripts():
    """Create convenient launch scripts"""
    print("🚀 Creating launch scripts...")
    
    # Training script
    train_script = """#!/bin/bash
# Quick training script
echo "Starting Digital Watermarking Training"
python train.py --epochs 50 --batch_size 32 --use_cifar --device cuda
"""
    
    with open('train_quick.sh', 'w', encoding='utf-8') as f:
        f.write(train_script)
    
    # Demo script
    demo_script = """#!/bin/bash
# Demo script
echo "Running Digital Watermarking Demo"
python demo.py --mode both --epochs 10 --image_size 64
"""
    
    with open('run_demo.sh', 'w', encoding='utf-8') as f:
        f.write(demo_script)
    
    # Web app script
    web_script = """#!/bin/bash
# Web application launcher
echo "Starting Digital Watermarking Web App"
python web_app.py --host 127.0.0.1 --port 5000
"""
    
    with open('start_web.sh', 'w', encoding='utf-8') as f:
        f.write(web_script)
    
    # Create Windows batch files
    if os.name == 'nt':  # Windows
        train_bat = """@echo off
REM Quick training script
echo Starting Digital Watermarking Training
python train.py --epochs 50 --batch_size 32 --use_cifar --device cuda
pause
"""
        
        demo_bat = """@echo off
REM Demo script
echo Running Digital Watermarking Demo
python demo.py --mode both --epochs 10 --image_size 64
pause
"""
        
        web_bat = """@echo off
REM Web application launcher
echo Starting Digital Watermarking Web App
python web_app.py --host 127.0.0.1 --port 5000
pause
"""
        
        with open('train_quick.bat', 'w', encoding='utf-8') as f:
            f.write(train_bat)
        with open('run_demo.bat', 'w', encoding='utf-8') as f:
            f.write(demo_bat)
        with open('start_web.bat', 'w', encoding='utf-8') as f:
            f.write(web_bat)
        
        print("   ✅ Launch scripts created (.bat for Windows)")
    else:
        # Make scripts executable on Unix systems
        for script in ['train_quick.sh', 'run_demo.sh', 'start_web.sh']:
            os.chmod(script, 0o755)
        print("   ✅ Launch scripts created (.sh for Unix)")
    
    return True


def download_sample_data():
    """Download sample images for testing"""
    print("🖼️  Setting up sample data...")
    
    sample_dir = Path('sample_images')
    sample_dir.mkdir(exist_ok=True)
    
    # Create synthetic sample images using the demo script
    try:
        success = run_command(
            f"{sys.executable} -c \"from demo import create_synthetic_images; create_synthetic_images()\"",
            "Creating sample images"
        )
        if success:
            print("   ✅ Sample images created in sample_images/")
        return success
    except Exception as e:
        print(f"   ⚠️  Could not create sample images: {e}")
        return False


def create_quick_start_guide():
    """Create a quick start guide"""
    guide = """# Digital Watermarking - Quick Start Guide

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
python batch_process.py --mode embed --checkpoint checkpoints/best_model.pth \\
  --cover_dir /path/to/covers --secret_dir /path/to/secrets --output_dir results

# Extract secrets from stego images  
python batch_process.py --mode extract --checkpoint checkpoints/best_model.pth \\
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
"""
    
    with open('QUICKSTART.md', 'w', encoding='utf-8') as f:
        f.write(guide)
    
    print("📖 Quick start guide created: QUICKSTART.md")
    return True


def main():
    """Main setup function"""
    print("🎭 Digital Watermarking Project Setup")
    print("=" * 50)
    
    steps = [
        ("Checking Python version", check_python_version),
        ("Creating directories", create_directories),
        ("Installing dependencies", install_dependencies),
        ("Checking GPU availability", check_gpu_availability),
        ("Creating configuration files", create_config_files),
        ("Creating launch scripts", create_launch_scripts),
        ("Setting up sample data", download_sample_data),
        ("Creating quick start guide", create_quick_start_guide)
    ]
    
    success_count = 0
    
    for description, step_function in steps:
        print(f"\n{description}...")
        if step_function():
            success_count += 1
        else:
            print(f"⚠️  {description} failed but continuing...")
    
    print("\n" + "=" * 50)
    print(f"Setup completed! {success_count}/{len(steps)} steps successful.")
    
    if success_count == len(steps):
        print("🎉 Everything is ready!")
        print("\n🚀 Quick start:")
        print("   python demo.py --mode both")
        print("   python web_app.py")
        print("\n📖 See QUICKSTART.md for detailed instructions.")
    else:
        print("⚠️  Some steps failed. Check the output above.")
        print("   You may need to install dependencies manually.")
    
    print("\nHappy experimenting with Digital Watermarking! 🎭")


if __name__ == '__main__':
    main()
