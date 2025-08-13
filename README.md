# Deep Learning-Based Digital Watermarking System

A comprehensive, production-ready implementation of image-in-image steganography using advanced Convolutional Autoencoders. This project provides multiple model variants, optimization techniques, and deployment options for digital watermarking applications.

## 🌟 Project Overview

This advanced implementation uses state-of-the-art deep learning techniques to perform robust image-in-image steganography:

- **🔧 Encoder Network**: Embeds secret images into cover images with minimal visual distortion
- **🔍 Decoder Network**: Extracts hidden secret images from stego images with high fidelity
- **🎯 Advanced Training**: Multi-objective optimization with perceptual loss functions
- **🚀 Multiple Deployment Options**: From research prototypes to production web applications

## ✨ Key Features

### Core Capabilities
- **🏗️ Multiple Model Architectures**: Lightweight, standard, high-capacity, and robust variants
- **🔬 Advanced Loss Functions**: MSE, SSIM, perceptual, and adversarial losses
- **📊 Comprehensive Evaluation**: Robustness testing against common attacks (JPEG, noise, blur, etc.)
- **⚡ Performance Optimization**: Mixed precision training, gradient clipping, advanced schedulers
- **🎭 Interactive Demo**: Real-time demonstration with synthetic and custom images

### User Interfaces
- **🌐 Web Application**: Beautiful, responsive web interface for easy usage
- **📱 RESTful API**: Programmatic access for integration with other systems
- **🖥️ Command Line Tools**: Comprehensive CLI for batch processing and automation
- **📊 Visualization Suite**: Advanced plotting and analysis tools

### Deployment & Production
- **🔄 Batch Processing**: Handle multiple images with parallel processing
- **📈 Model Monitoring**: Training metrics, validation curves, and performance tracking
- **🎨 Custom Datasets**: Support for any image dataset with flexible preprocessing
- **🔒 Robustness Testing**: Evaluation against compression, noise, geometric attacks

## Architecture

### Model Components

1. **Encoder Network**:
   - Input: Cover image (3 channels) + Secret image (3 channels) = 6 channels
   - Output: Stego image (3 channels)
   - Architecture: Convolutional layers with skip connections

2. **Decoder Network**:
   - Input: Stego image (3 channels)
   - Output: Recovered secret image (3 channels)
   - Architecture: Convolutional layers with skip connections

3. **Loss Functions**:
   - Cover Loss: MSE between cover and stego images
   - Secret Loss: MSE between original and recovered secret images
   - SSIM Loss: Structural similarity for better visual quality

## 🚀 Quick Start

### Automated Setup
```bash
# Run the automated setup script
python setup.py
```

### Manual Installation

#### Prerequisites
- Python 3.8+ (3.9+ recommended)
- PyTorch 2.0+
- CUDA 11.0+ (optional, for GPU acceleration)
- 8GB+ RAM (16GB+ recommended for training)

#### Step-by-Step Setup

1. **Clone and Navigate**:
   ```bash
   git clone <repository-url>
   cd DigitalWatermarking
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Setup**:
   ```bash
   python setup.py  # Creates directories and configuration files
   ```

4. **Verify Installation**:
   ```bash
   python demo.py --mode inference --epochs 5
   ```

### 🎭 Instant Demo
```bash
# Quick demonstration with synthetic images
python demo.py --mode both --epochs 10 --image_size 64

# Web interface (open http://127.0.0.1:5000)
python web_app.py
```

## 📚 Usage Guide

### 🎯 1. Training Models

#### Basic Training (CIFAR-10)
```bash
# Standard training
python train.py --epochs 50 --batch_size 32 --use_cifar --device cuda

# Advanced optimization
python optimization.py --epochs 100 --mixed_precision --config config.json
```

#### Custom Dataset Training
```bash
python train.py --epochs 50 --batch_size 16 \
  --custom_image_dir /path/to/images --device cuda
```

#### Model Variants
```bash
# Lightweight model for mobile deployment
python -c "
from model_variants import get_model_variant
model = get_model_variant('lightweight')
print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')
"

# High-capacity model for complex images
python -c "
from model_variants import get_model_variant  
model = get_model_variant('high_capacity')
print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')
"
```

### 🔍 2. Inference & Testing

#### Single Image Processing
```bash
python inference.py \
    --checkpoint checkpoints/best_model.pth \
    --cover_image cover.jpg \
    --secret_image secret.jpg \
    --output_dir results \
    --visualize
```

#### Batch Processing
```bash
# Embed multiple secrets
python batch_process.py --mode embed \
  --checkpoint checkpoints/best_model.pth \
  --cover_dir covers/ --secret_dir secrets/ \
  --output_dir batch_results/ --max_workers 4

# Extract from stego images
python batch_process.py --mode extract \
  --checkpoint checkpoints/best_model.pth \
  --stego_dir stegos/ --output_dir extracted/
```

### 🌐 3. Web Interface

#### Launch Web App
```bash
# Basic launch
python web_app.py

# Custom configuration
python web_app.py --host 0.0.0.0 --port 8080 \
  --checkpoint checkpoints/best_model.pth --device cuda
```

#### Web Features
- **📤 Upload Interface**: Drag-and-drop image upload
- **⚙️ Real-time Processing**: Instant steganography results
- **📊 Quality Metrics**: PSNR and SSIM calculations
- **💾 Download Results**: High-quality output images
- **📱 Responsive Design**: Works on desktop and mobile

### 📊 4. Evaluation & Analysis

#### Comprehensive Evaluation
```bash
python evaluate.py --checkpoint checkpoints/best_model.pth \
  --test_data cifar --num_samples 1000 --output_dir evaluation/
```

#### Robustness Testing
```bash
# Test against various attacks
python evaluate.py --checkpoint checkpoints/best_model.pth \
  --test_data cifar --num_samples 500 \
  --output_dir robustness_test/
```

#### Model Comparison
```bash
# Compare different model variants
python -c "
from model_variants import compare_model_sizes
compare_model_sizes()
"
```

## 📁 Project Structure

```
DigitalWatermarking/
├── 🏗️ Core Models
│   ├── models.py              # Base autoencoder architectures
│   ├── model_variants.py      # Lightweight, robust, high-capacity variants
│   └── optimization.py        # Advanced training techniques
│
├── 📊 Data & Training  
│   ├── dataset.py             # Data loading and preprocessing
│   ├── trainer.py             # Basic training loop
│   ├── train.py              # Main training script
│   └── config.json           # Training configuration
│
├── 🔍 Inference & Testing
│   ├── inference.py          # Single image processing
│   ├── batch_process.py      # Batch operations
│   ├── evaluate.py           # Comprehensive evaluation
│   └── demo.py               # Interactive demonstration
│
├── 🌐 Web Interface
│   ├── web_app.py            # Flask web application
│   ├── web_config.json       # Web app configuration
│   └── templates/            # HTML templates
│
├── 🚀 Deployment
│   ├── setup.py              # Automated setup script
│   ├── requirements.txt      # Python dependencies
│   ├── QUICKSTART.md         # Quick start guide
│   └── README.md             # This comprehensive guide
│
└── 📂 Generated Directories
    ├── checkpoints/          # Model checkpoints
    ├── visualizations/       # Training plots
    ├── results/             # Inference outputs
    ├── evaluation_results/   # Evaluation reports
    ├── optimized_results/    # Optimization results
    ├── web_uploads/         # Web app uploads
    ├── web_results/         # Web app outputs
    └── sample_images/       # Demo images
```

## Model Architecture Details

### Encoder Network
- Input: 6-channel concatenated image (cover + secret)
- Convolutional layers with increasing channel dimensions: 64 → 128 → 256 → 512
- Skip connections for better gradient flow
- Output: 3-channel stego image

### Decoder Network
- Input: 3-channel stego image
- Transposed convolutional layers with decreasing channel dimensions: 512 → 256 → 128 → 64
- Skip connections for feature preservation
- Output: 3-channel recovered secret image

### Loss Function
```python
Total Loss = α × Cover Loss + β × Secret Loss + γ × SSIM Loss
```
Where:
- Cover Loss: MSE between cover and stego images
- Secret Loss: MSE between original and recovered secret images
- SSIM Loss: Structural similarity for visual quality

## Training Process

1. **Data Preparation**: Images are resized to 128×128 and normalized to [-1, 1]
2. **Training Loop**: 
   - Forward pass through encoder and decoder
   - Loss calculation with multiple components
   - Backward pass and optimization
   - Regular visualization and checkpointing
3. **Validation**: Regular evaluation on validation set
4. **Metrics**: PSNR and SSIM calculation for quality assessment

## Results and Metrics

The model is evaluated using:

- **PSNR (Peak Signal-to-Noise Ratio)**: Measures image quality
- **SSIM (Structural Similarity Index)**: Measures perceptual similarity
- **Cover Loss**: Ensures minimal distortion to cover image
- **Secret Loss**: Ensures accurate secret image recovery

## Example Results

After training, you should see:
- Stego images that are visually similar to cover images
- Recovered secret images with high fidelity
- PSNR values > 30 dB for good quality
- SSIM values > 0.9 for high similarity

## Customization

### Modifying Architecture

Edit `models.py` to change:
- Network depth and width
- Activation functions
- Loss function weights

### Using Different Datasets

1. **Custom Dataset**: Place images in a directory and use `--custom_image_dir`
2. **Other Datasets**: Modify `dataset.py` to support other datasets

### Hyperparameter Tuning

Key parameters to experiment with:
- Learning rate (`--lr`)
- Batch size (`--batch_size`)
- Loss weights (in `SteganographyLoss`)
- Network architecture (hidden dimensions)

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size: `--batch_size 16`
   - Use CPU: `--device cpu`

2. **Slow Training**:
   - Use GPU: `--device cuda`
   - Reduce image size: `--image_size 64`

3. **Poor Results**:
   - Increase training epochs
   - Adjust learning rate
   - Check data quality

### Performance Tips

- Use GPU for faster training
- Start with CIFAR-10 for initial testing
- Monitor loss curves during training
- Use appropriate batch size for your hardware

## 🎯 Advanced Features

### 🏗️ Model Variants
- **Lightweight Models**: Optimized for mobile/edge deployment with depthwise separable convolutions
- **High-Capacity Models**: Enhanced architecture with attention mechanisms for complex images
- **Robust Models**: Noise-resistant training with dropout and augmentation for attack resilience
- **Adaptive Models**: Dynamic switching between different operating modes

### ⚡ Optimization Techniques
- **Mixed Precision Training**: Up to 2x speedup with automatic loss scaling
- **Advanced Optimizers**: AdamW with parameter group scheduling
- **Learning Rate Scheduling**: Cosine annealing and plateau reduction
- **Gradient Clipping**: Stable training with gradient norm constraints
- **Early Stopping**: Automatic training termination with patience monitoring

### 🔒 Robustness Features
- **Attack Simulation**: JPEG compression, Gaussian noise, blur, rotation, scaling
- **Quality Metrics**: PSNR, SSIM, perceptual similarity measurements
- **Batch Evaluation**: Automated testing across multiple attack scenarios
- **Performance Benchmarking**: Comprehensive model comparison tools

### 🌐 Production Deployment
- **Web Interface**: Professional Flask application with responsive design
- **RESTful API**: Programmatic access for system integration
- **Batch Processing**: Parallel processing for high-throughput scenarios
- **Configuration Management**: JSON-based configuration for different environments

### 📊 Analysis & Monitoring
- **Training Visualization**: Real-time loss curves and metric tracking
- **Model Profiling**: Parameter counting and memory usage analysis
- **Quality Assessment**: Automated image quality evaluation
- **Performance Logging**: Comprehensive training history and checkpointing

## 🔬 Research Applications

This comprehensive implementation enables research in:

1. **Video Steganography**: Frame-by-frame processing with temporal consistency
2. **Adversarial Steganography**: Robust hiding against detection algorithms
3. **Mobile Steganography**: Lightweight models for real-time mobile applications
4. **Multi-Modal Hiding**: Embedding different types of data (text, audio, video)
5. **Adaptive Steganography**: Dynamic adjustment based on content analysis
6. **Distributed Systems**: Scalable deployment across multiple machines

## Contributing

We welcome contributions! Please see our contribution guidelines:

### Development Setup
```bash
# Clone the repository
git clone <repository-url>
cd DigitalWatermarking

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If available

# Run tests
python -m pytest tests/

# Run code quality checks
flake8 *.py
black *.py --check
```
