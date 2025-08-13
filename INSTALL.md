# Installation Guide

## 🔧 NumPy Compatibility Issue Fix

If you're encountering NumPy compatibility warnings, follow these steps:

### Option 1: Quick Fix (Recommended)
```bash
pip install "numpy<2.0.0" --force-reinstall
pip install torch torchvision --force-reinstall
```

### Option 2: Clean Environment
```bash
# Create a virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install compatible versions
pip install "numpy<2.0.0"
pip install torch torchvision
pip install -r requirements.txt
```

### Option 3: Minimal Installation
Install only the essential packages:
```bash
pip install "numpy<2.0.0"
pip install torch torchvision
pip install Pillow matplotlib tqdm
pip install flask requests
```

## 🚀 Verify Installation

After fixing the NumPy issue, test the installation:

```bash
# Test basic functionality
python -c "import torch; import numpy; print('✅ Core libraries working')"

# Run a quick demo
python demo.py --mode inference --epochs 2 --image_size 64
```

## 📱 Simple Demo (No Dependencies)

If you're still having issues, here's a simple demo script that works with minimal dependencies:

```python
# simple_demo.py
import os
print("🎭 Digital Watermarking Project")
print("📁 Project structure created!")

# List all Python files
python_files = [f for f in os.listdir('.') if f.endswith('.py')]
print(f"📄 Python modules: {len(python_files)}")
for file in python_files:
    print(f"   - {file}")

print("\n🚀 Ready to use!")
print("💡 Fix NumPy compatibility and run: python demo.py")
```

## 🎯 Next Steps

1. **Fix NumPy compatibility** using one of the options above
2. **Run the demo**: `python demo.py --mode inference`
3. **Start web interface**: `python web_app.py`
4. **Train a model**: `python train.py --use_cifar --epochs 10`

## 🆘 Troubleshooting

### Common Issues:

1. **Permission Denied**: Run command prompt as administrator (Windows)
2. **Module Not Found**: Ensure you're in the correct directory
3. **CUDA Issues**: Add `--device cpu` to any command
4. **Memory Errors**: Reduce batch size with `--batch_size 8`

### Get Help:
- Check the error message carefully
- Try running in a fresh virtual environment
- Use CPU instead of GPU for testing
- Start with the minimal installation option

Happy experimenting! 🎭
