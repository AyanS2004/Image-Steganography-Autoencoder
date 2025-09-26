# 🎭 Enhanced Digital Watermarking Suite

A state-of-the-art, production-ready web application for image-in-image steganography using advanced deep learning techniques. This enhanced version provides a modern, intuitive interface with comprehensive functionality for all your steganography needs.

## ✨ New Features

### 🚀 Modern Web Interface
- **Beautiful, Responsive Design**: Modern UI with smooth animations and professional styling
- **Dark/Light Theme Support**: Automatically adapts to user preferences
- **Mobile-First Responsive**: Works perfectly on all devices and screen sizes
- **Real-time Progress Tracking**: Visual progress indicators for all operations
- **Drag & Drop Support**: Intuitive file upload with drag and drop functionality

### 🧠 Advanced Model Support
- **Multiple Model Variants**: Standard, Lightweight, Robust, and Adaptive models
- **Dynamic Model Switching**: Switch between models based on your needs
- **Real-time Model Information**: Live model status, parameters, and performance metrics
- **GPU Acceleration**: Full CUDA support with automatic fallback to CPU

### 📊 Comprehensive Analytics
- **Advanced Metrics Dashboard**: PSNR, SSIM, and custom quality metrics
- **Visual Quality Assessment**: Side-by-side comparison of results
- **Performance Monitoring**: Real-time processing statistics
- **Export Capabilities**: Download results in multiple formats

### 🔄 Batch Processing
- **Multi-file Operations**: Process hundreds of images simultaneously
- **Queue Management**: Visual queue with progress tracking
- **Flexible Pairing Strategies**: Sequential, random, or all-combinations pairing
- **Background Processing**: Non-blocking batch operations

### 🎯 Model Training & Evaluation
- **Interactive Training**: Start, monitor, and control training sessions
- **Real-time Training Metrics**: Live loss curves and validation metrics
- **Model Evaluation Suite**: Comprehensive model testing and analysis
- **Custom Dataset Support**: Train on your own datasets

## 🚀 Quick Start

### Option 1: Enhanced Launcher (Recommended)
```bash
# Install dependencies
pip install -r requirements.txt

# Launch the enhanced application
python launch_enhanced_app.py

# Or with custom settings
python launch_enhanced_app.py --port 8080 --demo
```

### Option 2: Direct Launch
```bash
# Run enhanced web app directly
python enhanced_web_app.py

# Or run original app with enhanced template
python web_app.py
```

### Option 3: Quick Demo
```bash
# Run a quick demonstration first
python launch_enhanced_app.py --demo --setup-only
```

## 🌐 Web Interface

Once started, access the application at:
- **Main Enhanced Interface**: http://localhost:5000/
- **Classic Interface**: http://localhost:5000/classic
- **API Endpoints**: Available for programmatic access

### 📱 Interface Sections

1. **Dashboard**: System overview, quick stats, and model status
2. **Embed Secret**: Hide secret images in cover images
3. **Extract Secret**: Recover hidden images from stego images
4. **Batch Processing**: Handle multiple images simultaneously
5. **Model Training**: Train custom steganography models
6. **Evaluation**: Comprehensive model performance analysis
7. **Settings**: Configure application preferences

## 🎛️ Advanced Features

### Model Variants
- **Standard**: Balanced performance and quality
- **Lightweight**: Fast processing for real-time applications
- **Robust**: Enhanced security and resistance to attacks
- **Adaptive**: Automatically adjusts based on input characteristics

### Quality Settings
- **Processing Size**: 64x64 (Fast) to 512x512 (Ultra Quality)
- **Quality Priority**: Balanced, Cover Quality, or Secret Recovery
- **Advanced Parameters**: Fine-tune model behavior

### Batch Operations
- **Sequential Pairing**: Match images in order
- **Random Pairing**: Random combinations for testing
- **All Combinations**: Process every possible pair

## 📊 Performance Metrics

The application provides comprehensive quality assessment:

- **PSNR (Peak Signal-to-Noise Ratio)**: Higher values indicate better quality
- **SSIM (Structural Similarity Index)**: Values closer to 1.0 indicate higher similarity
- **Processing Time**: Real-time performance monitoring
- **Memory Usage**: GPU/CPU memory tracking

## 🔧 Configuration

### Environment Variables
```bash
export STEGANOGRAPHY_DEVICE=cuda          # or cpu
export STEGANOGRAPHY_PORT=5000
export STEGANOGRAPHY_DEBUG=false
```

### Configuration Files
- `web_config.json`: Web application settings
- `config.json`: Model training configuration
- Custom model paths and parameters

## 🚀 API Endpoints

### Core Operations
- `POST /upload` - Embed secret image
- `POST /extract` - Extract secret image
- `POST /batch_process` - Start batch operation
- `GET /batch_status/<id>` - Check batch progress

### Training & Evaluation
- `POST /start_training` - Begin model training
- `GET /training_status/<id>` - Training progress
- `POST /evaluate_model` - Run model evaluation

### System Information
- `GET /model_info` - Model information
- `GET /system_stats` - System statistics
- `GET /download/<filename>` - Download results

## 🎯 Use Cases

### Personal Use
- Secure image communication
- Digital watermarking for copyright
- Privacy protection for sensitive images

### Professional Applications
- Forensic image analysis
- Secure document transmission
- Digital rights management
- Research and development

### Educational
- Computer vision research
- Steganography algorithm study
- Deep learning experimentation

## 🔒 Security Features

- **Advanced Encryption**: Military-grade steganography algorithms
- **Multiple Security Levels**: Choose appropriate security for your needs
- **Robustness Testing**: Resistance against common attacks
- **Secure File Handling**: Automatic cleanup of temporary files

## 🏗️ Architecture

### Frontend
- **Alpine.js**: Reactive JavaScript framework
- **Modern CSS**: Custom design system with animations
- **Progressive Web App**: Offline-capable functionality
- **Responsive Design**: Mobile-first approach

### Backend
- **Flask**: Python web framework
- **PyTorch**: Deep learning operations
- **Threading**: Background processing
- **REST API**: Clean, documented endpoints

### Models
- **Convolutional Autoencoders**: Core steganography models
- **Multiple Architectures**: Optimized for different use cases
- **Transfer Learning**: Pre-trained model support
- **Custom Training**: Train on your own datasets

## 📈 Performance Optimization

### GPU Acceleration
- **CUDA Support**: Full GPU acceleration
- **Batch Processing**: Efficient memory usage
- **Mixed Precision**: Faster training and inference
- **Memory Management**: Automatic GPU memory optimization

### Scalability
- **Background Processing**: Non-blocking operations
- **Queue Management**: Handle multiple concurrent requests
- **Caching**: Intelligent result caching
- **Load Balancing**: Ready for production deployment

## 🛠️ Development

### Project Structure
```
├── enhanced_web_app.py      # Enhanced backend application
├── templates/
│   ├── enhanced_index.html  # Modern web interface
│   └── index.html          # Classic interface
├── models.py               # Core steganography models
├── model_variants.py       # Alternative architectures
├── trainer.py             # Training utilities
├── batch_process.py       # Batch processing
├── launch_enhanced_app.py  # Application launcher
└── requirements.txt       # Dependencies
```

### Adding New Features
1. **Backend**: Add new routes in `enhanced_web_app.py`
2. **Frontend**: Extend the Alpine.js application
3. **Models**: Add new architectures in `model_variants.py`
4. **Processing**: Extend batch operations as needed

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- PyTorch team for the deep learning framework
- Flask community for the web framework
- Alpine.js for the reactive frontend framework
- All contributors and users of this project

## 📞 Support

- **Issues**: Report bugs on GitHub Issues
- **Discussions**: Join community discussions
- **Documentation**: Comprehensive guides available
- **Email**: Contact for commercial support

---

**🎭 Transform your images with invisible secrets using the power of deep learning!**
