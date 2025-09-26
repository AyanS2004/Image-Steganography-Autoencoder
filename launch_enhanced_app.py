#!/usr/bin/env python3
"""
Launch script for the Enhanced Digital Watermarking Web Application.
This script sets up and runs the comprehensive steganography suite.
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'torch', 'torchvision', 'flask', 'PIL', 'numpy', 
        'skimage', 'matplotlib', 'tqdm'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("Please install them using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ All dependencies are installed")
    return True

def setup_directories():
    """Create necessary directories"""
    directories = [
        'web_uploads',
        'web_results', 
        'batch_results',
        'training_results',
        'checkpoints',
        'templates',
        'static'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    print("✅ Directories created successfully")

def check_models():
    """Check if model files exist"""
    model_paths = [
        'checkpoints/checkpoint_epoch_5.pth',
        'checkpoints/best_model.pth'
    ]
    
    existing_models = []
    for path in model_paths:
        if os.path.exists(path):
            existing_models.append(path)
    
    if existing_models:
        print(f"✅ Found model checkpoints: {', '.join(existing_models)}")
    else:
        print("⚠️  No pre-trained models found. The app will use untrained models.")
        print("   To get better results, train a model first using:")
        print("   python train.py --epochs 50 --batch_size 32 --use_cifar")
    
    return len(existing_models) > 0

def run_enhanced_app(args):
    """Run the enhanced web application"""
    print("\n🎭 Starting Enhanced Digital Watermarking Suite")
    print("=" * 60)
    
    # Import and run the enhanced app
    try:
        from enhanced_web_app import main as run_enhanced
        sys.argv = [
            'enhanced_web_app.py',
            '--host', args.host,
            '--port', str(args.port),
            '--device', args.device
        ]
        if args.debug:
            sys.argv.append('--debug')
        
        run_enhanced()
        
    except ImportError:
        print("❌ Enhanced app not found. Falling back to original app.")
        run_original_app(args)
    except Exception as e:
        print(f"❌ Error running enhanced app: {e}")
        print("Falling back to original app...")
        run_original_app(args)

def run_original_app(args):
    """Run the original web application as fallback"""
    try:
        from web_app import main as run_original
        sys.argv = [
            'web_app.py',
            '--host', args.host,
            '--port', str(args.port),
            '--device', args.device
        ]
        if args.debug:
            sys.argv.append('--debug')
        
        run_original()
        
    except Exception as e:
        print(f"❌ Error running original app: {e}")
        sys.exit(1)

def run_quick_demo():
    """Run a quick demonstration"""
    print("\n🚀 Running Quick Demo")
    print("=" * 30)
    
    try:
        subprocess.run([sys.executable, 'demo.py', '--mode', 'both', '--epochs', '5'], 
                      check=True)
        print("✅ Demo completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Demo failed: {e}")
    except FileNotFoundError:
        print("❌ demo.py not found. Skipping demo.")

def main():
    parser = argparse.ArgumentParser(
        description='Enhanced Digital Watermarking Web Application Launcher',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python launch_enhanced_app.py                    # Start with default settings
  python launch_enhanced_app.py --port 8080        # Run on port 8080
  python launch_enhanced_app.py --demo             # Run demo first
  python launch_enhanced_app.py --setup-only       # Just setup, don't run
  python launch_enhanced_app.py --device cpu       # Force CPU usage
        """
    )
    
    parser.add_argument('--host', type=str, default='127.0.0.1',
                       help='Host to run the server on (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=5000,
                       help='Port to run the server on (default: 5000)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu', 'auto'],
                       help='Device to use for processing (default: cuda)')
    parser.add_argument('--debug', action='store_true',
                       help='Run in debug mode')
    parser.add_argument('--demo', action='store_true',
                       help='Run quick demo before starting the app')
    parser.add_argument('--setup-only', action='store_true',
                       help='Only perform setup, do not start the app')
    parser.add_argument('--check-only', action='store_true',
                       help='Only check dependencies and setup, do not run')
    
    args = parser.parse_args()
    
    print("🎭 Enhanced Digital Watermarking Suite Launcher")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Setup directories
    setup_directories()
    
    # Check models
    has_models = check_models()
    
    if args.check_only:
        print("\n✅ System check completed!")
        return
    
    # Run demo if requested
    if args.demo:
        run_quick_demo()
    
    if args.setup_only:
        print("\n✅ Setup completed!")
        return
    
    # Display startup information
    print(f"\n📊 Configuration:")
    print(f"   Host: {args.host}")
    print(f"   Port: {args.port}")
    print(f"   Device: {args.device}")
    print(f"   Debug: {args.debug}")
    print(f"   Models Available: {'Yes' if has_models else 'No'}")
    
    print(f"\n🌐 Web Interface will be available at:")
    print(f"   Main App: http://{args.host}:{args.port}/")
    print(f"   Classic UI: http://{args.host}:{args.port}/classic")
    
    if not has_models:
        print(f"\n⚠️  Recommendation: Train a model first for better results:")
        print(f"   python train.py --epochs 50 --batch_size 32 --use_cifar")
    
    print(f"\n🚀 Starting application...")
    print(f"   Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        # Small delay to let user read the information
        time.sleep(2)
        
        # Run the application
        run_enhanced_app(args)
        
    except KeyboardInterrupt:
        print(f"\n\n👋 Application stopped by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
