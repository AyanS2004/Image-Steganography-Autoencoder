#!/usr/bin/env python3
"""
Simple demo script that works without heavy dependencies.
Tests the project structure and basic functionality.
"""

import os
import json
from datetime import datetime

def check_project_structure():
    """Check if all project files are present"""
    print("🎭 Digital Watermarking Project - Simple Demo")
    print("=" * 50)
    
    required_files = [
        'models.py',
        'dataset.py', 
        'trainer.py',
        'train.py',
        'inference.py',
        'demo.py',
        'evaluate.py',
        'batch_process.py',
        'web_app.py',
        'model_variants.py',
        'optimization.py',
        'requirements.txt',
        'README.md'
    ]
    
    print("📁 Checking project structure...")
    missing_files = []
    
    for file in required_files:
        if os.path.exists(file):
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file}")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n⚠️  Missing files: {missing_files}")
        return False
    else:
        print("\n🎉 All core files present!")
        return True

def check_directories():
    """Check if directories are created"""
    print("\n📂 Checking directories...")
    
    directories = [
        'checkpoints',
        'visualizations', 
        'results',
        'data',
        'sample_images'
    ]
    
    for directory in directories:
        if os.path.exists(directory):
            print(f"   ✅ {directory}/")
        else:
            print(f"   ❌ {directory}/ (will be created when needed)")

def analyze_code_structure():
    """Analyze the codebase"""
    print("\n🔍 Code analysis...")
    
    total_lines = 0
    file_stats = {}
    
    for file in os.listdir('.'):
        if file.endswith('.py'):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    lines = len(f.readlines())
                    total_lines += lines
                    file_stats[file] = lines
            except:
                file_stats[file] = 0
    
    print(f"   📊 Total Python files: {len(file_stats)}")
    print(f"   📊 Total lines of code: {total_lines:,}")
    
    # Show largest files
    sorted_files = sorted(file_stats.items(), key=lambda x: x[1], reverse=True)
    print("\n   📄 Largest files:")
    for file, lines in sorted_files[:5]:
        print(f"      {file}: {lines} lines")

def test_configurations():
    """Test configuration files"""
    print("\n⚙️  Testing configurations...")
    
    config_files = ['config.json', 'web_config.json']
    
    for config_file in config_files:
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                print(f"   ✅ {config_file} - Valid JSON")
                
                # Show some config details
                if config_file == 'config.json' and 'training' in config:
                    print(f"      - Training epochs: {config['training'].get('epochs', 'N/A')}")
                    print(f"      - Batch size: {config['training'].get('batch_size', 'N/A')}")
                elif config_file == 'web_config.json' and 'server' in config:
                    print(f"      - Server port: {config['server'].get('port', 'N/A')}")
                    
            except json.JSONDecodeError:
                print(f"   ❌ {config_file} - Invalid JSON")
            except Exception as e:
                print(f"   ❌ {config_file} - Error: {e}")
        else:
            print(f"   ⚠️  {config_file} - Not found")

def show_usage_examples():
    """Show usage examples"""
    print("\n🚀 Usage Examples:")
    print("\n   1. Install dependencies:")
    print("      pip install -r requirements.txt")
    
    print("\n   2. Run quick demo:")
    print("      python demo.py --mode inference --epochs 5")
    
    print("\n   3. Start web interface:")
    print("      python web_app.py")
    
    print("\n   4. Train a model:")
    print("      python train.py --use_cifar --epochs 10 --batch_size 16")
    
    print("\n   5. Evaluate model:")
    print("      python evaluate.py --checkpoint checkpoints/best_model.pth")
    
    print("\n   6. Batch processing:")
    print("      python batch_process.py --mode embed --checkpoint checkpoints/best_model.pth")

def check_sample_images():
    """Check sample images"""
    print("\n🖼️  Sample images:")
    
    sample_dir = 'sample_images'
    if os.path.exists(sample_dir):
        images = [f for f in os.listdir(sample_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"   📊 Found {len(images)} sample images")
        for img in images:
            print(f"      - {img}")
    else:
        print("   ⚠️  No sample images directory found")
        print("      Run: python setup.py to create sample images")

def create_status_report():
    """Create a status report"""
    report = {
        'timestamp': datetime.now().isoformat(),
        'project_name': 'Digital Watermarking System',
        'status': 'Initialized',
        'core_files_present': check_project_structure(),
        'total_python_files': len([f for f in os.listdir('.') if f.endswith('.py')]),
        'directories_created': len([d for d in ['checkpoints', 'visualizations', 'results', 'data'] if os.path.exists(d)]),
        'next_steps': [
            'Fix NumPy compatibility if needed',
            'Run demo.py for quick test', 
            'Start web_app.py for GUI',
            'Train first model with train.py'
        ]
    }
    
    with open('project_status.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Status report saved to: project_status.json")

def main():
    """Main demo function"""
    
    # Run all checks
    structure_ok = check_project_structure()
    check_directories()
    analyze_code_structure()
    test_configurations()
    check_sample_images()
    show_usage_examples()
    create_status_report()
    
    print("\n" + "=" * 50)
    if structure_ok:
        print("🎉 Project is ready to use!")
        print("\n💡 To get started:")
        print("   1. Fix NumPy compatibility (see INSTALL.md)")
        print("   2. Run: python demo.py --mode inference")
        print("   3. Or start web interface: python web_app.py")
    else:
        print("⚠️  Project setup incomplete")
        print("   Run: python setup.py to complete setup")
    
    print("\nHappy experimenting with Digital Watermarking! 🎭")

if __name__ == '__main__':
    main()
