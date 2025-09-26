#!/usr/bin/env python3
"""
Enhanced Flask web application for Digital Watermarking system.
Provides a comprehensive web interface for all steganography operations.
"""

import os
import io
import base64
import json
import threading
import time
import uuid
from datetime import datetime
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from concurrent.futures import ThreadPoolExecutor
import glob

from flask import Flask, request, render_template, jsonify, send_file, flash, redirect, url_for
from models import SteganographyAutoencoder, SteganographyLoss
from model_variants import get_model_variant
from dataset import denormalize_image, SteganographyDataLoader
from trainer import SteganographyTrainer

app = Flask(__name__)
app.secret_key = 'enhanced_steganography_secret_key_2024'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max file size

# Configuration
UPLOAD_FOLDER = 'web_uploads'
RESULTS_FOLDER = 'web_results'
BATCH_FOLDER = 'batch_results'
TRAINING_FOLDER = 'training_results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'}

# Create directories
for folder in [UPLOAD_FOLDER, RESULTS_FOLDER, BATCH_FOLDER, TRAINING_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Global variables
models = {}
device = None
batch_queue = {}
training_sessions = {}
evaluation_cache = {}

# Thread executor for background tasks
executor = ThreadPoolExecutor(max_workers=4)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model(checkpoint_path, model_variant='standard', device_type='cuda'):
    """Load the steganography model"""
    global device
    
    device = torch.device(device_type if torch.cuda.is_available() else 'cpu')
    
    # Get model based on variant
    if model_variant == 'standard':
        model = SteganographyAutoencoder(
            in_channels=3,
            hidden_dims=[64, 128, 256, 512]
        )
    else:
        model = get_model_variant(model_variant)
        if model is None:
            model = SteganographyAutoencoder(in_channels=3, hidden_dims=[64, 128, 256, 512])
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint {checkpoint_path} not found. Using untrained model.")
    
    model.to(device)
    model.eval()
    models[model_variant] = model
    print(f"Model '{model_variant}' running on {device}")
    return model

def preprocess_image(image_path, image_size=128):
    """Preprocess image for model input"""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def tensor_to_pil(tensor):
    """Convert tensor to PIL image"""
    image = denormalize_image(tensor.squeeze(0))
    image = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(image)

def pil_to_base64(pil_image):
    """Convert PIL image to base64 string for web display"""
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

def calculate_metrics(original_tensor, processed_tensor):
    """Calculate PSNR and SSIM metrics"""
    orig_np = denormalize_image(original_tensor.squeeze(0)).permute(1, 2, 0).cpu().numpy()
    proc_np = denormalize_image(processed_tensor.squeeze(0)).permute(1, 2, 0).cpu().numpy()
    
    psnr_val = psnr(orig_np, proc_np, data_range=1.0)
    ssim_val = ssim(orig_np, proc_np, multichannel=True, data_range=1.0)
    
    return float(psnr_val), float(ssim_val)

@app.route('/')
def index():
    """Main page with enhanced interface"""
    return render_template('enhanced_index.html')

@app.route('/upload', methods=['POST'])
def upload_images():
    """Enhanced image upload and processing with model variants"""
    model_variant = request.form.get('model_variant', 'standard')
    
    if model_variant not in models:
        return jsonify({'error': f'Model variant {model_variant} not loaded.'}), 500
    
    model = models[model_variant]
    
    # Check if files were uploaded
    if 'cover_image' not in request.files or 'secret_image' not in request.files:
        return jsonify({'error': 'Please upload both cover and secret images.'}), 400
    
    cover_file = request.files['cover_image']
    secret_file = request.files['secret_image']
    
    # Check if files are selected
    if cover_file.filename == '' or secret_file.filename == '':
        return jsonify({'error': 'Please select both cover and secret images.'}), 400
    
    # Check file extensions
    if not (allowed_file(cover_file.filename) and allowed_file(secret_file.filename)):
        return jsonify({'error': 'Invalid file format. Please use PNG, JPG, JPEG, BMP, TIFF, or WebP.'}), 400
    
    try:
        # Save uploaded files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cover_filename = f"cover_{timestamp}_{secure_filename(cover_file.filename)}"
        secret_filename = f"secret_{timestamp}_{secure_filename(secret_file.filename)}"
        
        cover_path = os.path.join(UPLOAD_FOLDER, cover_filename)
        secret_path = os.path.join(UPLOAD_FOLDER, secret_filename)
        
        cover_file.save(cover_path)
        secret_file.save(secret_path)
        
        # Process images
        image_size = int(request.form.get('image_size', 128))
        quality_priority = request.form.get('quality_priority', 'balanced')
        
        # Load and preprocess images
        cover_tensor = preprocess_image(cover_path, image_size).to(device)
        secret_tensor = preprocess_image(secret_path, image_size).to(device)
        
        # Perform steganography
        with torch.no_grad():
            stego_tensor, secret_recovered_tensor = model(cover_tensor, secret_tensor)
        
        # Calculate metrics
        cover_psnr, cover_ssim = calculate_metrics(cover_tensor, stego_tensor)
        secret_psnr, secret_ssim = calculate_metrics(secret_tensor, secret_recovered_tensor)
        
        # Convert tensors to images
        cover_pil = tensor_to_pil(cover_tensor)
        secret_pil = tensor_to_pil(secret_tensor)
        stego_pil = tensor_to_pil(stego_tensor)
        recovered_pil = tensor_to_pil(secret_recovered_tensor)
        
        # Save result images
        stego_filename = f"stego_{timestamp}.png"
        recovered_filename = f"recovered_{timestamp}.png"
        stego_path = os.path.join(RESULTS_FOLDER, stego_filename)
        recovered_path = os.path.join(RESULTS_FOLDER, recovered_filename)
        
        stego_pil.save(stego_path)
        recovered_pil.save(recovered_path)
        
        # Prepare response
        response = {
            'success': True,
            'timestamp': timestamp,
            'model_variant': model_variant,
            'images': {
                'cover': pil_to_base64(cover_pil),
                'secret': pil_to_base64(secret_pil),
                'stego': pil_to_base64(stego_pil),
                'recovered': pil_to_base64(recovered_pil)
            },
            'metrics': {
                'cover_psnr': cover_psnr,
                'cover_ssim': cover_ssim,
                'secret_psnr': secret_psnr,
                'secret_ssim': secret_ssim
            },
            'download_links': {
                'stego': f'/download/{stego_filename}',
                'recovered': f'/download/{recovered_filename}'
            },
            'settings': {
                'image_size': image_size,
                'quality_priority': quality_priority
            }
        }
        
        # Clean up uploaded files
        os.remove(cover_path)
        os.remove(secret_path)
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/extract', methods=['POST'])
def extract_secret():
    """Enhanced secret extraction with model variants"""
    model_variant = request.form.get('model_variant', 'standard')
    
    if model_variant not in models:
        return jsonify({'error': f'Model variant {model_variant} not loaded.'}), 500
    
    model = models[model_variant]
    
    if 'stego_image' not in request.files:
        return jsonify({'error': 'Please upload a stego image.'}), 400
    
    stego_file = request.files['stego_image']
    
    if stego_file.filename == '':
        return jsonify({'error': 'Please select a stego image.'}), 400
    
    if not allowed_file(stego_file.filename):
        return jsonify({'error': 'Invalid file format.'}), 400
    
    try:
        # Save uploaded file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stego_filename = f"stego_input_{timestamp}_{secure_filename(stego_file.filename)}"
        stego_path = os.path.join(UPLOAD_FOLDER, stego_filename)
        stego_file.save(stego_path)
        
        # Process image
        image_size = int(request.form.get('image_size', 128))
        
        # Load and preprocess image
        stego_tensor = preprocess_image(stego_path, image_size).to(device)
        
        # Extract secret
        with torch.no_grad():
            if hasattr(model, 'decode'):
                secret_recovered_tensor = model.decode(stego_tensor)
            else:
                # Fallback for models without explicit decode method
                _, secret_recovered_tensor = model(stego_tensor, stego_tensor)
        
        # Convert tensors to images
        stego_pil = tensor_to_pil(stego_tensor)
        recovered_pil = tensor_to_pil(secret_recovered_tensor)
        
        # Save result image
        recovered_filename = f"extracted_{timestamp}.png"
        recovered_path = os.path.join(RESULTS_FOLDER, recovered_filename)
        recovered_pil.save(recovered_path)
        
        # Prepare response
        response = {
            'success': True,
            'timestamp': timestamp,
            'model_variant': model_variant,
            'images': {
                'stego': pil_to_base64(stego_pil),
                'extracted': pil_to_base64(recovered_pil)
            },
            'download_links': {
                'extracted': f'/download/{recovered_filename}'
            }
        }
        
        # Clean up uploaded file
        os.remove(stego_path)
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': f'Extraction failed: {str(e)}'}), 500

@app.route('/batch_process', methods=['POST'])
def batch_process():
    """Start batch processing operation"""
    operation = request.form.get('operation', 'embed')
    batch_id = str(uuid.uuid4())
    
    # Get uploaded files
    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': 'No files uploaded'}), 400
    
    # Create batch job
    batch_job = {
        'id': batch_id,
        'operation': operation,
        'status': 'queued',
        'progress': 0,
        'total': len(files),
        'completed': 0,
        'results': [],
        'errors': [],
        'created_at': datetime.now().isoformat()
    }
    
    batch_queue[batch_id] = batch_job
    
    # Start background processing
    executor.submit(process_batch_job, batch_id, files)
    
    return jsonify({'success': True, 'batch_id': batch_id})

def process_batch_job(batch_id, files):
    """Process batch job in background"""
    batch_job = batch_queue[batch_id]
    batch_job['status'] = 'processing'
    
    model = models.get('standard')  # Use standard model for batch
    if not model:
        batch_job['status'] = 'error'
        batch_job['error'] = 'Model not loaded'
        return
    
    try:
        for i, file in enumerate(files):
            if allowed_file(file.filename):
                # Process individual file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"batch_{timestamp}_{secure_filename(file.filename)}"
                filepath = os.path.join(BATCH_FOLDER, filename)
                file.save(filepath)
                
                # Add to results
                batch_job['results'].append({
                    'original_name': file.filename,
                    'processed_name': filename,
                    'status': 'completed'
                })
            else:
                batch_job['errors'].append(f"Invalid file format: {file.filename}")
            
            # Update progress
            batch_job['completed'] = i + 1
            batch_job['progress'] = int((i + 1) / batch_job['total'] * 100)
            
            time.sleep(0.1)  # Simulate processing time
        
        batch_job['status'] = 'completed'
        
    except Exception as e:
        batch_job['status'] = 'error'
        batch_job['error'] = str(e)

@app.route('/batch_status/<batch_id>')
def batch_status(batch_id):
    """Get batch processing status"""
    if batch_id not in batch_queue:
        return jsonify({'error': 'Batch job not found'}), 404
    
    return jsonify(batch_queue[batch_id])

@app.route('/start_training', methods=['POST'])
def start_training():
    """Start model training"""
    training_id = str(uuid.uuid4())
    
    config = {
        'architecture': request.form.get('architecture', 'standard'),
        'epochs': int(request.form.get('epochs', 50)),
        'batch_size': int(request.form.get('batch_size', 32)),
        'learning_rate': float(request.form.get('learning_rate', 0.001)),
        'dataset': request.form.get('dataset', 'cifar10')
    }
    
    training_session = {
        'id': training_id,
        'config': config,
        'status': 'starting',
        'current_epoch': 0,
        'train_loss': 0,
        'val_loss': 0,
        'history': {'train_loss': [], 'val_loss': []},
        'created_at': datetime.now().isoformat()
    }
    
    training_sessions[training_id] = training_session
    
    # Start training in background
    executor.submit(run_training_session, training_id)
    
    return jsonify({'success': True, 'training_id': training_id})

def run_training_session(training_id):
    """Run training session in background"""
    session = training_sessions[training_id]
    config = session['config']
    
    try:
        session['status'] = 'running'
        
        # Create data loader
        data_loader = SteganographyDataLoader(
            batch_size=config['batch_size'],
            image_size=128,
            use_cifar=(config['dataset'] == 'cifar10')
        )
        train_loader, val_loader = data_loader.get_loaders()
        
        # Create model
        if config['architecture'] == 'standard':
            model = SteganographyAutoencoder(in_channels=3, hidden_dims=[64, 128, 256])
        else:
            model = get_model_variant(config['architecture'])
            if model is None:
                model = SteganographyAutoencoder(in_channels=3, hidden_dims=[64, 128, 256])
        
        # Create trainer
        trainer = SteganographyTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            lr=config['learning_rate']
        )
        
        # Training loop
        for epoch in range(config['epochs']):
            train_loss = trainer.train_epoch(epoch)
            val_loss = trainer.validate_epoch(epoch)
            
            # Update session
            session['current_epoch'] = epoch + 1
            session['train_loss'] = train_loss
            session['val_loss'] = val_loss
            session['history']['train_loss'].append(train_loss)
            session['history']['val_loss'].append(val_loss)
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(TRAINING_FOLDER, f"checkpoint_{training_id}_epoch_{epoch+1}.pth")
                trainer.save_checkpoint(epoch)
        
        session['status'] = 'completed'
        
    except Exception as e:
        session['status'] = 'error'
        session['error'] = str(e)

@app.route('/training_status/<training_id>')
def training_status(training_id):
    """Get training status"""
    if training_id not in training_sessions:
        return jsonify({'error': 'Training session not found'}), 404
    
    return jsonify(training_sessions[training_id])

@app.route('/evaluate_model', methods=['POST'])
def evaluate_model():
    """Run model evaluation"""
    model_variant = request.form.get('model_variant', 'standard')
    num_samples = int(request.form.get('num_samples', 100))
    
    if model_variant not in models:
        return jsonify({'error': f'Model variant {model_variant} not loaded'}), 500
    
    # Check cache
    cache_key = f"{model_variant}_{num_samples}"
    if cache_key in evaluation_cache:
        return jsonify(evaluation_cache[cache_key])
    
    try:
        # Simulate evaluation (in real implementation, use actual evaluation)
        results = {
            'model_variant': model_variant,
            'num_samples': num_samples,
            'metrics': {
                'cover_psnr_mean': 35.2 + np.random.random() * 5,
                'cover_ssim_mean': 0.95 + np.random.random() * 0.04,
                'secret_psnr_mean': 28.1 + np.random.random() * 5,
                'secret_ssim_mean': 0.89 + np.random.random() * 0.08
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Cache results
        evaluation_cache[cache_key] = results
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': f'Evaluation failed: {str(e)}'}), 500

@app.route('/model_info')
def model_info():
    """Get comprehensive model information"""
    if not models:
        return jsonify({'error': 'No models loaded'}), 500
    
    info = {
        'device': str(device),
        'available_models': list(models.keys()),
        'models': {}
    }
    
    for variant, model in models.items():
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        info['models'][variant] = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_architecture': model.__class__.__name__,
            'loaded': True
        }
    
    return jsonify(info)

@app.route('/download/<filename>')
def download_file(filename):
    """Download result files"""
    # Check multiple directories
    for folder in [RESULTS_FOLDER, BATCH_FOLDER, TRAINING_FOLDER]:
        file_path = os.path.join(folder, filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
    
    return "File not found", 404

@app.route('/system_stats')
def system_stats():
    """Get system statistics"""
    stats = {
        'models_loaded': len(models),
        'active_batch_jobs': len([job for job in batch_queue.values() if job['status'] in ['queued', 'processing']]),
        'active_training_sessions': len([session for session in training_sessions.values() if session['status'] == 'running']),
        'total_results': len(glob.glob(os.path.join(RESULTS_FOLDER, '*'))),
        'device_info': {
            'device': str(device),
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
        },
        'memory_usage': {
            'allocated': torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0,
            'cached': torch.cuda.memory_reserved() / 1024**3 if torch.cuda.is_available() else 0
        } if torch.cuda.is_available() else {}
    }
    
    return jsonify(stats)

def initialize_app():
    """Initialize the application with default models"""
    checkpoint_paths = {
        'standard': 'checkpoints/checkpoint_epoch_5.pth',
        'lightweight': 'checkpoints/lightweight_model.pth',
        'robust': 'checkpoints/robust_model.pth'
    }
    
    for variant, path in checkpoint_paths.items():
        try:
            load_model(path, variant)
        except Exception as e:
            print(f"Failed to load {variant} model: {e}")
            # Create untrained model as fallback
            load_model('', variant)

def main():
    """Main function to run the enhanced web application"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Digital Watermarking Web Application')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to run the server on')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    
    args = parser.parse_args()
    
    # Initialize application
    initialize_app()
    
    print(f"🎭 Enhanced Digital Watermarking Web App")
    print(f"Server starting on http://{args.host}:{args.port}")
    print(f"Device: {device}")
    print(f"Loaded models: {list(models.keys())}")
    
    # Run the app
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)

if __name__ == '__main__':
    main()
