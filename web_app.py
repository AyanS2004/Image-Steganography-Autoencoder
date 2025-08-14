#!/usr/bin/env python3
"""
Flask web application for Digital Watermarking system.
Provides an easy-to-use web interface for image steganography.
"""

import os
import io
import base64
import json
from datetime import datetime
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from flask import Flask, request, render_template, jsonify, send_file, flash, redirect, url_for
from models import SteganographyAutoencoder
from dataset import denormalize_image

app = Flask(__name__)
app.secret_key = 'steganography_secret_key_2024'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Configuration
UPLOAD_FOLDER = 'web_uploads'
RESULTS_FOLDER = 'web_results'
# Additional artifact directories
CUDA_SAFE_RESULTS_FOLDER = 'cuda_safe_results'
EVAL_RESULTS_FOLDER = 'evaluation_results'
OPTIMIZED_RESULTS_FOLDER = 'optimized_results'
CHECKPOINTS_FOLDER = 'checkpoints'
CUDA_SAFE_CHECKPOINTS_FOLDER = 'cuda_safe_checkpoints'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'gif', 'webp'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(CUDA_SAFE_RESULTS_FOLDER, exist_ok=True)
os.makedirs(EVAL_RESULTS_FOLDER, exist_ok=True)
os.makedirs(OPTIMIZED_RESULTS_FOLDER, exist_ok=True)

# Global model variable
model = None
device = None
current_checkpoint_path = None


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def determine_model_architecture(checkpoint_path):
    """Determine the model architecture from checkpoint path or content"""
    # First try to determine from path
    if 'cuda_safe_checkpoints' in checkpoint_path:
        print(f"Detected CUDA-safe checkpoint from path: {checkpoint_path}")
        return [64, 128, 256]
    
    # If not CUDA-safe, try to load the checkpoint and determine from state dict
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            state_dict = checkpoint['model_state_dict']
            
            # Check for specific layer patterns to determine architecture
            # CUDA-safe checkpoints have fewer layers and different fusion layer sizes
            fusion_keys = [k for k in state_dict.keys() if k.startswith('fusion.')]
            
            # Check fusion layer sizes to determine architecture
            if 'fusion.0.weight' in state_dict:
                fusion_weight_shape = state_dict['fusion.0.weight'].shape
                print(f"Fusion layer shape: {fusion_weight_shape}")
                
                # CUDA-safe: fusion.0.weight has shape [256, 512, 3, 3]
                # Standard: fusion.0.weight has shape [512, 1024, 3, 3]
                if fusion_weight_shape[0] == 256:
                    print(f"Detected CUDA-safe architecture from fusion layer shape")
                    return [64, 128, 256]
                elif fusion_weight_shape[0] == 512:
                    print(f"Detected standard architecture from fusion layer shape")
                    return [64, 128, 256, 512]
            
            # Fallback: Count the number of layers in cover_encoder
            cover_encoder_keys = [k for k in state_dict.keys() if k.startswith('cover_encoder.')]
            max_layer = 0
            for key in cover_encoder_keys:
                try:
                    layer_num = int(key.split('.')[1])
                    max_layer = max(max_layer, layer_num)
                except (ValueError, IndexError):
                    continue
            
            print(f"Max cover_encoder layer: {max_layer}")
            # Based on the layer count, determine hidden_dims
            if max_layer <= 15:  # CUDA-safe architecture
                return [64, 128, 256]
            else:  # Standard architecture
                return [64, 128, 256, 512]
        except Exception as e:
            print(f"Could not determine architecture from checkpoint: {e}")
    
    # Default to standard architecture
    print(f"Using default standard architecture for: {checkpoint_path}")
    return [64, 128, 256, 512]

def load_model(checkpoint_path, device_type='cuda'):
    """Load the steganography model"""
    global model, device
    
    device = torch.device(device_type if torch.cuda.is_available() else 'cpu')
    
    # Determine model architecture
    hidden_dims = determine_model_architecture(checkpoint_path)
    print(f"Loading model with hidden_dims: {hidden_dims}")
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Try loading with determined architecture first
        try:
            model = SteganographyAutoencoder(
                in_channels=3,
                hidden_dims=hidden_dims
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded from {checkpoint_path}")
        except Exception as e:
            print(f"Failed to load with determined architecture: {e}")
            
            # Try alternative architectures in order of likelihood
            alternative_architectures = [
                [64, 128, 256],  # CUDA-safe (most likely for CUDA-safe checkpoints)
                [64, 128, 256, 512],  # Standard
                [32, 64, 128],  # Lightweight
                [128, 256, 512, 1024]  # High capacity
            ]
            
            # If this is a CUDA-safe checkpoint, prioritize CUDA-safe architecture
            if 'cuda_safe_checkpoints' in checkpoint_path:
                # Move CUDA-safe architecture to front
                alternative_architectures.insert(0, alternative_architectures.pop(0))
                print("Prioritizing CUDA-safe architecture for CUDA-safe checkpoint")
            
            for alt_dims in alternative_architectures:
                if alt_dims == hidden_dims:
                    continue  # Skip the one we already tried
                    
                try:
                    print(f"Trying alternative architecture: {alt_dims}")
                    model = SteganographyAutoencoder(
                        in_channels=3,
                        hidden_dims=alt_dims
                    )
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"Successfully loaded with architecture: {alt_dims}")
                    break
                except Exception as alt_e:
                    print(f"Failed with {alt_dims}: {alt_e}")
                    continue
            else:
                print("All architectures failed. Using untrained model.")
                model = SteganographyAutoencoder(
                    in_channels=3,
                    hidden_dims=hidden_dims
                )
    else:
        print(f"Warning: Checkpoint {checkpoint_path} not found. Using untrained model.")
        model = SteganographyAutoencoder(
            in_channels=3,
            hidden_dims=hidden_dims
        )
    
    model.to(device)
    model.eval()
    print(f"Model running on {device}")
    global current_checkpoint_path
    current_checkpoint_path = checkpoint_path


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
    """Convert tensor to PIL image with NaN handling"""
    try:
        image = denormalize_image(tensor.squeeze(0))
        image_np = image.permute(1, 2, 0).cpu().numpy()
        
        # Handle NaN and inf values
        if np.any(np.isnan(image_np)):
            print("Warning: NaN values detected in tensor, replacing with 0")
            image_np = np.nan_to_num(image_np, nan=0.0, posinf=1.0, neginf=0.0)
        
        if np.any(np.isinf(image_np)):
            print("Warning: Inf values detected in tensor, replacing with bounds")
            image_np = np.nan_to_num(image_np, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Clip values to valid range [0, 1]
        image_np = np.clip(image_np, 0.0, 1.0)
        
        # Convert to uint8
        image_uint8 = (image_np * 255).astype(np.uint8)
        
        return Image.fromarray(image_uint8)
    except Exception as e:
        print(f"Error converting tensor to PIL: {e}")
        # Return a black image as fallback
        return Image.new('RGB', (64, 64), (0, 0, 0))


def pil_to_base64(pil_image):
    """Convert PIL image to base64 string for web display"""
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


def ensure_valid_metric(value):
    """Ensure metric is a valid finite number"""
    if not isinstance(value, (int, float)) or not np.isfinite(value):
        return 0.0
    return float(value)


def calculate_metrics(original_tensor, processed_tensor):
    """Calculate PSNR and SSIM metrics with NaN handling"""
    try:
        orig_np = denormalize_image(original_tensor.squeeze(0)).permute(1, 2, 0).cpu().numpy()
        proc_np = denormalize_image(processed_tensor.squeeze(0)).permute(1, 2, 0).cpu().numpy()
        
        # Ensure images are valid (no NaN or inf values)
        if np.any(np.isnan(orig_np)) or np.any(np.isnan(proc_np)):
            print("Warning: NaN values detected in images, using fallback metrics")
            return 0.0, 0.0
        
        if np.any(np.isinf(orig_np)) or np.any(np.isinf(proc_np)):
            print("Warning: Inf values detected in images, using fallback metrics")
            return 0.0, 0.0
        
        # Calculate PSNR
        psnr_val = psnr(orig_np, proc_np, data_range=1.0)
        
        # Handle NaN in PSNR
        if np.isnan(psnr_val) or np.isinf(psnr_val):
            print("Warning: PSNR calculation returned NaN/Inf, using fallback")
            psnr_val = 0.0
        
        # Calculate SSIM with proper window size for small images
        try:
            # Get image dimensions
            height, width = orig_np.shape[:2]
            
            # Determine appropriate window size (must be odd and smaller than image)
            min_dim = min(height, width)
            if min_dim < 7:
                # For very small images, use a minimal window size
                win_size = 3
            elif min_dim < 11:
                # For small images, use a smaller window
                win_size = 5
            else:
                # For larger images, use default window size
                win_size = 7
            
            # Ensure window size is odd
            if win_size % 2 == 0:
                win_size -= 1
            
            # Calculate SSIM with appropriate parameters
            ssim_val = ssim(orig_np, proc_np, 
                           win_size=win_size,
                           channel_axis=2,  # Specify channel axis for RGB images
                           data_range=1.0)
            
            # Handle NaN in SSIM
            if np.isnan(ssim_val) or np.isinf(ssim_val):
                print("Warning: SSIM calculation returned NaN/Inf, using fallback")
                ssim_val = 0.0
                
        except Exception as e:
            print(f"SSIM calculation failed: {e}")
            # Fallback: use a simple correlation-based similarity
            try:
                ssim_val = np.corrcoef(orig_np.flatten(), proc_np.flatten())[0, 1]
                if np.isnan(ssim_val) or np.isinf(ssim_val):
                    ssim_val = 0.0
            except:
                ssim_val = 0.0
        
        # Ensure both values are finite numbers
        psnr_val = float(psnr_val) if np.isfinite(psnr_val) else 0.0
        ssim_val = float(ssim_val) if np.isfinite(ssim_val) else 0.0
        
        return psnr_val, ssim_val
        
    except Exception as e:
        print(f"Metrics calculation failed: {e}")
        return 0.0, 0.0


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_images():
    """Handle image upload and processing"""
    if model is None:
        return jsonify({'error': 'Model not loaded. Please check server configuration.'}), 500
    
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
        return jsonify({'error': 'Invalid file format. Please use PNG, JPG, JPEG, BMP, or TIFF.'}), 400
    
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
        
        # Load and preprocess images
        cover_tensor = preprocess_image(cover_path, image_size).to(device)
        secret_tensor = preprocess_image(secret_path, image_size).to(device)
        
        # Perform steganography
        with torch.no_grad():
            stego_tensor, secret_recovered_tensor = model(cover_tensor, secret_tensor)
        
        # Calculate metrics
        cover_psnr, cover_ssim = calculate_metrics(cover_tensor, stego_tensor)
        secret_psnr, secret_ssim = calculate_metrics(secret_tensor, secret_recovered_tensor)
        
        # Ensure all metrics are valid numbers
        cover_psnr = ensure_valid_metric(cover_psnr)
        cover_ssim = ensure_valid_metric(cover_ssim)
        secret_psnr = ensure_valid_metric(secret_psnr)
        secret_ssim = ensure_valid_metric(secret_ssim)
        
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
    """Extract secret image from stego image"""
    if model is None:
        return jsonify({'error': 'Model not loaded. Please check server configuration.'}), 500
    
    if 'stego_image' not in request.files:
        return jsonify({'error': 'Please upload a stego image.'}), 400
    
    stego_file = request.files['stego_image']
    
    if stego_file.filename == '':
        return jsonify({'error': 'Please select a stego image.'}), 400
    
    if not allowed_file(stego_file.filename):
        return jsonify({'error': 'Invalid file format. Please use PNG, JPG, JPEG, BMP, or TIFF.'}), 400
    
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
            secret_recovered_tensor = model.decode(stego_tensor)
        
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


@app.route('/download/<filename>')
def download_file(filename):
    """Download result files"""
    file_path = os.path.join(RESULTS_FOLDER, filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return "File not found", 404


def _list_dir_safe(base_dir, allowed_ext=None):
    entries = []
    if not os.path.exists(base_dir):
        return entries
    for name in os.listdir(base_dir):
        path = os.path.join(base_dir, name)
        if not os.path.isfile(path):
            continue
        ext = name.rsplit('.', 1)[-1].lower() if '.' in name else ''
        if allowed_ext and ext not in allowed_ext:
            continue
        stat = os.stat(path)
        entries.append({
            'name': name,
            'size_bytes': stat.st_size,
            'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
        })
    # Sort by modified desc
    entries.sort(key=lambda x: x['modified'], reverse=True)
    return entries


@app.route('/artifacts')
def list_artifacts():
    """List artifacts across results and reports directories."""
    any_ext = None
    data = {
        'web_results': _list_dir_safe(RESULTS_FOLDER, any_ext),
        'results': _list_dir_safe('results', any_ext),
        'cuda_safe_results': _list_dir_safe(CUDA_SAFE_RESULTS_FOLDER, any_ext),
        'evaluation_results': _list_dir_safe(EVAL_RESULTS_FOLDER, any_ext),
        'optimized_results': _list_dir_safe(OPTIMIZED_RESULTS_FOLDER, any_ext),
    }
    # Attach URLs for downloading/serving
    def attach_urls(folder_key, base_dir):
        for item in data[folder_key]:
            item['url'] = url_for('serve_artifact', folder=folder_key, filename=item['name'])
    attach_urls('web_results', RESULTS_FOLDER)
    attach_urls('results', 'results')
    attach_urls('cuda_safe_results', CUDA_SAFE_RESULTS_FOLDER)
    attach_urls('evaluation_results', EVAL_RESULTS_FOLDER)
    attach_urls('optimized_results', OPTIMIZED_RESULTS_FOLDER)
    return jsonify({'success': True, 'artifacts': data})


FOLDER_MAP = {
    'web_results': RESULTS_FOLDER,
    'results': 'results',
    'cuda_safe_results': CUDA_SAFE_RESULTS_FOLDER,
    'evaluation_results': EVAL_RESULTS_FOLDER,
    'optimized_results': OPTIMIZED_RESULTS_FOLDER,
}


@app.route('/artifact/<folder>/<path:filename>')
def serve_artifact(folder, filename):
    """Serve artifacts from whitelisted folders."""
    if folder not in FOLDER_MAP:
        return "Invalid folder", 400
    base_dir = FOLDER_MAP[folder]
    base_abs = os.path.abspath(base_dir)
    safe_path_abs = os.path.abspath(os.path.join(base_dir, filename))
    try:
        common = os.path.commonpath([safe_path_abs, base_abs])
    except Exception:
        return "Invalid path", 400
    if common != base_abs:
        return "Invalid path", 400
    if not os.path.exists(safe_path_abs):
        return "File not found", 404
    return send_file(safe_path_abs, as_attachment=True)


@app.route('/checkpoints', methods=['GET'])
def list_checkpoints():
    """List available checkpoints including CUDA-safe ones."""
    def collect(dir_path):
        files = []
        if os.path.exists(dir_path):
            for name in os.listdir(dir_path):
                if not name.lower().endswith('.pth'):
                    continue
                path = os.path.join(dir_path, name)
                if os.path.isfile(path):
                    files.append({
                        'name': name,
                        'relative_path': os.path.join(dir_path, name).replace('\\', '/'),
                        'modified': datetime.fromtimestamp(os.stat(path).st_mtime).isoformat(),
                        'size_bytes': os.stat(path).st_size,
                        'active': (current_checkpoint_path == os.path.join(dir_path, name))
                    })
        files.sort(key=lambda x: x['modified'], reverse=True)
        return files
    return jsonify({
        'success': True,
        'current': current_checkpoint_path,
        'checkpoints': collect(CHECKPOINTS_FOLDER),
        'cuda_safe_checkpoints': collect(CUDA_SAFE_CHECKPOINTS_FOLDER)
    })


@app.route('/switch_checkpoint', methods=['POST'])
def switch_checkpoint():
    """Switch the active checkpoint and reload the model."""
    try:
        data = request.get_json(force=True)
        rel_path = data.get('path')
        if not rel_path:
            return jsonify({'success': False, 'error': 'Missing path'}), 400
        
        # Restrict to allowed base dirs
        allowed_bases = [CHECKPOINTS_FOLDER, CUDA_SAFE_CHECKPOINTS_FOLDER]
        if not any(rel_path.startswith(base) for base in allowed_bases):
            return jsonify({'success': False, 'error': 'Path not allowed'}), 400
        
        abs_path = os.path.abspath(rel_path)
        if not os.path.exists(abs_path):
            return jsonify({'success': False, 'error': 'Checkpoint not found'}), 404
        
        # Try to load the model with proper error handling
        try:
            load_model(abs_path, str(device) if device is not None else 'cuda')
            return jsonify({'success': True, 'current': current_checkpoint_path})
        except Exception as model_error:
            error_msg = f"Failed to load model: {str(model_error)}"
            if "Missing key" in str(model_error):
                error_msg += "\nThis usually means the checkpoint was saved with a different model architecture."
            return jsonify({'success': False, 'error': error_msg}), 500
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/model_info')
def model_info():
    """Get model information"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Determine hidden dimensions based on current checkpoint
    if current_checkpoint_path:
        hidden_dims = determine_model_architecture(current_checkpoint_path)
    else:
        hidden_dims = [64, 128, 256, 512]  # Default
    
    info = {
        'device': str(device),
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_architecture': 'SteganographyAutoencoder',
        'hidden_dimensions': hidden_dims
    }
    
    return jsonify(info)


# HTML Templates
@app.route('/create_templates')
def create_templates():
    """Create HTML templates if they don't exist"""
    templates_dir = 'templates'
    os.makedirs(templates_dir, exist_ok=True)
    # If a custom template already exists, do not overwrite
    existing = os.path.join(templates_dir, 'index.html')
    if os.path.exists(existing):
        return "Templates exist. Not overwritten."
    
    # Create index.html template
    index_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Digital Watermarking - Steganography Web App</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #2c3e50, #3498db);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .main-content {
            padding: 30px;
        }
        
        .tabs {
            display: flex;
            margin-bottom: 20px;
            border-bottom: 2px solid #eee;
        }
        
        .tab {
            padding: 15px 30px;
            cursor: pointer;
            border: none;
            background: none;
            font-size: 1.1em;
            color: #666;
            transition: all 0.3s;
        }
        
        .tab.active {
            color: #3498db;
            border-bottom: 3px solid #3498db;
        }
        
        .tab-content {
            display: none;
        }
        
        .tab-content.active {
            display: block;
        }
        
        .upload-section {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 30px;
            margin-bottom: 20px;
            border: 2px dashed #ddd;
            text-align: center;
        }
        
        .file-input-wrapper {
            margin: 15px 0;
        }
        
        .file-input {
            display: none;
        }
        
        .file-label {
            display: inline-block;
            padding: 12px 25px;
            background: #3498db;
            color: white;
            border-radius: 5px;
            cursor: pointer;
            transition: background 0.3s;
        }
        
        .file-label:hover {
            background: #2980b9;
        }
        
        .controls {
            margin: 20px 0;
        }
        
        .control-group {
            margin: 15px 0;
        }
        
        .control-group label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
            color: #333;
        }
        
        .control-group select,
        .control-group input {
            width: 200px;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        
        .process-btn {
            background: linear-gradient(135deg, #27ae60, #2ecc71);
            color: white;
            border: none;
            padding: 15px 40px;
            border-radius: 8px;
            font-size: 1.1em;
            cursor: pointer;
            transition: all 0.3s;
            margin: 10px;
        }
        
        .process-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        
        .process-btn:disabled {
            background: #bdc3c7;
            cursor: not-allowed;
            transform: none;
        }
        
        .results {
            margin-top: 30px;
        }
        
        .image-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        
        .image-card {
            border: 1px solid #ddd;
            border-radius: 10px;
            padding: 15px;
            text-align: center;
            background: white;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .image-card h3 {
            margin-bottom: 15px;
            color: #333;
        }
        
        .image-card img {
            max-width: 100%;
            height: 200px;
            object-fit: contain;
            border-radius: 5px;
            border: 1px solid #eee;
        }
        
        .metrics {
            background: #ecf0f1;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
        }
        
        .metrics h3 {
            margin-bottom: 15px;
            color: #2c3e50;
        }
        
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        
        .metric-item {
            background: white;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }
        
        .metric-value {
            font-size: 1.5em;
            font-weight: bold;
            color: #3498db;
        }
        
        .metric-label {
            color: #666;
            margin-top: 5px;
        }
        
        .download-links {
            margin: 20px 0;
            text-align: center;
        }
        
        .download-btn {
            background: #e74c3c;
            color: white;
            text-decoration: none;
            padding: 10px 20px;
            border-radius: 5px;
            margin: 5px;
            display: inline-block;
            transition: all 0.3s;
        }
        
        .download-btn:hover {
            background: #c0392b;
            transform: translateY(-2px);
        }
        
        .loading {
            text-align: center;
            padding: 30px;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #3498db;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .error {
            background: #e74c3c;
            color: white;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }
        
        .success {
            background: #27ae60;
            color: white;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }
        
        .info-section {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
        }
        
        .info-section h3 {
            color: #2c3e50;
            margin-bottom: 15px;
        }
        
        .info-section p {
            line-height: 1.6;
            color: #555;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎭 Digital Watermarking</h1>
            <p>Advanced Image-in-Image Steganography using Deep Learning</p>
        </div>
        
        <div class="main-content">
            <div class="tabs">
                <button class="tab active" onclick="showTab('embed')">Embed Secret</button>
                <button class="tab" onclick="showTab('extract')">Extract Secret</button>
                <button class="tab" onclick="showTab('info')">Information</button>
            </div>
            
            <!-- Embed Tab -->
            <div id="embed-tab" class="tab-content active">
                <div class="upload-section">
                    <h2>Hide a Secret Image in a Cover Image</h2>
                    <p>Upload both a cover image and a secret image to create a stego image</p>
                    
                    <form id="embed-form" enctype="multipart/form-data">
                        <div class="file-input-wrapper">
                            <input type="file" id="cover-input" name="cover_image" class="file-input" accept="image/*" required>
                            <label for="cover-input" class="file-label">📁 Choose Cover Image</label>
                            <span id="cover-filename"></span>
                        </div>
                        
                        <div class="file-input-wrapper">
                            <input type="file" id="secret-input" name="secret_image" class="file-input" accept="image/*" required>
                            <label for="secret-input" class="file-label">🔒 Choose Secret Image</label>
                            <span id="secret-filename"></span>
                        </div>
                        
                        <div class="controls">
                            <div class="control-group">
                                <label for="image-size">Processing Size:</label>
                                <select id="image-size" name="image_size">
                                    <option value="64">64x64 (Fast)</option>
                                    <option value="128" selected>128x128 (Balanced)</option>
                                    <option value="256">256x256 (High Quality)</option>
                                </select>
                            </div>
                        </div>
                        
                        <button type="submit" class="process-btn">🎯 Embed Secret Image</button>
                    </form>
                </div>
                
                <div id="embed-results" class="results" style="display: none;">
                    <h2>Results</h2>
                    <div id="embed-images" class="image-grid"></div>
                    <div id="embed-metrics" class="metrics"></div>
                    <div id="embed-downloads" class="download-links"></div>
                </div>
            </div>
            
            <!-- Extract Tab -->
            <div id="extract-tab" class="tab-content">
                <div class="upload-section">
                    <h2>Extract Hidden Secret Image</h2>
                    <p>Upload a stego image to extract the hidden secret image</p>
                    
                    <form id="extract-form" enctype="multipart/form-data">
                        <div class="file-input-wrapper">
                            <input type="file" id="stego-input" name="stego_image" class="file-input" accept="image/*" required>
                            <label for="stego-input" class="file-label">📁 Choose Stego Image</label>
                            <span id="stego-filename"></span>
                        </div>
                        
                        <div class="controls">
                            <div class="control-group">
                                <label for="extract-image-size">Processing Size:</label>
                                <select id="extract-image-size" name="image_size">
                                    <option value="64">64x64 (Fast)</option>
                                    <option value="128" selected>128x128 (Balanced)</option>
                                    <option value="256">256x256 (High Quality)</option>
                                </select>
                            </div>
                        </div>
                        
                        <button type="submit" class="process-btn">🔍 Extract Secret Image</button>
                    </form>
                </div>
                
                <div id="extract-results" class="results" style="display: none;">
                    <h2>Extraction Results</h2>
                    <div id="extract-images" class="image-grid"></div>
                    <div id="extract-downloads" class="download-links"></div>
                </div>
            </div>
            
            <!-- Info Tab -->
            <div id="info-tab" class="tab-content">
                <div class="info-section">
                    <h3>About Digital Watermarking</h3>
                    <p>
                        This application demonstrates advanced image-in-image steganography using deep learning.
                        It uses a Convolutional Autoencoder architecture to hide a secret image within a cover image,
                        creating a stego image that appears visually identical to the original cover image.
                    </p>
                </div>
                
                <div class="info-section">
                    <h3>How It Works</h3>
                    <p>
                        1. <strong>Encoding:</strong> The encoder network takes both cover and secret images as input and produces a stego image that contains the hidden information.<br>
                        2. <strong>Decoding:</strong> The decoder network can extract the original secret image from the stego image.<br>
                        3. <strong>Training:</strong> The model is trained end-to-end with custom loss functions to minimize distortion and maximize recovery quality.
                    </p>
                </div>
                
                <div class="info-section">
                    <h3>Quality Metrics</h3>
                    <p>
                        • <strong>PSNR (Peak Signal-to-Noise Ratio):</strong> Higher values indicate better quality (>30 dB is good)<br>
                        • <strong>SSIM (Structural Similarity Index):</strong> Values closer to 1.0 indicate higher perceptual similarity
                    </p>
                </div>
                
                <div id="model-info" class="info-section">
                    <h3>Model Information</h3>
                    <p>Loading model information...</p>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Tab switching
        function showTab(tabName) {
            // Hide all tabs
            document.querySelectorAll('.tab-content').forEach(tab => {
                tab.classList.remove('active');
            });
            document.querySelectorAll('.tab').forEach(tab => {
                tab.classList.remove('active');
            });
            
            // Show selected tab
            document.getElementById(tabName + '-tab').classList.add('active');
            event.target.classList.add('active');
        }
        
        // File input handlers
        document.getElementById('cover-input').addEventListener('change', function(e) {
            document.getElementById('cover-filename').textContent = e.target.files[0] ? e.target.files[0].name : '';
        });
        
        document.getElementById('secret-input').addEventListener('change', function(e) {
            document.getElementById('secret-filename').textContent = e.target.files[0] ? e.target.files[0].name : '';
        });
        
        document.getElementById('stego-input').addEventListener('change', function(e) {
            document.getElementById('stego-filename').textContent = e.target.files[0] ? e.target.files[0].name : '';
        });
        
        // Embed form handler
        document.getElementById('embed-form').addEventListener('submit', function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            const submitBtn = this.querySelector('button[type="submit"]');
            const resultsDiv = document.getElementById('embed-results');
            
            // Show loading
            submitBtn.disabled = true;
            submitBtn.innerHTML = '⏳ Processing...';
            resultsDiv.style.display = 'block';
            resultsDiv.innerHTML = '<div class="loading"><div class="spinner"></div><p>Embedding secret image...</p></div>';
            
            fetch('/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    displayEmbedResults(data);
                } else {
                    resultsDiv.innerHTML = '<div class="error">Error: ' + data.error + '</div>';
                }
            })
            .catch(error => {
                resultsDiv.innerHTML = '<div class="error">Network error: ' + error.message + '</div>';
            })
            .finally(() => {
                submitBtn.disabled = false;
                submitBtn.innerHTML = '🎯 Embed Secret Image';
            });
        });
        
        // Extract form handler
        document.getElementById('extract-form').addEventListener('submit', function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            const submitBtn = this.querySelector('button[type="submit"]');
            const resultsDiv = document.getElementById('extract-results');
            
            // Show loading
            submitBtn.disabled = true;
            submitBtn.innerHTML = '⏳ Extracting...';
            resultsDiv.style.display = 'block';
            resultsDiv.innerHTML = '<div class="loading"><div class="spinner"></div><p>Extracting secret image...</p></div>';
            
            fetch('/extract', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    displayExtractResults(data);
                } else {
                    resultsDiv.innerHTML = '<div class="error">Error: ' + data.error + '</div>';
                }
            })
            .catch(error => {
                resultsDiv.innerHTML = '<div class="error">Network error: ' + error.message + '</div>';
            })
            .finally(() => {
                submitBtn.disabled = false;
                submitBtn.innerHTML = '🔍 Extract Secret Image';
            });
        });
        
        function displayEmbedResults(data) {
            const resultsDiv = document.getElementById('embed-results');
            
            // Display images
            const imagesHtml = `
                <div class="image-card">
                    <h3>Cover Image</h3>
                    <img src="${data.images.cover}" alt="Cover Image">
                </div>
                <div class="image-card">
                    <h3>Secret Image</h3>
                    <img src="${data.images.secret}" alt="Secret Image">
                </div>
                <div class="image-card">
                    <h3>Stego Image</h3>
                    <img src="${data.images.stego}" alt="Stego Image">
                </div>
                <div class="image-card">
                    <h3>Recovered Secret</h3>
                    <img src="${data.images.recovered}" alt="Recovered Secret">
                </div>
            `;
            
            // Display metrics
            const metricsHtml = `
                <h3>Quality Metrics</h3>
                <div class="metric-grid">
                    <div class="metric-item">
                        <div class="metric-value">${data.metrics.cover_psnr.toFixed(2)} dB</div>
                        <div class="metric-label">Cover PSNR</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-value">${data.metrics.cover_ssim.toFixed(4)}</div>
                        <div class="metric-label">Cover SSIM</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-value">${data.metrics.secret_psnr.toFixed(2)} dB</div>
                        <div class="metric-label">Secret PSNR</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-value">${data.metrics.secret_ssim.toFixed(4)}</div>
                        <div class="metric-label">Secret SSIM</div>
                    </div>
                </div>
            `;
            
            // Download links
            const downloadsHtml = `
                <a href="${data.download_links.stego}" class="download-btn" download>📥 Download Stego Image</a>
                <a href="${data.download_links.recovered}" class="download-btn" download>📥 Download Recovered Secret</a>
            `;
            
            document.getElementById('embed-images').innerHTML = imagesHtml;
            document.getElementById('embed-metrics').innerHTML = metricsHtml;
            document.getElementById('embed-downloads').innerHTML = downloadsHtml;
        }
        
        function displayExtractResults(data) {
            const resultsDiv = document.getElementById('extract-results');
            
            // Display images
            const imagesHtml = `
                <div class="image-card">
                    <h3>Stego Image</h3>
                    <img src="${data.images.stego}" alt="Stego Image">
                </div>
                <div class="image-card">
                    <h3>Extracted Secret</h3>
                    <img src="${data.images.extracted}" alt="Extracted Secret">
                </div>
            `;
            
            // Download links
            const downloadsHtml = `
                <a href="${data.download_links.extracted}" class="download-btn" download>📥 Download Extracted Secret</a>
            `;
            
            document.getElementById('extract-images').innerHTML = imagesHtml;
            document.getElementById('extract-downloads').innerHTML = downloadsHtml;
        }
        
        // Load model information
        fetch('/model_info')
            .then(response => response.json())
            .then(data => {
                if (!data.error) {
                    const modelInfoHtml = `
                        <h3>Model Information</h3>
                        <p>
                            <strong>Architecture:</strong> ${data.model_architecture}<br>
                            <strong>Device:</strong> ${data.device}<br>
                            <strong>Total Parameters:</strong> ${data.total_parameters.toLocaleString()}<br>
                            <strong>Trainable Parameters:</strong> ${data.trainable_parameters.toLocaleString()}<br>
                            <strong>Hidden Dimensions:</strong> ${data.hidden_dimensions.join(' → ')}
                        </p>
                    `;
                    document.getElementById('model-info').innerHTML = modelInfoHtml;
                }
            })
            .catch(error => {
                document.getElementById('model-info').innerHTML = '<h3>Model Information</h3><p>Could not load model information.</p>';
            });
    </script>
</body>
</html>"""
    
    with open(os.path.join(templates_dir, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(index_html)
    
    return "Templates created successfully!"


def main():
    """Main function to run the web application"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Digital Watermarking Web Application')
    parser.add_argument('--checkpoint', type=str, default='cuda_safe_checkpoints/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to run the server on')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    
    args = parser.parse_args()
    
    # Create templates
    create_templates()
    
    # Load model
    load_model(args.checkpoint, args.device)
    
    print(f"🎭 Digital Watermarking Web App")
    print(f"Server starting on http://{args.host}:{args.port}")
    print(f"Model checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    
    # Run the app
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
