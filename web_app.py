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
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Global model variable
model = None
device = None


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_model(checkpoint_path, device_type='cuda'):
    """Load the steganography model"""
    global model, device
    
    device = torch.device(device_type if torch.cuda.is_available() else 'cpu')
    
    model = SteganographyAutoencoder(
        in_channels=3,
        hidden_dims=[64, 128, 256, 512]
    )
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint {checkpoint_path} not found. Using untrained model.")
    
    model.to(device)
    model.eval()
    print(f"Model running on {device}")


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


@app.route('/model_info')
def model_info():
    """Get model information"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    info = {
        'device': str(device),
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_architecture': 'SteganographyAutoencoder',
        'hidden_dimensions': [64, 128, 256, 512]
    }
    
    return jsonify(info)


# HTML Templates
@app.route('/create_templates')
def create_templates():
    """Create HTML templates if they don't exist"""
    templates_dir = 'templates'
    os.makedirs(templates_dir, exist_ok=True)
    
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
    parser.add_argument('--checkpoint', type=str, default='checkpoints/checkpoint_epoch_5.pth',
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
