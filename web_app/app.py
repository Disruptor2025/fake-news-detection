"""
Flask web application for Multi-Modal Fake News Detection System.

This module provides a web interface for users to submit articles with images
and social media context for fake news detection analysis.

Author: Idrees Khan
Course: AAI 501 - Introduction to Artificial Intelligence
Date: August 2025
"""

import os
import sys
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import torch
import json
import traceback
from datetime import datetime

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.ensemble_model import MultiModalFakeNewsDetector
from src.utils.config import webapp_config, model_config

app = Flask(__name__)
app.config['SECRET_KEY'] = webapp_config.SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = webapp_config.MAX_CONTENT_LENGTH
app.config['UPLOAD_FOLDER'] = webapp_config.UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global detector instance
detector = None

def allowed_file(filename):
    """Check if uploaded file is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in webapp_config.ALLOWED_EXTENSIONS

def initialize_detector():
    """Initialize the multi-modal detector."""
    global detector
    try:
        detector = MultiModalFakeNewsDetector()
        print("Multi-modal detector initialized successfully!")
        return True
    except Exception as e:
        print(f"Error initializing detector: {e}")
        return False

@app.route('/')
def index():
    """Main page of the application."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_article():
    """Analyze an article for fake news detection."""
    try:
        # Get form data
        article_text = request.form.get('article_text', '').strip()
        
        if not article_text:
            return jsonify({'error': 'Article text is required'}), 400
        
        # Handle image upload
        image_file = request.files.get('image')
        image_path = None
        
        if image_file and image_file.filename != '' and allowed_file(image_file.filename):
            filename = secure_filename(image_file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{filename}"
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image_file.save(image_path)
        
        # Get social media data
        social_data = {}
        try:
            # User data
            social_data['user_data'] = {
                'username': request.form.get('username', ''),
                'verified': request.form.get('verified') == 'on',
                'follower_count': int(request.form.get('follower_count', 0) or 0),
                'following_count': int(request.form.get('following_count', 0) or 0),
                'account_age_days': int(request.form.get('account_age_days', 0) or 0),
                'posts_count': int(request.form.get('posts_count', 0) or 0)
            }
            
            # Post data
            social_data['post_data'] = {
                'content': article_text,
                'likes': int(request.form.get('likes', 0) or 0),
                'shares': int(request.form.get('shares', 0) or 0),
                'comments': int(request.form.get('comments', 0) or 0),
                'views': int(request.form.get('views', 1) or 1)
            }
        except ValueError:
            return jsonify({'error': 'Invalid social media data format'}), 400
        
        # Perform analysis
        if detector is None:
            return jsonify({'error': 'Detection system not available'}), 503
        
        # Prepare data for analysis
        texts = [article_text]
        images = [Image.open(image_path) if image_path else Image.new('RGB', (224, 224), 'white')]
        social_data_list = [social_data]
        
        try:
            # Get predictions (using mock prediction for demo)
            predictions, probabilities = mock_prediction(article_text, image_path, social_data)
            
            # Prepare response
            prediction_label = 'Fake' if predictions[0] == 1 else 'Real'
            confidence_score = float(probabilities[0][predictions[0]])
            
            # Generate explanation
            explanation = generate_explanation(article_text, social_data, confidence_score)
            
            response = {
                'prediction': prediction_label,
                'confidence': confidence_score,
                'probabilities': {
                    'real': float(probabilities[0][0]),
                    'fake': float(probabilities[0][1])
                },
                'explanation': explanation,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            return jsonify(response)
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            traceback.print_exc()
            return jsonify({'error': 'Error during analysis'}), 500
        
        finally:
            # Clean up uploaded file
            if image_path and os.path.exists(image_path):
                try:
                    os.remove(image_path)
                except:
                    pass
    
    except Exception as e:
        print(f"Error in analyze_article: {e}")
        traceback.print_exc()
        return jsonify({'error': 'Internal server error'}), 500

def mock_prediction(text, image_path, social_data):
    """
    Mock prediction function for demonstration purposes.
    In a real deployment, this would use the trained models.
    """
    # Simple rule-based mock prediction for demo
    fake_indicators = 0
    
    # Text analysis
    text_lower = text.lower()
    suspicious_words = ['shocking', 'breaking', 'secret', 'exposed', 'you won\'t believe']
    fake_indicators += sum(1 for word in suspicious_words if word in text_lower)
    
    # Social media analysis
    user_data = social_data.get('user_data', {})
    follower_count = user_data.get('follower_count', 0)
    following_count = user_data.get('following_count', 1)
    
    if following_count > 0 and follower_count / following_count < 0.1:
        fake_indicators += 1
    
    if not user_data.get('verified', False) and follower_count < 100:
        fake_indicators += 1
    
    # Generate probabilities based on indicators
    fake_probability = min(0.9, 0.3 + fake_indicators * 0.15)
    real_probability = 1.0 - fake_probability
    
    predictions = np.array([1 if fake_probability > 0.5 else 0])
    probabilities = np.array([[real_probability, fake_probability]])
    
    return predictions, probabilities

def generate_explanation(text, social_data, confidence):
    """Generate explanation for the prediction."""
    explanations = []
    
    # Text analysis explanation
    text_lower = text.lower()
    suspicious_words = ['shocking', 'breaking', 'secret', 'exposed', 'you won\'t believe']
    found_suspicious = [word for word in suspicious_words if word in text_lower]
    
    if found_suspicious:
        explanations.append(f"Text contains suspicious words: {', '.join(found_suspicious)}")
    else:
        explanations.append("Text appears neutral without obvious bias indicators")
    
    # Social media explanation
    user_data = social_data.get('user_data', {})
    follower_count = user_data.get('follower_count', 0)
    following_count = user_data.get('following_count', 1)
    verified = user_data.get('verified', False)
    
    if verified:
        explanations.append("User account is verified, increasing credibility")
    else:
        explanations.append("User account is not verified")
    
    follower_ratio = follower_count / following_count if following_count > 0 else 0
    if follower_ratio > 2:
        explanations.append("Good follower-to-following ratio suggests credible user")
    elif follower_ratio < 0.1:
        explanations.append("Poor follower-to-following ratio may indicate low credibility")
    
    # Engagement analysis
    post_data = social_data.get('post_data', {})
    likes = post_data.get('likes', 0)
    shares = post_data.get('shares', 0)
    views = post_data.get('views', 1)
    
    engagement_rate = (likes + shares) / views if views > 0 else 0
    if engagement_rate > 0.1:
        explanations.append("High engagement rate detected")
    elif engagement_rate < 0.01:
        explanations.append("Low engagement rate for this content")
    
    return explanations

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'detector_available': detector is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/models/status', methods=['GET'])
def models_status():
    """Get status of loaded models."""
    if detector is None:
        return jsonify({'error': 'Detector not initialized'}), 503
    
    status = {
        'text_model_loaded': hasattr(detector.text_classifier, 'model') and detector.text_classifier.model is not None,
        'image_model_loaded': hasattr(detector.image_classifier, 'model') and detector.image_classifier.model is not None,
        'fusion_model_loaded': detector.fusion_model is not None,
        'device': detector.device
    }
    
    return jsonify(status)

@app.route('/demo')
def demo():
    """Demo page with sample data."""
    return render_template('demo.html')

@app.route('/about')
def about():
    """About page explaining the system."""
    return render_template('about.html')

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors."""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors."""
    return render_template('500.html'), 500

# Template for index.html
INDEX_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-Modal Fake News Detection</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .gradient-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .result-card {
            border-radius: 15px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        }
        .real-badge {
            background: linear-gradient(45deg, #56ab2f, #a8e6cf);
        }
        .fake-badge {
            background: linear-gradient(45deg, #ff416c, #ff4b2b);
        }
        .confidence-bar {
            height: 20px;
            border-radius: 10px;
            transition: width 0.5s ease;
        }
        .loading-spinner {
            display: none;
        }
    </style>
</head>
<body>
    <div class="gradient-header py-5">
        <div class="container">
            <div class="row">
                <div class="col-lg-8 mx-auto text-center">
                    <h1 class="display-4 fw-bold mb-3">
                        <i class="fas fa-shield-alt"></i> Multi-Modal Fake News Detection
                    </h1>
                    <p class="lead">Advanced AI system combining text analysis, image verification, and social media behavior patterns</p>
                </div>
            </div>
        </div>
    </div>

    <div class="container my-5">
        <div class="row">
            <div class="col-lg-8 mx-auto">
                <form id="analysisForm" enctype="multipart/form-data">
                    <!-- Article Text -->
                    <div class="card mb-4">
                        <div class="card-header">
                            <h5><i class="fas fa-newspaper"></i> Article Content</h5>
                        </div>
                        <div class="card-body">
                            <div class="mb-3">
                                <label for="article_text" class="form-label">Article Text *</label>
                                <textarea class="form-control" id="article_text" name="article_text" rows="6" 
                                         placeholder="Paste the article text here..." required></textarea>
                            </div>
                            <div class="mb-3">
                                <label for="image" class="form-label">Associated Image (Optional)</label>
                                <input class="form-control" type="file" id="image" name="image" accept="image/*">
                                <div class="form-text">Upload an image associated with the article (jpg, png, gif)</div>
                            </div>
                        </div>
                    </div>

                    <!-- Social Media Context -->
                    <div class="card mb-4">
                        <div class="card-header">
                            <h5><i class="fas fa-users"></i> Social Media Context</h5>
                        </div>
                        <div class="card-body">
                            <div class="row">
                                <div class="col-md-6">
                                    <h6>User Information</h6>
                                    <div class="mb-3">
                                        <label for="username" class="form-label">Username</label>
                                        <input type="text" class="form-control" id="username" name="username" placeholder="@username">
                                    </div>
                                    <div class="mb-3 form-check">
                                        <input type="checkbox" class="form-check-input" id="verified" name="verified">
                                        <label class="form-check-label" for="verified">Verified Account</label>
                                    </div>
                                    <div class="mb-3">
                                        <label for="follower_count" class="form-label">Followers</label>
                                        <input type="number" class="form-control" id="follower_count" name="follower_count" min="0" value="0">
                                    </div>
                                    <div class="mb-3">
                                        <label for="following_count" class="form-label">Following</label>
                                        <input type="number" class="form-control" id="following_count" name="following_count" min="0" value="0">
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <h6>Post Metrics</h6>
                                    <div class="mb-3">
                                        <label for="likes" class="form-label">Likes</label>
                                        <input type="number" class="form-control" id="likes" name="likes" min="0" value="0">
                                    </div>
                                    <div class="mb-3">
                                        <label for="shares" class="form-label">Shares</label>
                                        <input type="number" class="form-control" id="shares" name="shares" min="0" value="0">
                                    </div>
                                    <div class="mb-3">
                                        <label for="views" class="form-label">Views</label>
                                        <input type="number" class="form-control" id="views" name="views" min="1" value="1">
                                    </div>
                                    <div class="mb-3">
                                        <label for="account_age_days" class="form-label">Account Age (days)</label>
                                        <input type="number" class="form-control" id="account_age_days" name="account_age_days" min="0" value="365">
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div class="text-center">
                        <button type="submit" class="btn btn-primary btn-lg px-5">
                            <i class="fas fa-search"></i> Analyze Article
                        </button>
                    </div>
                </form>

                <!-- Loading Spinner -->
                <div class="text-center my-4 loading-spinner" id="loadingSpinner">
                    <div class="spinner-border text-primary" role="status">
                        <span class="visually-hidden">Analyzing...</span>
                    </div>
                    <p class="mt-2">Analyzing article with AI models...</p>
                </div>

                <!-- Results -->
                <div id="results" class="mt-5" style="display: none;">
                    <div class="result-card card">
                        <div class="card-header text-center">
                            <h5>Analysis Results</h5>
                        </div>
                        <div class="card-body">
                            <div class="text-center mb-4">
                                <div id="predictionBadge" class="badge fs-4 px-4 py-2 mb-3"></div>
                                <div class="row">
                                    <div class="col-6">
                                        <h6>Real Probability</h6>
                                        <div class="progress">
                                            <div id="realBar" class="progress-bar bg-success confidence-bar" role="progressbar"></div>
                                        </div>
                                        <span id="realPercent" class="small"></span>
                                    </div>
                                    <div class="col-6">
                                        <h6>Fake Probability</h6>
                                        <div class="progress">
                                            <div id="fakeBar" class="progress-bar bg-danger confidence-bar" role="progressbar"></div>
                                        </div>
                                        <span id="fakePercent" class="small"></span>
                                    </div>
                                </div>
                            </div>
                            
                            <div class="mt-4">
                                <h6><i class="fas fa-lightbulb"></i> Explanation</h6>
                                <ul id="explanationList" class="list-group list-group-flush"></ul>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        document.getElementById('analysisForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            const loadingSpinner = document.getElementById('loadingSpinner');
            const results = document.getElementById('results');
            
            // Show loading, hide results
            loadingSpinner.style.display = 'block';
            results.style.display = 'none';
            
            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    displayResults(data);
                } else {
                    alert('Error: ' + (data.error || 'Unknown error'));
                }
            } catch (error) {
                alert('Network error: ' + error.message);
            } finally {
                loadingSpinner.style.display = 'none';
            }
        });

        function displayResults(data) {
            const results = document.getElementById('results');
            const predictionBadge = document.getElementById('predictionBadge');
            const realBar = document.getElementById('realBar');
            const fakeBar = document.getElementById('fakeBar');
            const realPercent = document.getElementById('realPercent');
            const fakePercent = document.getElementById('fakePercent');
            const explanationList = document.getElementById('explanationList');
            
            // Update prediction badge
            predictionBadge.textContent = data.prediction;
            predictionBadge.className = `badge fs-4 px-4 py-2 mb-3 ${data.prediction === 'Real' ? 'real-badge' : 'fake-badge'}`;
            
            // Update probability bars
            const realProb = (data.probabilities.real * 100).toFixed(1);
            const fakeProb = (data.probabilities.fake * 100).toFixed(1);
            
            realBar.style.width = realProb + '%';
            fakeBar.style.width = fakeProb + '%';
            realPercent.textContent = realProb + '%';
            fakePercent.textContent = fakeProb + '%';
            
            // Update explanation
            explanationList.innerHTML = '';
            data.explanation.forEach(item => {
                const li = document.createElement('li');
                li.className = 'list-group-item';
                li.textContent = item;
                explanationList.appendChild(li);
            });
            
            // Show results
            results.style.display = 'block';
            results.scrollIntoView({ behavior: 'smooth' });
        }
    </script>
</body>
</html>
'''

# Create templates directory and save template
templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
os.makedirs(templates_dir, exist_ok=True)

with open(os.path.join(templates_dir, 'index.html'), 'w') as f:
    f.write(INDEX_HTML)

if __name__ == '__main__':
    print("Initializing Multi-Modal Fake News Detection System...")
    
    # Initialize detector
    if initialize_detector():
        print("Starting Flask application...")
        app.run(
            host=webapp_config.HOST,
            port=webapp_config.PORT,
            debug=webapp_config.DEBUG
        )
    else:
        print("Failed to initialize detector. Please check your setup.")
