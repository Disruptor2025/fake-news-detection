"""
Image processing module for fake news detection.

This module handles image preprocessing, manipulation detection, and feature
extraction for the multi-modal fake news detection system.

Author: Idrees Khan
Course: AAI 501 - Introduction to Artificial Intelligence
Date: August 2025
"""

import os
import cv2
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from PIL import Image, ExifTags
import imagehash
import requests
from io import BytesIO
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
import hashlib

from ..utils.config import model_config, feature_config, data_config


class ImagePreprocessor:
    """
    Comprehensive image preprocessing for fake news detection.
    
    This class provides methods for loading, processing, and extracting
    features from images associated with news articles.
    """
    
    def __init__(self):
        """Initialize the image preprocessor with required models and tools."""
        # Load pre-trained ResNet model for feature extraction
        self.resnet_model = resnet50(pretrained=True)
        self.resnet_model.eval()
        
        # Remove the final classification layer to get features
        self.resnet_features = torch.nn.Sequential(*list(self.resnet_model.children())[:-1])
        
        # Image preprocessing pipeline for ResNet
        self.transform = transforms.Compose([
            transforms.Resize(model_config.IMAGE_SIZE),
            transforms.CenterCrop(model_config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Error analysis patterns for manipulation detection
        self.manipulation_patterns = {
            'jpeg_artifacts': self._detect_jpeg_artifacts,
            'ela_analysis': self._error_level_analysis,
            'noise_analysis': self._noise_inconsistency,
            'lighting_analysis': self._lighting_inconsistency,
            'compression_analysis': self._compression_artifacts
        }
    
    def load_image(self, image_source: Union[str, np.ndarray, Image.Image]) -> Optional[Image.Image]:
        """
        Load image from various sources (file path, URL, array, PIL Image).
        
        Args:
            image_source: Path, URL, numpy array, or PIL Image
            
        Returns:
            PIL Image object or None if loading fails
        """
        try:
            if isinstance(image_source, str):
                if image_source.startswith(('http://', 'https://')):
                    # Load from URL
                    response = requests.get(image_source, timeout=10)
                    response.raise_for_status()
                    image = Image.open(BytesIO(response.content))
                else:
                    # Load from file path
                    image = Image.open(image_source)
            elif isinstance(image_source, np.ndarray):
                # Convert numpy array to PIL Image
                image = Image.fromarray(image_source)
            elif isinstance(image_source, Image.Image):
                # Already a PIL Image
                image = image_source
            else:
                raise ValueError(f"Unsupported image source type: {type(image_source)}")
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return image
            
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
    
    def extract_metadata(self, image: Image.Image) -> Dict[str, any]:
        """
        Extract EXIF metadata from image.
        
        Args:
            image: PIL Image object
            
        Returns:
            Dictionary containing metadata features
        """
        metadata_features = {
            'has_exif': False,
            'camera_make': '',
            'camera_model': '',
            'creation_date': '',
            'gps_info': False,
            'software_info': '',
            'metadata_inconsistency_score': 0.0
        }
        
        try:
            exif_data = image._getexif()
            if exif_data is not None:
                metadata_features['has_exif'] = True
                
                for tag_id, value in exif_data.items():
                    tag = ExifTags.TAGS.get(tag_id, tag_id)
                    
                    if tag == 'Make':
                        metadata_features['camera_make'] = str(value)
                    elif tag == 'Model':
                        metadata_features['camera_model'] = str(value)
                    elif tag == 'DateTime':
                        metadata_features['creation_date'] = str(value)
                    elif tag == 'Software':
                        metadata_features['software_info'] = str(value)
                    elif tag == 'GPSInfo':
                        metadata_features['gps_info'] = True
                
                # Calculate metadata inconsistency score
                metadata_features['metadata_inconsistency_score'] = self._calculate_metadata_inconsistency(exif_data)
        
        except Exception as e:
            print(f"Error extracting metadata: {e}")
        
        return metadata_features
    
    def _calculate_metadata_inconsistency(self, exif_data: dict) -> float:
        """Calculate inconsistency score in metadata."""
        inconsistency_score = 0.0
        
        # Check for suspicious software signatures
        software_info = exif_data.get(ExifTags.TAGS.get('Software', ''), '')
        suspicious_software = ['photoshop', 'gimp', 'editing', 'manipulated']
        
        if any(sus in software_info.lower() for sus in suspicious_software):
            inconsistency_score += 0.3
        
        # Check for missing typical camera metadata
        expected_tags = ['Make', 'Model', 'DateTime']
        missing_tags = sum(1 for tag in expected_tags if tag not in str(exif_data))
        inconsistency_score += missing_tags * 0.2
        
        return min(inconsistency_score, 1.0)
    
    def detect_manipulation(self, image: Image.Image) -> Dict[str, float]:
        """
        Detect potential image manipulation using multiple techniques.
        
        Args:
            image: PIL Image object
            
        Returns:
            Dictionary containing manipulation detection scores
        """
        manipulation_scores = {}
        
        # Convert to OpenCV format for analysis
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Apply different manipulation detection techniques
        for technique_name, technique_func in self.manipulation_patterns.items():
            try:
                score = technique_func(cv_image)
                manipulation_scores[technique_name] = score
            except Exception as e:
                print(f"Error in {technique_name}: {e}")
                manipulation_scores[technique_name] = 0.0
        
        # Calculate overall manipulation score
        manipulation_scores['overall_manipulation_score'] = np.mean(list(manipulation_scores.values()))
        
        return manipulation_scores
    
    def _detect_jpeg_artifacts(self, image: np.ndarray) -> float:
        """Detect JPEG compression artifacts."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply DCT to detect JPEG artifacts
        # Focus on 8x8 blocks which are characteristic of JPEG compression
        h, w = gray.shape
        artifact_score = 0.0
        block_count = 0
        
        for i in range(0, h-8, 8):
            for j in range(0, w-8, 8):
                block = gray[i:i+8, j:j+8].astype(np.float32)
                
                # Apply DCT
                dct_block = cv2.dct(block)
                
                # Check for characteristic JPEG patterns
                # High frequency components should show specific patterns in manipulated images
                high_freq = dct_block[4:, 4:]
                artifact_score += np.std(high_freq)
                block_count += 1
        
        # Normalize score
        return min(artifact_score / (block_count + 1), 1.0) if block_count > 0 else 0.0
    
    def _error_level_analysis(self, image: np.ndarray) -> float:
        """Perform Error Level Analysis (ELA) for manipulation detection."""
        # Convert to PIL for JPEG compression
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Save with specific JPEG quality
        buffer = BytesIO()
        pil_image.save(buffer, format='JPEG', quality=90)
        buffer.seek(0)
        
        # Reload the compressed image
        compressed_image = Image.open(buffer)
        compressed_array = np.array(compressed_image)
        
        # Calculate difference
        original_array = np.array(pil_image)
        difference = cv2.absdiff(original_array, compressed_array)
        
        # Calculate ELA score
        ela_score = np.mean(difference) / 255.0
        
        return min(ela_score, 1.0)
    
    def _noise_inconsistency(self, image: np.ndarray) -> float:
        """Analyze noise patterns for inconsistencies."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply noise analysis using different filters
        # Gaussian noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = cv2.subtract(gray, blurred)
        
        # Calculate noise variance in different regions
        h, w = gray.shape
        region_variances = []
        
        # Divide image into regions and analyze noise variance
        for i in range(0, h, h//4):
            for j in range(0, w, w//4):
                region = noise[i:min(i+h//4, h), j:min(j+w//4, w)]
                if region.size > 0:
                    region_variances.append(np.var(region))
        
        # Inconsistent noise patterns suggest manipulation
        if len(region_variances) > 1:
            noise_inconsistency = np.std(region_variances) / (np.mean(region_variances) + 1e-6)
            return min(noise_inconsistency / 10.0, 1.0)
        
        return 0.0
    
    def _lighting_inconsistency(self, image: np.ndarray) -> float:
        """Analyze lighting inconsistencies."""
        # Convert to LAB color space for better lighting analysis
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        
        # Calculate gradients to find lighting directions
        grad_x = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate gradient magnitude and direction
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x)
        
        # Analyze consistency of lighting direction
        # Inconsistent lighting suggests composite/manipulated images
        direction_variance = np.var(direction[magnitude > np.percentile(magnitude, 75)])
        
        # Normalize to 0-1 range
        lighting_score = min(direction_variance / (np.pi**2), 1.0)
        
        return lighting_score
    
    def _compression_artifacts(self, image: np.ndarray) -> float:
        """Detect compression artifacts and inconsistencies."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply FFT to analyze frequency domain
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude_spectrum = np.log(np.abs(fft_shift) + 1)
        
        # Look for patterns that suggest different compression levels
        # in different parts of the image
        h, w = magnitude_spectrum.shape
        
        # Analyze quadrants
        quadrants = [
            magnitude_spectrum[:h//2, :w//2],
            magnitude_spectrum[:h//2, w//2:],
            magnitude_spectrum[h//2:, :w//2],
            magnitude_spectrum[h//2:, w//2:]
        ]
        
        # Calculate compression consistency
        quad_means = [np.mean(quad) for quad in quadrants]
        compression_inconsistency = np.std(quad_means) / (np.mean(quad_means) + 1e-6)
        
        return min(compression_inconsistency / 5.0, 1.0)
    
    def extract_visual_features(self, image: Image.Image) -> Dict[str, float]:
        """
        Extract visual features using deep learning and traditional computer vision.
        
        Args:
            image: PIL Image object
            
        Returns:
            Dictionary containing visual features
        """
        visual_features = {}
        
        try:
            # Convert to tensor for deep learning model
            image_tensor = self.transform(image).unsqueeze(0)
            
            # Extract ResNet features
            with torch.no_grad():
                features = self.resnet_features(image_tensor)
                # Flatten the features
                resnet_features = features.view(features.size(0), -1).numpy()[0]
            
            # Store ResNet features (we'll use these for the main model)
            visual_features['resnet_features'] = resnet_features
            
            # Extract traditional computer vision features
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Color distribution features
            visual_features.update(self._extract_color_features(cv_image))
            
            # Texture features
            visual_features.update(self._extract_texture_features(cv_image))
            
            # Edge and shape features
            visual_features.update(self._extract_edge_features(cv_image))
            
            # Image quality features
            visual_features.update(self._extract_quality_features(cv_image))
            
        except Exception as e:
            print(f"Error extracting visual features: {e}")
            # Return empty features in case of error
            visual_features = {
                'resnet_features': np.zeros(model_config.IMAGE_EMBEDDING_DIM),
                'brightness_score': 0.0,
                'contrast_score': 0.0,
                'saturation_score': 0.0,
                'texture_score': 0.0,
                'edge_density': 0.0,
                'sharpness_score': 0.0
            }
        
        return visual_features
    
    def _extract_color_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract color-based features."""
        features = {}
        
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Brightness (from LAB L channel)
        l_channel = lab[:, :, 0]
        features['brightness_score'] = np.mean(l_channel) / 255.0
        
        # Contrast (standard deviation of L channel)
        features['contrast_score'] = np.std(l_channel) / 255.0
        
        # Saturation (from HSV S channel)
        s_channel = hsv[:, :, 1]
        features['saturation_score'] = np.mean(s_channel) / 255.0
        
        # Color diversity (histogram entropy)
        hist_b = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
        hist_r = cv2.calcHist([image], [2], None, [256], [0, 256])
        
        # Calculate entropy for each channel
        def calculate_entropy(hist):
            hist = hist.flatten()
            hist = hist[hist > 0]
            hist = hist / np.sum(hist)
            return -np.sum(hist * np.log2(hist + 1e-7))
        
        features['color_entropy_b'] = calculate_entropy(hist_b)
        features['color_entropy_g'] = calculate_entropy(hist_g)
        features['color_entropy_r'] = calculate_entropy(hist_r)
        
        return features
    
    def _extract_texture_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract texture-based features."""
        features = {}
        
        # Convert to grayscale for texture analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Local Binary Pattern (simplified version)
        def local_binary_pattern(img, radius=1):
            rows, cols = img.shape
            lbp = np.zeros_like(img)
            
            for i in range(radius, rows - radius):
                for j in range(radius, cols - radius):
                    center = img[i, j]
                    code = 0
                    code |= (img[i-radius, j-radius] >= center) << 7
                    code |= (img[i-radius, j] >= center) << 6
                    code |= (img[i-radius, j+radius] >= center) << 5
                    code |= (img[i, j+radius] >= center) << 4
                    code |= (img[i+radius, j+radius] >= center) << 3
                    code |= (img[i+radius, j] >= center) << 2
                    code |= (img[i+radius, j-radius] >= center) << 1
                    code |= (img[i, j-radius] >= center) << 0
                    lbp[i, j] = code
            
            return lbp
        
        lbp = local_binary_pattern(gray)
        features['texture_uniformity'] = np.std(lbp) / 255.0
        
        # Gabor filter response (simplified)
        kernel = cv2.getGaborKernel((21, 21), 5, 0, 10, 0.5, 0, ktype=cv2.CV_32F)
        gabor_response = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
        features['texture_score'] = np.mean(gabor_response) / 255.0
        
        return features
    
    def _extract_edge_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract edge and shape features."""
        features = {}
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Edge detection using Canny
        edges = cv2.Canny(gray, 50, 150)
        features['edge_density'] = np.sum(edges > 0) / edges.size
        
        # Gradient magnitude
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        features['gradient_strength'] = np.mean(gradient_magnitude) / 255.0
        
        # Corner detection (Harris corner detector)
        corners = cv2.cornerHarris(gray, 2, 3, 0.04)
        features['corner_density'] = np.sum(corners > 0.01 * corners.max()) / corners.size
        
        return features
    
    def _extract_quality_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract image quality features."""
        features = {}
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Sharpness using Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        features['sharpness_score'] = np.var(laplacian) / 10000.0  # Normalize
        
        # Blur detection using FFT
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude_spectrum = np.abs(fft_shift)
        
        # High frequency content indicates sharpness
        h, w = magnitude_spectrum.shape
        center_x, center_y = h // 2, w // 2
        high_freq_region = magnitude_spectrum[center_x-20:center_x+20, center_y-20:center_y+20]
        features['high_freq_content'] = np.mean(high_freq_region) / np.mean(magnitude_spectrum)
        
        # Noise estimation
        noise = cv2.medianBlur(gray, 5)
        noise_diff = cv2.absdiff(gray, noise)
        features['noise_level'] = np.mean(noise_diff) / 255.0
        
        return features
    
    def detect_faces_and_text(self, image: Image.Image) -> Dict[str, any]:
        """
        Detect faces and text in the image.
        
        Args:
            image: PIL Image object
            
        Returns:
            Dictionary containing face and text detection results
        """
        detection_results = {
            'face_count': 0,
            'face_areas': [],
            'text_regions': 0,
            'text_overlay_ratio': 0.0
        }
        
        try:
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Face detection using Haar cascades
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            detection_results['face_count'] = len(faces)
            detection_results['face_areas'] = [w * h for (x, y, w, h) in faces]
            
            # Text detection (simplified using edge detection)
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours that might be text (rectangular shapes)
            text_regions = 0
            text_area = 0
            total_area = cv_image.shape[0] * cv_image.shape[1]
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # Text regions typically have certain aspect ratios
                if 0.2 < aspect_ratio < 5.0 and w * h > 100:
                    text_regions += 1
                    text_area += w * h
            
            detection_results['text_regions'] = text_regions
            detection_results['text_overlay_ratio'] = text_area / total_area
            
        except Exception as e:
            print(f"Error in face/text detection: {e}")
        
        return detection_results
    
    def calculate_image_hash(self, image: Image.Image) -> Dict[str, str]:
        """
        Calculate various image hashes for duplicate detection.
        
        Args:
            image: PIL Image object
            
        Returns:
            Dictionary containing different hash values
        """
        try:
            hashes = {
                'ahash': str(imagehash.average_hash(image)),
                'phash': str(imagehash.phash(image)),
                'dhash': str(imagehash.dhash(image)),
                'whash': str(imagehash.whash(image))
            }
        except Exception as e:
            print(f"Error calculating image hashes: {e}")
            hashes = {
                'ahash': '',
                'phash': '',
                'dhash': '',
                'whash': ''
            }
        
        return hashes
    
    def process_image(self, image_source: Union[str, np.ndarray, Image.Image]) -> Dict[str, any]:
        """
        Complete image processing pipeline for a single image.
        
        Args:
            image_source: Image source (path, URL, array, or PIL Image)
            
        Returns:
            Dictionary containing all processed image data and features
        """
        # Load the image
        image = self.load_image(image_source)
        
        if image is None:
            return self._get_empty_image_features()
        
        # Extract all features
        try:
            # Basic image information
            width, height = image.size
            
            # Extract metadata
            metadata_features = self.extract_metadata(image)
            
            # Detect manipulation
            manipulation_features = self.detect_manipulation(image)
            
            # Extract visual features
            visual_features = self.extract_visual_features(image)
            
            # Detect faces and text
            detection_features = self.detect_faces_and_text(image)
            
            # Calculate image hashes
            image_hashes = self.calculate_image_hash(image)
            
            # Combine all features
            processed_data = {
                'image_loaded': True,
                'image_width': width,
                'image_height': height,
                'image_aspect_ratio': width / height if height > 0 else 1.0,
                'image_size_pixels': width * height,
                'metadata_features': metadata_features,
                'manipulation_features': manipulation_features,
                'visual_features': visual_features,
                'detection_features': detection_features,
                'image_hashes': image_hashes
            }
            
            return processed_data
            
        except Exception as e:
            print(f"Error processing image: {e}")
            return self._get_empty_image_features()
    
    def _get_empty_image_features(self) -> Dict[str, any]:
        """Return empty feature dictionary for failed image processing."""
        return {
            'image_loaded': False,
            'image_width': 0,
            'image_height': 0,
            'image_aspect_ratio': 1.0,
            'image_size_pixels': 0,
            'metadata_features': {
                'has_exif': False,
                'camera_make': '',
                'camera_model': '',
                'creation_date': '',
                'gps_info': False,
                'software_info': '',
                'metadata_inconsistency_score': 0.0
            },
            'manipulation_features': {feature: 0.0 for feature in ['jpeg_artifacts', 'ela_analysis', 
                                     'noise_analysis', 'lighting_analysis', 'compression_analysis', 
                                     'overall_manipulation_score']},
            'visual_features': {
                'resnet_features': np.zeros(model_config.IMAGE_EMBEDDING_DIM),
                'brightness_score': 0.0,
                'contrast_score': 0.0,
                'saturation_score': 0.0,
                'texture_score': 0.0,
                'edge_density': 0.0,
                'sharpness_score': 0.0
            },
            'detection_features': {
                'face_count': 0,
                'face_areas': [],
                'text_regions': 0,
                'text_overlay_ratio': 0.0
            },
            'image_hashes': {
                'ahash': '',
                'phash': '',
                'dhash': '',
                'whash': ''
            }
        }
    
    def batch_process(self, image_sources: List[Union[str, np.ndarray, Image.Image]], 
                     show_progress: bool = True) -> List[Dict[str, any]]:
        """
        Process multiple images in batch.
        
        Args:
            image_sources: List of image sources
            show_progress: Whether to show progress bar
            
        Returns:
            List of processed image data
        """
        from tqdm import tqdm
        
        results = []
        iterator = tqdm(image_sources, desc="Processing images") if show_progress else image_sources
        
        for image_source in iterator:
            processed = self.process_image(image_source)
            results.append(processed)
        
        return results


class ImageAugmentor:
    """
    Image augmentation for data augmentation and robustness testing.
    """
    
    def __init__(self):
        """Initialize image augmentor."""
        self.augmentation_transforms = [
            self.add_noise,
            self.adjust_brightness,
            self.adjust_contrast,
            self.add_blur,
            self.add_compression_artifacts
        ]
    
    def add_noise(self, image: Image.Image, noise_level: float = 0.1) -> Image.Image:
        """Add random noise to image."""
        img_array = np.array(image)
        noise = np.random.normal(0, noise_level * 255, img_array.shape)
        noisy_image = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_image)
    
    def adjust_brightness(self, image: Image.Image, factor: float = 1.2) -> Image.Image:
        """Adjust image brightness."""
        img_array = np.array(image).astype(np.float32)
        bright_image = np.clip(img_array * factor, 0, 255).astype(np.uint8)
        return Image.fromarray(bright_image)
    
    def adjust_contrast(self, image: Image.Image, factor: float = 1.2) -> Image.Image:
        """Adjust image contrast."""
        img_array = np.array(image).astype(np.float32)
        mean = np.mean(img_array)
        contrast_image = np.clip((img_array - mean) * factor + mean, 0, 255).astype(np.uint8)
        return Image.fromarray(contrast_image)
    
    def add_blur(self, image: Image.Image, blur_radius: int = 2) -> Image.Image:
        """Add blur to image."""
        return image.filter(Image.ImageFilter.GaussianBlur(radius=blur_radius))
    
    def add_compression_artifacts(self, image: Image.Image, quality: int = 30) -> Image.Image:
        """Add JPEG compression artifacts."""
        buffer = BytesIO()
        image.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        return Image.open(buffer)
    
    def augment_image(self, image: Image.Image, num_augmentations: int = 3) -> List[Image.Image]:
        """
        Apply random augmentations to an image.
        
        Args:
            image: Source image
            num_augmentations: Number of augmented versions to create
            
        Returns:
            List of augmented images
        """
        augmented_images = []
        
        for _ in range(num_augmentations):
            # Randomly select and apply transformations
            num_transforms = np.random.randint(1, 3)
            current_image = image.copy()
            
            for _ in range(num_transforms):
                transform = np.random.choice(self.augmentation_transforms)
                current_image = transform(current_image)
            
            augmented_images.append(current_image)
        
        return augmented_images


def create_image_features_dataframe(processed_images: List[Dict[str, any]]) -> pd.DataFrame:
    """
    Create a pandas DataFrame from processed image features.
    
    Args:
        processed_images: List of processed image data
        
    Returns:
        DataFrame with image features
    """
    features_list = []
    
    for processed in processed_images:
        # Combine all feature dictionaries (excluding ResNet features for now)
        row_features = {}
        
        # Basic image info
        row_features['image_loaded'] = processed.get('image_loaded', False)
        row_features['image_width'] = processed.get('image_width', 0)
        row_features['image_height'] = processed.get('image_height', 0)
        row_features['image_aspect_ratio'] = processed.get('image_aspect_ratio', 1.0)
        row_features['image_size_pixels'] = processed.get('image_size_pixels', 0)
        
        # Metadata features
        row_features.update(processed.get('metadata_features', {}))
        
        # Manipulation features
        row_features.update(processed.get('manipulation_features', {}))
        
        # Visual features (excluding ResNet features)
        visual_features = processed.get('visual_features', {})
        for key, value in visual_features.items():
            if key != 'resnet_features':  # Skip ResNet features for tabular data
                row_features[key] = value
        
        # Detection features
        detection_features = processed.get('detection_features', {})
        row_features['face_count'] = detection_features.get('face_count', 0)
        row_features['text_regions'] = detection_features.get('text_regions', 0)
        row_features['text_overlay_ratio'] = detection_features.get('text_overlay_ratio', 0.0)
        
        features_list.append(row_features)
    
    return pd.DataFrame(features_list)


# Example usage and testing
if __name__ == "__main__":
    # Initialize processor
    processor = ImagePreprocessor()
    
    # Example: Create a test image
    test_image = Image.new('RGB', (224, 224), color='red')
    
    # Process the image
    result = processor.process_image(test_image)
    
    print("Image Processing Results:")
    print(f"Image loaded: {result['image_loaded']}")
    print(f"Image size: {result['image_width']}x{result['image_height']}")
    print(f"Aspect ratio: {result['image_aspect_ratio']:.2f}")
    
    print("\nManipulation Detection:")
    for feature, value in result['manipulation_features'].items():
        print(f"  {feature}: {value:.3f}")
    
    print("\nVisual Features (sample):")
    visual_features = result['visual_features']
    for feature, value in visual_features.items():
        if feature != 'resnet_features':  # Skip the large feature vector
            print(f"  {feature}: {value:.3f}")
    
    print(f"\nResNet features shape: {visual_features['resnet_features'].shape}")
