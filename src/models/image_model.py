"""
CNN-based image classification model for fake news detection.

This module implements image manipulation detection and feature extraction
using ResNet architecture for the multi-modal fake news detection system.

Author: Idrees Khan
Course: AAI 501 - Introduction to Artificial Intelligence
Date: August 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from ..utils.config import model_config
from ..data_preprocessing.image_processor import ImagePreprocessor


class FakeNewsImageDataset(Dataset):
    """
    Custom dataset for fake news image classification.
    """
    
    def __init__(self, image_paths: List[str], labels: List[int], 
                 transform: transforms.Compose = None, load_images: bool = True):
        """
        Initialize the dataset.
        
        Args:
            image_paths: List of image file paths or PIL Images
            labels: List of labels (0 for real, 1 for fake)
            transform: Image transformations
            load_images: Whether to load images immediately
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.load_images = load_images
        
        # Load images if requested
        self.images = []
        if load_images:
            for path in tqdm(image_paths, desc="Loading images"):
                try:
                    if isinstance(path, str):
                        img = Image.open(path).convert('RGB')
                    else:
                        img = path  # Assume it's already a PIL Image
                    self.images.append(img)
                except Exception as e:
                    print(f"Error loading image {path}: {e}")
                    # Create a default image
                    self.images.append(Image.new('RGB', (224, 224), color='gray'))
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        if self.load_images:
            image = self.images[idx]
        else:
            # Load image on demand
            try:
                if isinstance(self.image_paths[idx], str):
                    image = Image.open(self.image_paths[idx]).convert('RGB')
                else:
                    image = self.image_paths[idx]
            except:
                image = Image.new('RGB', (224, 224), color='gray')
        
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long)
        }


class ManipulationDetectionCNN(nn.Module):
    """
    CNN model for detecting image manipulation based on ResNet architecture.
    """
    
    def __init__(self, pretrained: bool = True, num_classes: int = 2, 
                 n_additional_features: int = 8, dropout_rate: float = 0.3):
        """
        Initialize the manipulation detection CNN.
        
        Args:
            pretrained: Whether to use pre-trained ResNet weights
            num_classes: Number of output classes
            n_additional_features: Number of additional image features
            dropout_rate: Dropout rate for regularization
        """
        super(ManipulationDetectionCNN, self).__init__()
        
        # Load pre-trained ResNet-50
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Feature dimensions
        resnet_features = 2048  # ResNet-50 output dimension
        self.n_additional_features = n_additional_features
        combined_features = resnet_features + n_additional_features
        
        # Additional feature processing layer
        self.feature_fusion = nn.Sequential(
            nn.Linear(combined_features, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(512)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights for new layers
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for newly added layers."""
        for module in [self.feature_fusion, self.classifier]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(layer.bias, 0)
    
    def forward(self, images, additional_features=None):
        """
        Forward pass through the model.
        
        Args:
            images: Batch of input images
            additional_features: Additional image analysis features
            
        Returns:
            Logits for classification
        """
        # Extract ResNet features
        resnet_features = self.backbone(images)
        resnet_features = resnet_features.view(resnet_features.size(0), -1)
        
        # Combine with additional features if provided
        if additional_features is not None:
            combined_features = torch.cat([resnet_features, additional_features], dim=1)
        else:
            # Use zero padding if no additional features provided
            batch_size = resnet_features.size(0)
            zero_features = torch.zeros(batch_size, self.n_additional_features, 
                                       device=resnet_features.device)
            combined_features = torch.cat([resnet_features, zero_features], dim=1)
        
        # Feature fusion
        fused_features = self.feature_fusion(combined_features)
        
        # Classification
        logits = self.classifier(fused_features)
        
        return logits


class FakeNewsImageClassifier:
    """
    Complete image classification pipeline for fake news detection.
    """
    
    def __init__(self, device: str = None):
        """
        Initialize the image classifier.
        
        Args:
            device: Device to run model on (auto-detect if None)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize image processor
        self.image_processor = ImagePreprocessor()
        
        # Define image transformations
        self.train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(model_config.IMAGE_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize(model_config.IMAGE_SIZE),
            transforms.CenterCrop(model_config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Model and training components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        
        # Training history
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
    
    def extract_image_features(self, images: List[Union[str, Image.Image]]) -> np.ndarray:
        """
        Extract additional image analysis features.
        
        Args:
            images: List of image paths or PIL Images
            
        Returns:
            NumPy array of image features
        """
        features_list = []
        
        for image in tqdm(images, desc="Extracting image features"):
            processed = self.image_processor.process_image(image)
            
            # Extract specific features for the model
            feature_vector = []
            
            # Manipulation features
            manipulation_features = processed['manipulation_features']
            feature_vector.append(manipulation_features.get('overall_manipulation_score', 0.0))
            
            # Visual quality features
            visual_features = processed['visual_features']
            feature_vector.extend([
                visual_features.get('brightness_score', 0.0),
                visual_features.get('contrast_score', 0.0),
                visual_features.get('saturation_score', 0.0),
                visual_features.get('sharpness_score', 0.0),
                visual_features.get('edge_density', 0.0)
            ])
            
            # Detection features
            detection_features = processed['detection_features']
            feature_vector.extend([
                detection_features.get('face_count', 0) / 10.0,  # Normalize
                detection_features.get('text_overlay_ratio', 0.0)
            ])
            
            features_list.append(feature_vector)
        
        return np.array(features_list, dtype=np.float32)
    
    def prepare_data(self, images: List[Union[str, Image.Image]], labels: List[int],
                    additional_features: Optional[np.ndarray] = None,
                    is_training: bool = True, batch_size: int = None) -> DataLoader:
        """
        Prepare data for training or inference.
        
        Args:
            images: List of image paths or PIL Images
            labels: List of labels (0 for real, 1 for fake)
            additional_features: Additional image features
            is_training: Whether this is training data (affects transforms)
            batch_size: Batch size for DataLoader
            
        Returns:
            DataLoader for the dataset
        """
        batch_size = batch_size or model_config.BATCH_SIZE
        transform = self.train_transform if is_training else self.val_transform
        
        # Create dataset
        dataset = FakeNewsImageDataset(
            image_paths=images,
            labels=labels,
            transform=transform,
            load_images=True
        )
        
        # Create DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_training,
            num_workers=2,
            pin_memory=True if self.device == 'cuda' else False
        )
        
        return dataloader
    
    def initialize_model(self, n_additional_features: int = 8):
        """
        Initialize the CNN model.
        
        Args:
            n_additional_features: Number of additional image features
        """
        self.model = ManipulationDetectionCNN(
            pretrained=True,
            num_classes=2,
            n_additional_features=n_additional_features,
            dropout_rate=model_config.DROPOUT_RATE
        )
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=model_config.LEARNING_RATE,
            weight_decay=0.01
        )
        
        # Initialize scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )
    
    def train_epoch(self, train_dataloader: DataLoader,
                   additional_features: Optional[np.ndarray] = None) -> Tuple[float, float]:
        """
        Train the model for one epoch.
        
        Args:
            train_dataloader: Training data loader
            additional_features: Additional image features
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="Training")):
            # Move batch to device
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Get additional features for this batch if provided
            batch_additional_features = None
            if additional_features is not None:
                start_idx = batch_idx * train_dataloader.batch_size
                end_idx = start_idx + images.size(0)
                batch_additional_features = torch.tensor(
                    additional_features[start_idx:end_idx],
                    dtype=torch.float32,
                    device=self.device
                )
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(images, batch_additional_features)
            
            # Calculate loss
            loss = F.cross_entropy(logits, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Calculate accuracy
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_dataloader)
        accuracy = correct_predictions / total_predictions
        
        return avg_loss, accuracy
    
    def evaluate(self, val_dataloader: DataLoader,
                additional_features: Optional[np.ndarray] = None) -> Tuple[float, float, Dict]:
        """
        Evaluate the model.
        
        Args:
            val_dataloader: Validation data loader
            additional_features: Additional image features
            
        Returns:
            Tuple of (average_loss, accuracy, detailed_metrics)
        """
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_dataloader, desc="Evaluating")):
                # Move batch to device
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Get additional features for this batch if provided
                batch_additional_features = None
                if additional_features is not None:
                    start_idx = batch_idx * val_dataloader.batch_size
                    end_idx = start_idx + images.size(0)
                    batch_additional_features = torch.tensor(
                        additional_features[start_idx:end_idx],
                        dtype=torch.float32,
                        device=self.device
                    )
                
                # Forward pass
                logits = self.model(images, batch_additional_features)
                
                # Calculate loss
                loss = F.cross_entropy(logits, labels)
                total_loss += loss.item()
                
                # Store predictions and labels
                predictions = torch.argmax(logits, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_dataloader)
        accuracy = accuracy_score(all_labels, all_predictions)
        
        # Calculate detailed metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        detailed_metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': confusion_matrix(all_labels, all_predictions),
            'predictions': all_predictions,
            'true_labels': all_labels
        }
        
        return avg_loss, accuracy, detailed_metrics
    
    def train(self, train_images: List[Union[str, Image.Image]], train_labels: List[int],
              val_images: List[Union[str, Image.Image]], val_labels: List[int],
              epochs: int = None, use_additional_features: bool = True) -> Dict:
        """
        Train the complete model.
        
        Args:
            train_images: Training images
            train_labels: Training labels
            val_images: Validation images
            val_labels: Validation labels
            epochs: Number of training epochs
            use_additional_features: Whether to use additional image features
            
        Returns:
            Training history dictionary
        """
        epochs = epochs or model_config.NUM_EPOCHS
        
        # Extract additional features if requested
        train_additional_features = None
        val_additional_features = None
        
        if use_additional_features:
            print("Extracting additional features for training images...")
            train_additional_features = self.extract_image_features(train_images)
            print("Extracting additional features for validation images...")
            val_additional_features = self.extract_image_features(val_images)
            n_features = train_additional_features.shape[1]
        else:
            n_features = 0
        
        # Initialize model
        self.initialize_model(n_additional_features=n_features)
        
        # Prepare data loaders
        train_dataloader = self.prepare_data(train_images, train_labels, 
                                           is_training=True)
        val_dataloader = self.prepare_data(val_images, val_labels, 
                                         is_training=False)
        
        # Training loop
        best_val_accuracy = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_dataloader, train_additional_features)
            
            # Validate
            val_loss, val_acc, val_metrics = self.evaluate(val_dataloader, val_additional_features)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Store history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_acc'].append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"Val F1: {val_metrics['f1_score']:.4f}")
            
            # Early stopping
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                patience_counter = 0
                # Save best model
                self.save_model('best_image_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= model_config.EARLY_STOPPING_PATIENCE:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break
        
        return self.training_history
    
    def predict(self, images: List[Union[str, Image.Image]], 
               use_additional_features: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new images.
        
        Args:
            images: List of images to classify
            use_additional_features: Whether to use additional features
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded. Train the model first.")
        
        self.model.eval()
        
        # Extract additional features if needed
        additional_features = None
        if use_additional_features:
            additional_features = self.extract_image_features(images)
        
        # Create dummy labels for dataset
        dummy_labels = [0] * len(images)
        dataloader = self.prepare_data(images, dummy_labels, is_training=False, batch_size=32)
        
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Predicting")):
                # Move batch to device
                images_batch = batch['image'].to(self.device)
                
                # Get additional features for this batch if provided
                batch_additional_features = None
                if additional_features is not None:
                    start_idx = batch_idx * dataloader.batch_size
                    end_idx = start_idx + images_batch.size(0)
                    batch_additional_features = torch.tensor(
                        additional_features[start_idx:end_idx],
                        dtype=torch.float32,
                        device=self.device
                    )
                
                # Forward pass
                logits = self.model(images_batch, batch_additional_features)
                
                # Get predictions and probabilities
                probabilities = F.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        return np.array(all_predictions), np.array(all_probabilities)
    
    def extract_features(self, images: List[Union[str, Image.Image]]) -> np.ndarray:
        """
        Extract deep features from images using the trained model.
        
        Args:
            images: List of images
            
        Returns:
            Feature embeddings
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded.")
        
        self.model.eval()
        
        # Create dummy labels and dataloader
        dummy_labels = [0] * len(images)
        dataloader = self.prepare_data(images, dummy_labels, is_training=False, batch_size=32)
        
        all_features = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting features"):
                images_batch = batch['image'].to(self.device)
                
                # Get features from backbone (before classification)
                features = self.model.backbone(images_batch)
                features = features.view(features.size(0), -1)
                
                all_features.extend(features.cpu().numpy())
        
        return np.array(all_features)
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'training_history': self.training_history,
            'model_config': {
                'n_additional_features': self.model.n_additional_features,
                'device': self.device
            }
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Initialize model with saved config
        model_config_saved = checkpoint['model_config']
        self.initialize_model(model_config_saved['n_additional_features'])
        
        # Load state dict
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.training_history = checkpoint['training_history']
        
        print(f"Model loaded from {filepath}")
    
    def plot_training_history(self, save_path: str = None):
        """Plot training history."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        axes[0].plot(self.training_history['train_loss'], label='Training Loss')
        axes[0].plot(self.training_history['val_loss'], label='Validation Loss')
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot accuracy
        axes[1].plot(self.training_history['train_acc'], label='Training Accuracy')
        axes[1].plot(self.training_history['val_acc'], label='Validation Accuracy')
        axes[1].set_title('Model Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_confusion_matrix(self, true_labels: List[int], predictions: List[int],
                             save_path: str = None):
        """Plot confusion matrix."""
        cm = confusion_matrix(true_labels, predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Real', 'Fake'],
                   yticklabels=['Real', 'Fake'])
        plt.title('Confusion Matrix - Image Classifier')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def visualize_predictions(self, images: List[Union[str, Image.Image]], 
                            predictions: np.ndarray, probabilities: np.ndarray,
                            num_samples: int = 8, save_path: str = None):
        """
        Visualize model predictions on sample images.
        
        Args:
            images: List of images
            predictions: Model predictions
            probabilities: Model prediction probabilities
            num_samples: Number of samples to visualize
            save_path: Path to save the visualization
        """
        num_samples = min(num_samples, len(images))
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for i in range(num_samples):
            # Load and display image
            if isinstance(images[i], str):
                img = Image.open(images[i]).convert('RGB')
            else:
                img = images[i]
            
            axes[i].imshow(img)
            axes[i].axis('off')
            
            # Add prediction information
            pred_label = 'Fake' if predictions[i] == 1 else 'Real'
            confidence = probabilities[i][predictions[i]]
            color = 'red' if predictions[i] == 1 else 'green'
            
            axes[i].set_title(f'{pred_label}\nConfidence: {confidence:.3f}', 
                            color=color, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


class SimpleImageClassifier:
    """
    Simple baseline image classifier for comparison.
    """
    
    def __init__(self, method: str = 'svm'):
        """
        Initialize simple classifier.
        
        Args:
            method: 'svm', 'random_forest', or 'logistic_regression'
        """
        self.method = method
        self.classifier = None
        self.image_processor = ImagePreprocessor()
    
    def extract_simple_features(self, images: List[Union[str, Image.Image]]) -> np.ndarray:
        """Extract simple features from images."""
        features_list = []
        
        for image in tqdm(images, desc="Extracting simple features"):
            processed = self.image_processor.process_image(image)
            
            # Extract basic features
            feature_vector = []
            
            # Basic image properties
            feature_vector.extend([
                processed.get('image_width', 0) / 1000.0,  # Normalize
                processed.get('image_height', 0) / 1000.0,
                processed.get('image_aspect_ratio', 1.0)
            ])
            
            # Manipulation features
            manipulation_features = processed['manipulation_features']
            for key in ['overall_manipulation_score', 'jpeg_artifacts', 'ela_analysis']:
                feature_vector.append(manipulation_features.get(key, 0.0))
            
            # Visual features
            visual_features = processed['visual_features']
            for key in ['brightness_score', 'contrast_score', 'sharpness_score']:
                feature_vector.append(visual_features.get(key, 0.0))
            
            features_list.append(feature_vector)
        
        return np.array(features_list)
    
    def train(self, train_images: List[Union[str, Image.Image]], 
             train_labels: List[int]) -> Dict:
        """Train the simple classifier."""
        from sklearn.svm import SVC
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        
        # Extract features
        features = self.extract_simple_features(train_images)
        
        # Choose classifier
        if self.method == 'svm':
            classifier = SVC(probability=True, random_state=42)
        elif self.method == 'random_forest':
            classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        elif self.method == 'logistic_regression':
            classifier = LogisticRegression(random_state=42)
        
        # Create pipeline with scaling
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', classifier)
        ])
        
        # Train
        self.pipeline.fit(features, train_labels)
        
        return {'method': self.method, 'trained': True}
    
    def predict(self, images: List[Union[str, Image.Image]]) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions."""
        if self.pipeline is None:
            raise ValueError("Model not trained")
        
        features = self.extract_simple_features(images)
        predictions = self.pipeline.predict(features)
        probabilities = self.pipeline.predict_proba(features)
        
        return predictions, probabilities


# Example usage and testing
if __name__ == "__main__":
    # Create sample images (in practice, you would load real images)
    sample_images = []
    for i in range(4):
        # Create test images with different characteristics
        if i % 2 == 0:
            # "Real" image - simple, clean
            img = Image.new('RGB', (224, 224), color=(100, 150, 200))
        else:
            # "Fake" image - more complex, artificial-looking
            img = Image.new('RGB', (224, 224), color=(255, 100, 100))
        sample_images.append(img)
    
    sample_labels = [0, 1, 0, 1]  # 0 = real, 1 = fake
    
    # Test simple classifier
    print("Testing Simple Image Classifier:")
    simple_classifier = SimpleImageClassifier('svm')
    simple_classifier.train(sample_images, sample_labels)
    predictions, probabilities = simple_classifier.predict(sample_images)
    
    print("Simple classifier predictions:")
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        print(f"Image {i+1}: {'Fake' if pred == 1 else 'Real'} (confidence: {prob[pred]:.3f})")
    
    # Test CNN classifier (commented out due to computational requirements)
    # print("\nTesting CNN Image Classifier:")
    # cnn_classifier = FakeNewsImageClassifier()
    # # Note: In practice, you would need larger datasets for training
    # history = cnn_classifier.train(sample_images, sample_labels, sample_images, sample_labels, epochs=1)
    # predictions, probabilities = cnn_classifier.predict(sample_images)
