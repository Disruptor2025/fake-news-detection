"""
Multi-modal ensemble model for fake news detection.

This module combines text, image, and social media analysis into a unified
classification system using neural network fusion.

Author: Idrees Khan
Course: AAI 501 - Introduction to Artificial Intelligence
Date: August 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union, Any
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import joblib
import warnings
warnings.filterwarnings('ignore')

from ..utils.config import model_config, feature_config
from .text_model import FakeNewsTextClassifier
from .image_model import FakeNewsImageClassifier
from ..data_preprocessing.social_processor import SocialMediaProcessor


class MultiModalDataset(Dataset):
    """
    Dataset for multi-modal fake news detection.
    """
    
    def __init__(self, text_features: np.ndarray, image_features: np.ndarray,
                 social_features: np.ndarray, labels: List[int]):
        """
        Initialize the multi-modal dataset.
        
        Args:
            text_features: Text feature embeddings
            image_features: Image feature embeddings
            social_features: Social media features
            labels: Classification labels
        """
        self.text_features = torch.tensor(text_features, dtype=torch.float32)
        self.image_features = torch.tensor(image_features, dtype=torch.float32)
        self.social_features = torch.tensor(social_features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        
        # Ensure all feature arrays have the same number of samples
        assert len(text_features) == len(image_features) == len(social_features) == len(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'text_features': self.text_features[idx],
            'image_features': self.image_features[idx],
            'social_features': self.social_features[idx],
            'label': self.labels[idx]
        }


class MultiModalFusionNetwork(nn.Module):
    """
    Neural network for fusing multi-modal features.
    """
    
    def __init__(self, text_dim: int, image_dim: int, social_dim: int,
                 hidden_dims: List[int] = None, dropout_rate: float = 0.3,
                 fusion_method: str = 'concatenation'):
        """
        Initialize the fusion network.
        
        Args:
            text_dim: Dimension of text features
            image_dim: Dimension of image features
            social_dim: Dimension of social media features
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout rate for regularization
            fusion_method: Method for combining features ('concatenation', 'attention', 'bilinear')
        """
        super(MultiModalFusionNetwork, self).__init__()
        
        self.fusion_method = fusion_method
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.social_dim = social_dim
        
        # Default hidden dimensions
        if hidden_dims is None:
            hidden_dims = model_config.FUSION_HIDDEN_DIMS
        
        # Feature projection layers (to normalize different modalities)
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(256)
        )
        
        self.image_projection = nn.Sequential(
            nn.Linear(image_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(256)
        )
        
        self.social_projection = nn.Sequential(
            nn.Linear(social_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(64)
        )
        
        # Fusion layer
        if fusion_method == 'concatenation':
            fusion_input_dim = 256 + 256 + 64  # Sum of projected dimensions
            self.fusion_layer = self._build_concatenation_fusion(fusion_input_dim, hidden_dims, dropout_rate)
        elif fusion_method == 'attention':
            fusion_input_dim = 256  # Attention outputs fixed size
            self.fusion_layer = self._build_attention_fusion(hidden_dims, dropout_rate)
        elif fusion_method == 'bilinear':
            fusion_input_dim = 256  # Bilinear fusion output size
            self.fusion_layer = self._build_bilinear_fusion(hidden_dims, dropout_rate)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 2)  # Binary classification
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _build_concatenation_fusion(self, input_dim: int, hidden_dims: List[int], dropout_rate: float):
        """Build concatenation-based fusion layer."""
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        return nn.Sequential(*layers)
    
    def _build_attention_fusion(self, hidden_dims: List[int], dropout_rate: float):
        """Build attention-based fusion layer."""
        # Multi-head attention for feature fusion
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Post-attention processing
        layers = []
        prev_dim = 256
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        return nn.Sequential(*layers)
    
    def _build_bilinear_fusion(self, hidden_dims: List[int], dropout_rate: float):
        """Build bilinear fusion layer."""
        # Bilinear fusion between text and image, then combine with social
        self.bilinear_text_image = nn.Bilinear(256, 256, 256)
        self.social_combiner = nn.Linear(64, 256)
        
        # Post-fusion processing
        layers = []
        prev_dim = 256
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, text_features, image_features, social_features):
        """
        Forward pass through the fusion network.
        
        Args:
            text_features: Text feature embeddings
            image_features: Image feature embeddings
            social_features: Social media features
            
        Returns:
            Classification logits
        """
        # Project features to common spaces
        text_proj = self.text_projection(text_features)
        image_proj = self.image_projection(image_features)
        social_proj = self.social_projection(social_features)
        
        # Fusion based on method
        if self.fusion_method == 'concatenation':
            # Simple concatenation
            fused_features = torch.cat([text_proj, image_proj, social_proj], dim=1)
            fusion_output = self.fusion_layer(fused_features)
            
        elif self.fusion_method == 'attention':
            # Attention-based fusion
            # Stack features for attention (batch_size, seq_len=3, embed_dim=256)
            features_stack = torch.stack([text_proj, image_proj, 
                                        F.adaptive_avg_pool1d(social_proj.unsqueeze(1), 256).squeeze(1)], dim=1)
            
            # Apply multi-head attention
            attended_features, _ = self.multihead_attention(features_stack, features_stack, features_stack)
            
            # Pool attended features
            pooled_features = torch.mean(attended_features, dim=1)
            fusion_output = self.fusion_layer(pooled_features)
            
        elif self.fusion_method == 'bilinear':
            # Bilinear fusion
            # Combine text and image with bilinear layer
            text_image_fused = self.bilinear_text_image(text_proj, image_proj)
            
            # Expand social features and combine
            social_expanded = self.social_combiner(social_proj)
            final_fused = text_image_fused + social_expanded
            
            fusion_output = self.fusion_layer(final_fused)
        
        # Classification
        logits = self.classifier(fusion_output)
        
        return logits


class MultiModalFakeNewsDetector:
    """
    Complete multi-modal fake news detection system.
    """
    
    def __init__(self, device: str = None, fusion_method: str = 'concatenation'):
        """
        Initialize the multi-modal detector.
        
        Args:
            device: Device to run models on (auto-detect if None)
            fusion_method: Method for combining features
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.fusion_method = fusion_method
        
        # Initialize individual models
        self.text_classifier = FakeNewsTextClassifier(device=self.device)
        self.image_classifier = FakeNewsImageClassifier(device=self.device)
        self.social_processor = SocialMediaProcessor()
        
        # Fusion model
        self.fusion_model = None
        self.optimizer = None
        self.scheduler = None
        
        # Feature scalers
        self.text_scaler = StandardScaler()
        self.image_scaler = StandardScaler()
        self.social_scaler = StandardScaler()
        
        # Training history
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        # Performance comparison
        self.model_comparisons = {}
    
    def prepare_multimodal_features(self, texts: List[str], images: List[Any],
                                   social_data: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare features from all modalities.
        
        Args:
            texts: List of article texts
            images: List of images (paths or PIL Images)
            social_data: List of social media data dictionaries
            
        Returns:
            Tuple of (text_features, image_features, social_features)
        """
        print("Extracting text features...")
        # Extract text features using BERT
        if not hasattr(self.text_classifier, 'model') or self.text_classifier.model is None:
            # Use pre-trained BERT for feature extraction without fine-tuning
            text_features = self.text_classifier.extract_linguistic_features(texts)
        else:
            # Use trained model to extract features
            dummy_labels = [0] * len(texts)
            text_dataloader = self.text_classifier.prepare_data(texts, dummy_labels, batch_size=32)
            
            text_embeddings = []
            self.text_classifier.model.eval()
            with torch.no_grad():
                for batch in tqdm(text_dataloader, desc="Extracting text embeddings"):
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    
                    # Get BERT embeddings
                    bert_output = self.text_classifier.model.bert(input_ids=input_ids, attention_mask=attention_mask)
                    embeddings = bert_output.pooler_output
                    text_embeddings.extend(embeddings.cpu().numpy())
            
            text_features = np.array(text_embeddings)
        
        print("Extracting image features...")
        # Extract image features
        if not hasattr(self.image_classifier, 'model') or self.image_classifier.model is None:
            # Use image processor features
            image_features = self.image_classifier.extract_image_features(images)
        else:
            # Use trained CNN to extract features
            image_features = self.image_classifier.extract_features(images)
        
        print("Extracting social media features...")
        # Extract social media features
        social_features_list = []
        for social_item in tqdm(social_data, desc="Processing social data"):
            processed_social = self.social_processor.process_social_data(
                user_data=social_item.get('user_data', {}),
                post_data=social_item.get('post_data'),
                sharing_data=social_item.get('sharing_data'),
                activity_data=social_item.get('activity_data')
            )
            
            # Convert to feature vector
            feature_vector = []
            for feature_name in feature_config.SOCIAL_FEATURES:
                feature_vector.append(processed_social.get(feature_name, 0.0))
            
            social_features_list.append(feature_vector)
        
        social_features = np.array(social_features_list, dtype=np.float32)
        
        return text_features, image_features, social_features
    
    def initialize_fusion_model(self, text_dim: int, image_dim: int, social_dim: int):
        """
        Initialize the fusion model.
        
        Args:
            text_dim: Dimension of text features
            image_dim: Dimension of image features
            social_dim: Dimension of social media features
        """
        self.fusion_model = MultiModalFusionNetwork(
            text_dim=text_dim,
            image_dim=image_dim,
            social_dim=social_dim,
            hidden_dims=model_config.FUSION_HIDDEN_DIMS,
            dropout_rate=model_config.DROPOUT_RATE,
            fusion_method=self.fusion_method
        )
        self.fusion_model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.fusion_model.parameters(),
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
    
    def train_individual_models(self, train_texts: List[str], train_images: List[Any],
                               train_social: List[Dict[str, Any]], train_labels: List[int],
                               val_texts: List[str], val_images: List[Any],
                               val_social: List[Dict[str, Any]], val_labels: List[int],
                               epochs: int = None):
        """
        Train individual modality models.
        
        Args:
            train_texts: Training texts
            train_images: Training images
            train_social: Training social media data
            train_labels: Training labels
            val_texts: Validation texts
            val_images: Validation images
            val_social: Validation social media data
            val_labels: Validation labels
            epochs: Number of training epochs
        """
        epochs = epochs or model_config.NUM_EPOCHS
        
        print("Training text classifier...")
        text_history = self.text_classifier.train(
            train_texts, train_labels, val_texts, val_labels, epochs=epochs
        )
        
        print("Training image classifier...")
        image_history = self.image_classifier.train(
            train_images, train_labels, val_images, val_labels, epochs=epochs
        )
        
        # Store individual model performance
        self.model_comparisons['text_only'] = {
            'final_val_acc': text_history['val_acc'][-1],
            'best_val_acc': max(text_history['val_acc']),
            'history': text_history
        }
        
        self.model_comparisons['image_only'] = {
            'final_val_acc': image_history['val_acc'][-1],
            'best_val_acc': max(image_history['val_acc']),
            'history': image_history
        }
    
    def train_fusion_model(self, train_texts: List[str], train_images: List[Any],
                          train_social: List[Dict[str, Any]], train_labels: List[int],
                          val_texts: List[str], val_images: List[Any],
                          val_social: List[Dict[str, Any]], val_labels: List[int],
                          epochs: int = None) -> Dict:
        """
        Train the fusion model.
        
        Args:
            train_texts: Training texts
            train_images: Training images
            train_social: Training social media data
            train_labels: Training labels
            val_texts: Validation texts
            val_images: Validation images
            val_social: Validation social media data
            val_labels: Validation labels
            epochs: Number of training epochs
            
        Returns:
            Training history dictionary
        """
        epochs = epochs or model_config.NUM_EPOCHS
        
        # Extract features from all modalities
        print("Preparing training features...")
        train_text_features, train_image_features, train_social_features = self.prepare_multimodal_features(
            train_texts, train_images, train_social
        )
        
        print("Preparing validation features...")
        val_text_features, val_image_features, val_social_features = self.prepare_multimodal_features(
            val_texts, val_images, val_social
        )
        
        # Normalize features
        train_text_features = self.text_scaler.fit_transform(train_text_features)
        train_image_features = self.image_scaler.fit_transform(train_image_features)
        train_social_features = self.social_scaler.fit_transform(train_social_features)
        
        val_text_features = self.text_scaler.transform(val_text_features)
        val_image_features = self.image_scaler.transform(val_image_features)
        val_social_features = self.social_scaler.transform(val_social_features)
        
        # Initialize fusion model
        self.initialize_fusion_model(
            text_dim=train_text_features.shape[1],
            image_dim=train_image_features.shape[1],
            social_dim=train_social_features.shape[1]
        )
        
        # Create datasets and dataloaders
        train_dataset = MultiModalDataset(
            train_text_features, train_image_features, train_social_features, train_labels
        )
        val_dataset = MultiModalDataset(
            val_text_features, val_image_features, val_social_features, val_labels
        )
        
        train_dataloader = DataLoader(
            train_dataset, batch_size=model_config.BATCH_SIZE, shuffle=True, num_workers=2
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=model_config.BATCH_SIZE, shuffle=False, num_workers=2
        )
        
        # Training loop
        best_val_accuracy = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = self._train_fusion_epoch(train_dataloader)
            
            # Validate
            val_loss, val_acc, val_metrics = self._evaluate_fusion(val_dataloader)
            
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
                self.save_fusion_model('best_fusion_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= model_config.EARLY_STOPPING_PATIENCE:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break
        
        # Store fusion model performance
        self.model_comparisons['multimodal_fusion'] = {
            'final_val_acc': self.training_history['val_acc'][-1],
            'best_val_acc': best_val_accuracy,
            'history': self.training_history
        }
        
        return self.training_history
    
    def _train_fusion_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Train fusion model for one epoch."""
        self.fusion_model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        for batch in tqdm(dataloader, desc="Training fusion"):
            # Move batch to device
            text_features = batch['text_features'].to(self.device)
            image_features = batch['image_features'].to(self.device)
            social_features = batch['social_features'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.fusion_model(text_features, image_features, social_features)
            
            # Calculate loss
            loss = F.cross_entropy(logits, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.fusion_model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Calculate accuracy
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_predictions
        
        return avg_loss, accuracy
    
    def _evaluate_fusion(self, dataloader: DataLoader) -> Tuple[float, float, Dict]:
        """Evaluate fusion model."""
        self.fusion_model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating fusion"):
                # Move batch to device
                text_features = batch['text_features'].to(self.device)
                image_features = batch['image_features'].to(self.device)
                social_features = batch['social_features'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                logits = self.fusion_model(text_features, image_features, social_features)
                
                # Calculate loss
                loss = F.cross_entropy(logits, labels)
                total_loss += loss.item()
                
                # Store predictions and labels
                probabilities = F.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_predictions)
        
        # Calculate detailed metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        # Calculate AUC-ROC
        probabilities_array = np.array(all_probabilities)
        auc_roc = roc_auc_score(all_labels, probabilities_array[:, 1])
        
        detailed_metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc_roc,
            'confusion_matrix': confusion_matrix(all_labels, all_predictions),
            'predictions': all_predictions,
            'true_labels': all_labels,
            'probabilities': all_probabilities
        }
        
        return avg_loss, accuracy, detailed_metrics
    
    def predict(self, texts: List[str], images: List[Any],
               social_data: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using the complete multi-modal system.
        
        Args:
            texts: List of article texts
            images: List of images
            social_data: List of social media data
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if self.fusion_model is None:
            raise ValueError("Fusion model not trained. Train the model first.")
        
        # Extract and normalize features
        text_features, image_features, social_features = self.prepare_multimodal_features(
            texts, images, social_data
        )
        
        text_features = self.text_scaler.transform(text_features)
        image_features = self.image_scaler.transform(image_features)
        social_features = self.social_scaler.transform(social_features)
        
        # Create dataset and dataloader
        dummy_labels = [0] * len(texts)
        dataset = MultiModalDataset(text_features, image_features, social_features, dummy_labels)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        # Make predictions
        self.fusion_model.eval()
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                text_feat = batch['text_features'].to(self.device)
                image_feat = batch['image_features'].to(self.device)
                social_feat = batch['social_features'].to(self.device)
                
                logits = self.fusion_model(text_feat, image_feat, social_feat)
                probabilities = F.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        return np.array(all_predictions), np.array(all_probabilities)
    
    def compare_models(self, test_texts: List[str], test_images: List[Any],
                      test_social: List[Dict[str, Any]], test_labels: List[int]) -> Dict:
        """
        Compare performance of different model configurations.
        
        Args:
            test_texts: Test texts
            test_images: Test images
            test_social: Test social media data
            test_labels: Test labels
            
        Returns:
            Comparison results dictionary
        """
        results = {}
        
        # Test text-only model
        if hasattr(self.text_classifier, 'model') and self.text_classifier.model is not None:
            text_predictions, text_probs = self.text_classifier.predict(test_texts)
            results['text_only'] = self._calculate_metrics(test_labels, text_predictions, text_probs)
        
        # Test image-only model
        if hasattr(self.image_classifier, 'model') and self.image_classifier.model is not None:
            image_predictions, image_probs = self.image_classifier.predict(test_images)
            results['image_only'] = self._calculate_metrics(test_labels, image_predictions, image_probs)
        
        # Test multi-modal model
        if self.fusion_model is not None:
            fusion_predictions, fusion_probs = self.predict(test_texts, test_images, test_social)
            results['multimodal'] = self._calculate_metrics(test_labels, fusion_predictions, fusion_probs)
        
        return results
    
    def _calculate_metrics(self, true_labels: List[int], predictions: np.ndarray,
                          probabilities: np.ndarray) -> Dict:
        """Calculate comprehensive metrics."""
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='weighted'
        )
        auc_roc = roc_auc_score(true_labels, probabilities[:, 1])
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc_roc,
            'confusion_matrix': confusion_matrix(true_labels, predictions)
        }
    
    def save_fusion_model(self, filepath: str):
        """Save the complete fusion model."""
        if self.fusion_model is None:
            raise ValueError("No fusion model to save")
        
        torch.save({
            'fusion_model_state_dict': self.fusion_model.state_dict(),
            'text_scaler': self.text_scaler,
            'image_scaler': self.image_scaler,
            'social_scaler': self.social_scaler,
            'training_history': self.training_history,
            'model_comparisons': self.model_comparisons,
            'fusion_method': self.fusion_method
        }, filepath)
        print(f"Fusion model saved to {filepath}")
    
    def load_fusion_model(self, filepath: str):
        """Load the complete fusion model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Load scalers and configuration
        self.text_scaler = checkpoint['text_scaler']
        self.image_scaler = checkpoint['image_scaler']
        self.social_scaler = checkpoint['social_scaler']
        self.training_history = checkpoint['training_history']
        self.model_comparisons = checkpoint['model_comparisons']
        self.fusion_method = checkpoint['fusion_method']
        
        # Initialize and load fusion model
        # Note: This requires knowing the feature dimensions, which should be stored in checkpoint
        # For now, we'll use default dimensions
        self.initialize_fusion_model(text_dim=768, image_dim=2048, social_dim=25)
        self.fusion_model.load_state_dict(checkpoint['fusion_model_state_dict'])
        
        print(f"Fusion model loaded from {filepath}")
    
    def plot_model_comparison(self, save_path: str = None):
        """Plot comparison of different models."""
        if not self.model_comparisons:
            print("No model comparisons available. Train models first.")
            return
        
        models = list(self.model_comparisons.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        # Extract metrics for plotting
        metric_values = {metric: [] for metric in metrics}
        
        for model in models:
            if 'final_val_acc' in self.model_comparisons[model]:
                # Use validation metrics if available
                val_acc = self.model_comparisons[model]['final_val_acc']
                metric_values['accuracy'].append(val_acc)
                # For now, use accuracy as proxy for other metrics
                metric_values['precision'].append(val_acc)
                metric_values['recall'].append(val_acc)
                metric_values['f1_score'].append(val_acc)
        
        # Create comparison plot
        x = np.arange(len(models))
        width = 0.2
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, metric in enumerate(metrics):
            ax.bar(x + i * width, metric_values[metric], width, label=metric.replace('_', ' ').title())
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels([model.replace('_', ' ').title() for model in models])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


# Example usage and testing
if __name__ == "__main__":
    # Create sample multi-modal data
    sample_texts = [
        "Scientists publish peer-reviewed research on climate change effects.",
        "SHOCKING discovery doctors don't want you to know! Click here!",
        "Federal Reserve announces interest rate decision after meeting.",
        "You won't BELIEVE what happened next! Number 5 will AMAZE you!"
    ]
    
    # Create sample images (PIL Images)
    from PIL import Image
    sample_images = [Image.new('RGB', (224, 224), color=(i*50, 100, 150)) for i in range(4)]
    
    # Create sample social data
    sample_social = [
        {
            'user_data': {
                'username': f'user_{i}',
                'verified': i % 2 == 0,
                'follower_count': 1000 + i * 500,
                'following_count': 200 + i * 100,
                'account_age_days': 365 + i * 100
            },
            'post_data': {
                'content': sample_texts[i],
                'likes': 100 + i * 50,
                'shares': 10 + i * 5,
                'views': 1000 + i * 500
            }
        }
        for i in range(4)
    ]
    
    sample_labels = [0, 1, 0, 1]  # 0 = real, 1 = fake
    
    # Test multi-modal detector
    print("Testing Multi-Modal Fake News Detector:")
    detector = MultiModalFakeNewsDetector(fusion_method='concatenation')
    
    # Note: In practice, you would need larger datasets for training
    print("This is a minimal example. For full training, use larger datasets.")
    
    # Extract features (without training individual models)
    text_features, image_features, social_features = detector.prepare_multimodal_features(
        sample_texts, sample_images, sample_social
    )
    
    print(f"Text features shape: {text_features.shape}")
    print(f"Image features shape: {image_features.shape}")
    print(f"Social features shape: {social_features.shape}")
