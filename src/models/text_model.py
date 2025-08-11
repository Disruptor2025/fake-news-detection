"""
BERT-based text classification model for fake news detection.

This module implements a transformer-based text classifier using BERT
for contextual understanding of news article content.

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
from typing import List, Dict, Tuple, Optional
from transformers import (
    BertModel, BertTokenizer, BertForSequenceClassification,
    AdamW, get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from ..utils.config import model_config
from ..data_preprocessing.text_processor import TextPreprocessor


class FakeNewsTextDataset(Dataset):
    """
    Custom dataset for fake news text classification.
    """
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer: BertTokenizer, 
                 max_length: int = None):
        """
        Initialize the dataset.
        
        Args:
            texts: List of text articles
            labels: List of labels (0 for real, 1 for fake)
            tokenizer: BERT tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length or model_config.MAX_SEQUENCE_LENGTH
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize the text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


class BertFakeNewsClassifier(nn.Module):
    """
    BERT-based fake news classifier with additional linguistic features.
    """
    
    def __init__(self, n_linguistic_features: int = 15, dropout_rate: float = 0.3):
        """
        Initialize the BERT classifier.
        
        Args:
            n_linguistic_features: Number of additional linguistic features
            dropout_rate: Dropout rate for regularization
        """
        super(BertFakeNewsClassifier, self).__init__()
        
        # Load pre-trained BERT model
        self.bert = BertModel.from_pretrained(model_config.BERT_MODEL_NAME)
        
        # Additional feature processing
        self.linguistic_features_dim = n_linguistic_features
        
        # Feature fusion layers
        bert_output_dim = self.bert.config.hidden_size
        combined_dim = bert_output_dim + n_linguistic_features
        
        self.feature_fusion = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 2)  # Binary classification
        )
        
        # Freeze BERT layers initially (can be unfrozen for fine-tuning)
        self.freeze_bert()
    
    def freeze_bert(self):
        """Freeze BERT parameters."""
        for param in self.bert.parameters():
            param.requires_grad = False
    
    def unfreeze_bert(self):
        """Unfreeze BERT parameters for fine-tuning."""
        for param in self.bert.parameters():
            param.requires_grad = True
    
    def forward(self, input_ids, attention_mask, linguistic_features=None):
        """
        Forward pass through the model.
        
        Args:
            input_ids: Tokenized input text
            attention_mask: Attention mask for input
            linguistic_features: Additional linguistic features
            
        Returns:
            Logits for binary classification
        """
        # BERT encoding
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = bert_output.pooler_output  # [CLS] token representation
        
        # Combine with linguistic features if provided
        if linguistic_features is not None:
            combined_features = torch.cat([pooled_output, linguistic_features], dim=1)
        else:
            # Use zero padding if no linguistic features provided
            batch_size = pooled_output.size(0)
            zero_features = torch.zeros(batch_size, self.linguistic_features_dim, 
                                       device=pooled_output.device)
            combined_features = torch.cat([pooled_output, zero_features], dim=1)
        
        # Feature fusion
        fused_features = self.feature_fusion(combined_features)
        
        # Classification
        logits = self.classifier(fused_features)
        
        return logits


class FakeNewsTextClassifier:
    """
    Complete text classification pipeline for fake news detection.
    """
    
    def __init__(self, model_name: str = None, device: str = None):
        """
        Initialize the text classifier.
        
        Args:
            model_name: BERT model name (default from config)
            device: Device to run model on (auto-detect if None)
        """
        self.model_name = model_name or model_config.BERT_MODEL_NAME
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        
        # Initialize text processor for feature extraction
        self.text_processor = TextPreprocessor()
        
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
    
    def prepare_data(self, texts: List[str], labels: List[int], 
                    linguistic_features: Optional[np.ndarray] = None,
                    batch_size: int = None) -> DataLoader:
        """
        Prepare data for training or inference.
        
        Args:
            texts: List of text articles
            labels: List of labels (0 for real, 1 for fake)
            linguistic_features: Additional linguistic features
            batch_size: Batch size for DataLoader
            
        Returns:
            DataLoader for the dataset
        """
        batch_size = batch_size or model_config.BATCH_SIZE
        
        # Create dataset
        dataset = FakeNewsTextDataset(
            texts=texts,
            labels=labels,
            tokenizer=self.tokenizer,
            max_length=model_config.MAX_SEQUENCE_LENGTH
        )
        
        # Create DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
        )
        
        return dataloader
    
    def extract_linguistic_features(self, texts: List[str]) -> np.ndarray:
        """
        Extract linguistic features from texts.
        
        Args:
            texts: List of text articles
            
        Returns:
            NumPy array of linguistic features
        """
        features_list = []
        
        for text in tqdm(texts, desc="Extracting linguistic features"):
            processed = self.text_processor.process_article(text)
            
            # Get linguistic features
            ling_features = processed['linguistic_features']
            bias_features = processed['bias_features']
            
            # Combine features
            feature_vector = []
            for feature_name in ['sentiment_polarity', 'sentiment_subjectivity', 'readability_score',
                               'word_count', 'sentence_count', 'avg_word_length', 'caps_ratio',
                               'punctuation_ratio', 'question_marks', 'exclamation_marks',
                               'urls_count', 'mentions_count', 'hashtags_count', 'numbers_count',
                               'emotional_words_ratio']:
                feature_vector.append(ling_features.get(feature_name, 0.0))
            
            # Add bias features
            for feature_name in ['sensational_score', 'urgency_score', 'authority_claims', 'absolute_terms']:
                feature_vector.append(bias_features.get(feature_name, 0.0))
            
            features_list.append(feature_vector)
        
        return np.array(features_list, dtype=np.float32)
    
    def initialize_model(self, n_linguistic_features: int = 19):
        """
        Initialize the BERT model.
        
        Args:
            n_linguistic_features: Number of linguistic features
        """
        self.model = BertFakeNewsClassifier(
            n_linguistic_features=n_linguistic_features,
            dropout_rate=model_config.DROPOUT_RATE
        )
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=model_config.LEARNING_RATE,
            weight_decay=0.01
        )
    
    def train_epoch(self, train_dataloader: DataLoader, 
                   linguistic_features: Optional[np.ndarray] = None) -> Tuple[float, float]:
        """
        Train the model for one epoch.
        
        Args:
            train_dataloader: Training data loader
            linguistic_features: Linguistic features array
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="Training")):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Get linguistic features for this batch if provided
            batch_linguistic_features = None
            if linguistic_features is not None:
                start_idx = batch_idx * train_dataloader.batch_size
                end_idx = start_idx + input_ids.size(0)
                batch_linguistic_features = torch.tensor(
                    linguistic_features[start_idx:end_idx],
                    dtype=torch.float32,
                    device=self.device
                )
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                linguistic_features=batch_linguistic_features
            )
            
            # Calculate loss
            loss = F.cross_entropy(logits, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Update scheduler if available
            if self.scheduler:
                self.scheduler.step()
            
            # Calculate accuracy
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_dataloader)
        accuracy = correct_predictions / total_predictions
        
        return avg_loss, accuracy
    
    def evaluate(self, val_dataloader: DataLoader, 
                linguistic_features: Optional[np.ndarray] = None) -> Tuple[float, float, Dict]:
        """
        Evaluate the model.
        
        Args:
            val_dataloader: Validation data loader
            linguistic_features: Linguistic features array
            
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
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Get linguistic features for this batch if provided
                batch_linguistic_features = None
                if linguistic_features is not None:
                    start_idx = batch_idx * val_dataloader.batch_size
                    end_idx = start_idx + input_ids.size(0)
                    batch_linguistic_features = torch.tensor(
                        linguistic_features[start_idx:end_idx],
                        dtype=torch.float32,
                        device=self.device
                    )
                
                # Forward pass
                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    linguistic_features=batch_linguistic_features
                )
                
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
    
    def train(self, train_texts: List[str], train_labels: List[int],
              val_texts: List[str], val_labels: List[int],
              epochs: int = None, use_linguistic_features: bool = True) -> Dict:
        """
        Train the complete model.
        
        Args:
            train_texts: Training texts
            train_labels: Training labels
            val_texts: Validation texts
            val_labels: Validation labels
            epochs: Number of training epochs
            use_linguistic_features: Whether to use linguistic features
            
        Returns:
            Training history dictionary
        """
        epochs = epochs or model_config.NUM_EPOCHS
        
        # Extract linguistic features if requested
        train_linguistic_features = None
        val_linguistic_features = None
        
        if use_linguistic_features:
            print("Extracting linguistic features for training data...")
            train_linguistic_features = self.extract_linguistic_features(train_texts)
            print("Extracting linguistic features for validation data...")
            val_linguistic_features = self.extract_linguistic_features(val_texts)
            n_features = train_linguistic_features.shape[1]
        else:
            n_features = 0
        
        # Initialize model
        self.initialize_model(n_linguistic_features=n_features)
        
        # Prepare data loaders
        train_dataloader = self.prepare_data(train_texts, train_labels)
        val_dataloader = self.prepare_data(val_texts, val_labels)
        
        # Initialize scheduler
        total_steps = len(train_dataloader) * epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0.1 * total_steps,
            num_training_steps=total_steps
        )
        
        # Training loop
        best_val_accuracy = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_dataloader, train_linguistic_features)
            
            # Validate
            val_loss, val_acc, val_metrics = self.evaluate(val_dataloader, val_linguistic_features)
            
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
                self.save_model('best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= model_config.EARLY_STOPPING_PATIENCE:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break
        
        return self.training_history
    
    def predict(self, texts: List[str], use_linguistic_features: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new texts.
        
        Args:
            texts: List of texts to classify
            use_linguistic_features: Whether to use linguistic features
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded. Train the model first.")
        
        self.model.eval()
        
        # Extract linguistic features if needed
        linguistic_features = None
        if use_linguistic_features:
            linguistic_features = self.extract_linguistic_features(texts)
        
        # Create dummy labels for dataset
        dummy_labels = [0] * len(texts)
        dataloader = self.prepare_data(texts, dummy_labels, batch_size=32)
        
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Predicting")):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Get linguistic features for this batch if provided
                batch_linguistic_features = None
                if linguistic_features is not None:
                    start_idx = batch_idx * dataloader.batch_size
                    end_idx = start_idx + input_ids.size(0)
                    batch_linguistic_features = torch.tensor(
                        linguistic_features[start_idx:end_idx],
                        dtype=torch.float32,
                        device=self.device
                    )
                
                # Forward pass
                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    linguistic_features=batch_linguistic_features
                )
                
                # Get predictions and probabilities
                probabilities = F.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        return np.array(all_predictions), np.array(all_probabilities)
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'tokenizer': self.tokenizer,
            'training_history': self.training_history,
            'model_config': {
                'n_linguistic_features': self.model.linguistic_features_dim,
                'model_name': self.model_name
            }
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Initialize model with saved config
        model_config_saved = checkpoint['model_config']
        self.initialize_model(model_config_saved['n_linguistic_features'])
        
        # Load state dict
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.tokenizer = checkpoint['tokenizer']
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
        plt.title('Confusion Matrix - BERT Text Classifier')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


class SimpleTextClassifier:
    """
    Simple baseline text classifier for comparison.
    """
    
    def __init__(self, method: str = 'naive_bayes'):
        """
        Initialize simple classifier.
        
        Args:
            method: 'naive_bayes', 'svm', or 'random_forest'
        """
        self.method = method
        self.vectorizer = None
        self.classifier = None
        self.text_processor = TextPreprocessor()
    
    def train(self, train_texts: List[str], train_labels: List[int]) -> Dict:
        """Train the simple classifier."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.svm import SVC
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline
        
        # Create vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Choose classifier
        if self.method == 'naive_bayes':
            self.classifier = MultinomialNB()
        elif self.method == 'svm':
            self.classifier = SVC(probability=True, random_state=42)
        elif self.method == 'random_forest':
            self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Create pipeline
        pipeline = Pipeline([
            ('vectorizer', self.vectorizer),
            ('classifier', self.classifier)
        ])
        
        # Train
        pipeline.fit(train_texts, train_labels)
        self.pipeline = pipeline
        
        return {'method': self.method, 'trained': True}
    
    def predict(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions."""
        if self.pipeline is None:
            raise ValueError("Model not trained")
        
        predictions = self.pipeline.predict(texts)
        probabilities = self.pipeline.predict_proba(texts)
        
        return predictions, probabilities


# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    sample_texts = [
        "Scientists at Harvard University published a peer-reviewed study showing climate change effects.",
        "BREAKING: SHOCKING discovery that doctors don't want you to know! Click here for the truth!",
        "The Federal Reserve announced a 0.25% interest rate change following today's meeting.",
        "You won't BELIEVE what this celebrity did! Number 7 will SHOCK you completely!"
    ]
    
    sample_labels = [0, 1, 0, 1]  # 0 = real, 1 = fake
    
    # Test simple classifier
    print("Testing Simple Text Classifier:")
    simple_classifier = SimpleTextClassifier('naive_bayes')
    simple_classifier.train(sample_texts, sample_labels)
    predictions, probabilities = simple_classifier.predict(sample_texts)
    
    print("Simple classifier predictions:")
    for i, (text, pred, prob) in enumerate(zip(sample_texts, predictions, probabilities)):
        print(f"Text {i+1}: {'Fake' if pred == 1 else 'Real'} (confidence: {prob[pred]:.3f})")
    
    # Test BERT classifier (commented out due to computational requirements)
    # print("\nTesting BERT Text Classifier:")
    # bert_classifier = FakeNewsTextClassifier()
    # # Note: In practice, you would need larger datasets for training
    # history = bert_classifier.train(sample_texts, sample_labels, sample_texts, sample_labels, epochs=1)
    # predictions, probabilities = bert_classifier.predict(sample_texts)
