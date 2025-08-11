"""
Configuration settings for the Multi-Modal Fake News Detection System.

This module contains all configuration parameters, file paths, and model
settings used throughout the project.

Author: Idrees Khan
Course: AAI 501 - Introduction to Artificial Intelligence
Date: August 2025
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"

# Model directories
MODELS_DIR = BASE_DIR / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
SAVED_MODELS_DIR = MODELS_DIR / "saved_models"

# Results and outputs
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "performance_metrics"

# Web application
WEB_APP_DIR = BASE_DIR / "web_app"
TEMPLATES_DIR = WEB_APP_DIR / "templates"
STATIC_DIR = WEB_APP_DIR / "static"

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, SYNTHETIC_DATA_DIR,
                  MODELS_DIR, CHECKPOINTS_DIR, SAVED_MODELS_DIR, RESULTS_DIR,
                  FIGURES_DIR, METRICS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


class ModelConfig:
    """Configuration for machine learning models."""
    
    # Text Model Configuration
    BERT_MODEL_NAME = "bert-base-uncased"
    MAX_SEQUENCE_LENGTH = 512
    TEXT_EMBEDDING_DIM = 768
    
    # Image Model Configuration
    IMAGE_MODEL_NAME = "resnet50"
    IMAGE_SIZE = (224, 224)
    IMAGE_EMBEDDING_DIM = 2048
    
    # Social Media Model Configuration
    SOCIAL_FEATURES_DIM = 25
    
    # Ensemble Model Configuration
    FUSION_HIDDEN_DIMS = [512, 256, 128]
    DROPOUT_RATE = 0.3
    
    # Training Configuration
    BATCH_SIZE = 32
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 10
    EARLY_STOPPING_PATIENCE = 3
    
    # Cross-validation
    N_FOLDS = 5
    RANDOM_STATE = 42


class DataConfig:
    """Configuration for data processing."""
    
    # Dataset URLs and paths
    FAKENEWSNET_URL = "https://github.com/KaiDMML/FakeNewsNet"
    LIAR_DATASET_URL = "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip"
    ISOT_DATASET_URL = "https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset"
    
    # Data splits
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # Text preprocessing
    MIN_ARTICLE_LENGTH = 50
    MAX_ARTICLE_LENGTH = 10000
    
    # Image preprocessing
    SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    MAX_IMAGE_SIZE_MB = 10
    
    # Synthetic data generation
    SYNTHETIC_ARTICLES_PER_TOPIC = 100
    BIAS_TYPES = ['sensational', 'misleading', 'fabricated', 'clickbait']


class FeatureConfig:
    """Configuration for feature extraction."""
    
    # Text features
    TEXT_FEATURES = [
        'sentiment_polarity',
        'sentiment_subjectivity',
        'readability_score',
        'word_count',
        'sentence_count',
        'avg_word_length',
        'caps_ratio',
        'punctuation_ratio',
        'question_marks',
        'exclamation_marks',
        'urls_count',
        'mentions_count',
        'hashtags_count',
        'numbers_count',
        'emotional_words_ratio'
    ]
    
    # Image features
    IMAGE_FEATURES = [
        'manipulation_score',
        'compression_artifacts',
        'metadata_inconsistency',
        'reverse_image_matches',
        'face_detection_score',
        'text_overlay_ratio',
        'brightness_score',
        'contrast_score'
    ]
    
    # Social media features
    SOCIAL_FEATURES = [
        'user_credibility_score',
        'follower_count',
        'following_count',
        'account_age_days',
        'verified_status',
        'posts_count',
        'engagement_rate',
        'retweet_ratio',
        'mention_diversity',
        'posting_frequency',
        'time_between_posts',
        'bot_probability',
        'network_centrality',
        'influence_score',
        'controversial_topics_ratio',
        'fact_check_history',
        'source_diversity',
        'viral_content_ratio',
        'duplicate_content_ratio',
        'external_links_ratio',
        'multimedia_ratio',
        'emotional_manipulation_score',
        'urgency_indicators',
        'authority_claims',
        'social_proof_indicators'
    ]


class WebAppConfig:
    """Configuration for web application."""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    HOST = os.environ.get('FLASK_HOST', '127.0.0.1')
    PORT = int(os.environ.get('FLASK_PORT', 5000))
    
    # Upload settings
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    UPLOAD_FOLDER = STATIC_DIR / 'uploads'
    ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
    
    # API rate limiting
    API_RATE_LIMIT = '100 per hour'
    
    # Model serving
    MODEL_CACHE_SIZE = 3  # Keep 3 models in memory
    PREDICTION_TIMEOUT = 30  # seconds


class ExperimentConfig:
    """Configuration for experiments and evaluation."""
    
    # Evaluation metrics
    CLASSIFICATION_METRICS = [
        'accuracy',
        'precision',
        'recall',
        'f1_score',
        'roc_auc',
        'confusion_matrix'
    ]
    
    # Baseline models for comparison
    BASELINE_MODELS = [
        'naive_bayes',
        'svm_tfidf',
        'random_forest',
        'bert_base',
        'multimodal_ensemble'
    ]
    
    # Hyperparameter tuning
    HYPERPARAMETER_SEARCH = {
        'method': 'bayesian',  # 'grid', 'random', 'bayesian'
        'n_trials': 100,
        'cv_folds': 3
    }
    
    # Explainability
    EXPLAINABILITY_SAMPLES = 100
    SHAP_BACKGROUND_SAMPLES = 50


class LoggingConfig:
    """Configuration for logging."""
    
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_DIR = BASE_DIR / 'logs'
    LOG_DIR.mkdir(exist_ok=True)
    
    # Log files
    MAIN_LOG_FILE = LOG_DIR / 'main.log'
    ERROR_LOG_FILE = LOG_DIR / 'errors.log'
    TRAINING_LOG_FILE = LOG_DIR / 'training.log'


# Environment-specific configurations
class DevelopmentConfig:
    """Development environment configuration."""
    DEBUG = True
    TESTING = False
    DATABASE_URI = 'sqlite:///dev.db'


class TestingConfig:
    """Testing environment configuration."""
    DEBUG = False
    TESTING = True
    DATABASE_URI = 'sqlite:///test.db'


class ProductionConfig:
    """Production environment configuration."""
    DEBUG = False
    TESTING = False
    DATABASE_URI = os.environ.get('DATABASE_URL', 'postgresql://localhost/fakenews')


# Configuration mapping
config_map = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}


def get_config(environment='default'):
    """Get configuration based on environment."""
    return config_map.get(environment, DevelopmentConfig)


# Global configuration instances
model_config = ModelConfig()
data_config = DataConfig()
feature_config = FeatureConfig()
webapp_config = WebAppConfig()
experiment_config = ExperimentConfig()
logging_config = LoggingConfig()
