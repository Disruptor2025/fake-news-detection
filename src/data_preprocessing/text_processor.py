"""
Text processing module for fake news detection.

This module handles text preprocessing, cleaning, and feature extraction
for the multi-modal fake news detection system.

Author: Idrees Khan
Course: AAI 501 - Introduction to Artificial Intelligence
Date: August 2025
"""

import re
import string
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import nltk
import spacy
import textstat
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import BertTokenizer, BertModel
import torch

from ..utils.config import model_config, feature_config


class TextPreprocessor:
    """
    Comprehensive text preprocessing for fake news detection.
    
    This class provides methods for cleaning, normalizing, and extracting
    features from text data for fake news detection.
    """
    
    def __init__(self):
        """Initialize the text preprocessor with required models and tools."""
        # Download required NLTK data
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('vader_lexicon', quiet=True)
        nltk.download('wordnet', quiet=True)
        
        # Initialize sentiment analyzer
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Initialize spaCy model for NLP tasks
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy model 'en_core_web_sm' not found. "
                  "Some features may not work. Install with: "
                  "python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Initialize BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_config.BERT_MODEL_NAME)
        
        # Compile regex patterns for efficiency
        self._compile_patterns()
        
        # Emotional words dictionary (simplified version)
        self.emotional_words = {
            'positive': ['amazing', 'incredible', 'fantastic', 'wonderful', 
                        'excellent', 'outstanding', '
