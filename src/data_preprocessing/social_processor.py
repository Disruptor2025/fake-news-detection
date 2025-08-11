"""
Social media processing module for fake news detection.

This module handles social media data preprocessing, user credibility analysis,
and engagement pattern extraction for the multi-modal fake news detection system.

Author: Idrees Khan
Course: AAI 501 - Introduction to Artificial Intelligence
Date: August 2025
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from datetime import datetime, timedelta
import networkx as nx
from collections import defaultdict
import re

from ..utils.config import feature_config


class SocialMediaProcessor:
    """
    Comprehensive social media data processing for fake news detection.
    
    This class analyzes user behavior patterns, engagement metrics, and
    network characteristics to identify potential misinformation propagation.
    """
    
    def __init__(self):
        """Initialize the social media processor."""
        # Bot detection patterns
        self.bot_indicators = {
            'username_patterns': [
                r'^[a-zA-Z]+\d{4,}$',  # Name followed by many digits
                r'^.{1,3}$',           # Very short usernames
                r'[0-9]{6,}',          # Many consecutive digits
                r'[a-zA-Z]{20,}'       # Extremely long usernames
            ],
            'bio_patterns': [
                r'follow\s*back',      # Follow back requests
                r'dm\s*for',           # DM for something
                r'link\s*in\s*bio',    # Link in bio
                r'click\s*link'        # Click link requests
            ]
        }
        
        # Credibility indicators
        self.credibility_factors = {
            'positive': ['verified', 'journalist', 'news', 'reporter', 'official'],
            'negative': ['fake', 'spam', 'bot', 'troll', 'sock']
        }
        
        # Compile regex patterns for efficiency
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for social media analysis."""
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.mention_pattern = re.compile(r'@[A-Za-z0-9_]+')
        self.hashtag_pattern = re.compile(r'#[A-Za-z0-9_]+')
        self.emoji_pattern = re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]')
    
    def calculate_user_credibility(self, user_data: Dict[str, any]) -> float:
        """
        Calculate user credibility score based on profile information.
        
        Args:
            user_data: Dictionary containing user profile information
            
        Returns:
            Float credibility score between 0 and 1
        """
        credibility_score = 0.5  # Start with neutral score
        
        # Account verification status
        if user_data.get('verified', False):
            credibility_score += 0.3
        
        # Account age (older accounts are generally more credible)
        account_age_days = user_data.get('account_age_days', 0)
        if account_age_days > 365:  # More than 1 year
            credibility_score += 0.1
        elif account_age_days > 30:  # More than 1 month
            credibility_score += 0.05
        elif account_age_days < 7:  # Very new account
            credibility_score -= 0.2
        
        # Follower to following ratio
        followers = user_data.get('follower_count', 0)
        following = user_data.get('following_count', 1)  # Avoid division by zero
        follower_ratio = followers / following if following > 0 else 0
        
        if follower_ratio > 2:  # More followers than following
            credibility_score += 0.1
        elif follower_ratio < 0.1:  # Following many more than followers
            credibility_score -= 0.1
        
        # Bio/description analysis
        bio = user_data.get('bio', '').lower()
        for positive_term in self.credibility_factors['positive']:
            if positive_term in bio:
                credibility_score += 0.05
        
        for negative_term in self.credibility_factors['negative']:
            if negative_term in bio:
                credibility_score -= 0.1
        
        # Profile completeness
        profile_fields = ['bio', 'location', 'website', 'profile_image']
        completeness = sum(1 for field in profile_fields if user_data.get(field))
        credibility_score += (completeness / len(profile_fields)) * 0.1
        
        # Posting frequency (moderate posting is better)
        posts_count = user_data.get('posts_count', 0)
        if account_age_days > 0:
            posts_per_day = posts_count / account_age_days
            if 0.1 <= posts_per_day <= 5:  # Reasonable posting frequency
                credibility_score += 0.05
            elif posts_per_day > 50:  # Excessive posting
                credibility_score -= 0.2
        
        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, credibility_score))
    
    def detect_bot_probability(self, user_data: Dict[str, any], 
                              activity_data: List[Dict[str, any]] = None) -> float:
        """
        Calculate probability that a user is a bot.
        
        Args:
            user_data: User profile information
            activity_data: List of user's recent activities (optional)
            
        Returns:
            Float bot probability between 0 and 1
        """
        bot_score = 0.0
        
        # Username analysis
        username = user_data.get('username', '')
        for pattern in self.bot_indicators['username_patterns']:
            if re.search(pattern, username):
                bot_score += 0.2
                break
        
        # Bio analysis
        bio = user_data.get('bio', '').lower()
        for pattern in self.bot_indicators['bio_patterns']:
            if re.search(pattern, bio):
                bot_score += 0.1
        
        # Profile image analysis
        if not user_data.get('profile_image') or user_data.get('default_profile_image', False):
            bot_score += 0.1
        
        # Account creation and activity patterns
        account_age_days = user_data.get('account_age_days', 0)
        posts_count = user_data.get('posts_count', 0)
        
        if account_age_days > 0:
            posts_per_day = posts_count / account_age_days
            
            # Very high posting frequency suggests bot
            if posts_per_day > 100:
                bot_score += 0.3
            elif posts_per_day > 50:
                bot_score += 0.2
        
        # Follower patterns
        followers = user_data.get('follower_count', 0)
        following = user_data.get('following_count', 0)
        
        # Bots often have unusual follower patterns
        if followers == 0 and following > 100:
            bot_score += 0.2
        elif followers > 0 and following / followers > 10:
            bot_score += 0.1
        
        # Activity pattern analysis (if available)
        if activity_data:
            bot_score += self._analyze_activity_patterns(activity_data)
        
        return min(1.0, bot_score)
    
    def _analyze_activity_patterns(self, activity_data: List[Dict[str, any]]) -> float:
        """Analyze activity patterns for bot-like behavior."""
        if not activity_data:
            return 0.0
        
        bot_indicators = 0.0
        
        # Time pattern analysis
        timestamps = []
        for activity in activity_data:
            if 'timestamp' in activity:
                try:
                    if isinstance(activity['timestamp'], str):
                        timestamp = datetime.fromisoformat(activity['timestamp'].replace('Z', '+00:00'))
                    else:
                        timestamp = activity['timestamp']
                    timestamps.append(timestamp)
                except:
                    continue
        
        if len(timestamps) > 5:
            # Check for unnaturally regular posting intervals
            intervals = []
            for i in range(1, len(timestamps)):
                interval = (timestamps[i] - timestamps[i-1]).total_seconds()
                intervals.append(interval)
            
            # Very regular intervals suggest automation
            if len(intervals) > 2:
                interval_std = np.std(intervals)
                interval_mean = np.mean(intervals)
                if interval_std / (interval_mean + 1) < 0.1:  # Very consistent timing
                    bot_indicators += 0.2
        
        # Content repetition analysis
        content_texts = [activity.get('content', '') for activity in activity_data]
        unique_content = len(set(content_texts))
        total_content = len(content_texts)
        
        if total_content > 0:
            content_diversity = unique_content / total_content
            if content_diversity < 0.3:  # Very repetitive content
                bot_indicators += 0.2
        
        return min(0.5, bot_indicators)  # Cap contribution from activity patterns
    
    def analyze_engagement_patterns(self, post_data: Dict[str, any]) -> Dict[str, float]:
        """
        Analyze engagement patterns for a specific post.
        
        Args:
            post_data: Dictionary containing post information and engagement metrics
            
        Returns:
            Dictionary containing engagement analysis features
        """
        engagement_features = {}
        
        # Basic engagement metrics
        likes = post_data.get('likes', 0)
        shares = post_data.get('shares', 0)
        comments = post_data.get('comments', 0)
        views = post_data.get('views', 1)  # Avoid division by zero
        
        # Engagement rates
        engagement_features['like_rate'] = likes / views
        engagement_features['share_rate'] = shares / views
        engagement_features['comment_rate'] = comments / views
        engagement_features['total_engagement_rate'] = (likes + shares + comments) / views
        
        # Engagement ratios
        total_engagement = likes + shares + comments
        if total_engagement > 0:
            engagement_features['like_ratio'] = likes / total_engagement
            engagement_features['share_ratio'] = shares / total_engagement
            engagement_features['comment_ratio'] = comments / total_engagement
        else:
            engagement_features['like_ratio'] = 0
            engagement_features['share_ratio'] = 0
            engagement_features['comment_ratio'] = 0
        
        # Engagement velocity (if timestamp data available)
        post_time = post_data.get('timestamp')
        if post_time and total_engagement > 0:
            if isinstance(post_time, str):
                post_time = datetime.fromisoformat(post_time.replace('Z', '+00:00'))
            
            time_since_post = (datetime.now() - post_time).total_seconds() / 3600  # Hours
            if time_since_post > 0:
                engagement_features['engagement_velocity'] = total_engagement / time_since_post
            else:
                engagement_features['engagement_velocity'] = 0
        else:
            engagement_features['engagement_velocity'] = 0
        
        # Suspicious engagement patterns
        # High share-to-like ratio might indicate coordinated sharing
        if likes > 0:
            engagement_features['share_to_like_ratio'] = shares / likes
        else:
            engagement_features['share_to_like_ratio'] = 0
        
        # Very high engagement rate on controversial content
        content = post_data.get('content', '').lower()
        controversial_keywords = ['breaking', 'shocking', 'exposed', 'scandal', 'urgent']
        controversy_score = sum(1 for keyword in controversial_keywords if keyword in content)
        engagement_features['controversy_engagement_score'] = controversy_score * engagement_features['total_engagement_rate']
        
        return engagement_features
    
    def analyze_sharing_network(self, sharing_data: List[Dict[str, any]]) -> Dict[str, float]:
        """
        Analyze network characteristics of content sharing.
        
        Args:
            sharing_data: List of sharing events with user information
            
        Returns:
            Dictionary containing network analysis features
        """
        network_features = {}
        
        if not sharing_data:
            return {
                'network_density': 0,
                'clustering_coefficient': 0,
                'centrality_variance': 0,
                'bot_share_ratio': 0,
                'rapid_cascade_score': 0
            }
        
        # Build sharing network
        G = nx.DiGraph()
        
        # Add edges for sharing relationships
        for share in sharing_data:
            sharer_id = share.get('user_id')
            original_author = share.get('original_author_id')
            if sharer_id and original_author:
                G.add_edge(original_author, sharer_id)
        
        # Calculate network metrics
        if G.number_of_nodes() > 1:
            # Network density
            network_features['network_density'] = nx.density(G)
            
            # Clustering coefficient
            try:
                clustering = nx.clustering(G.to_undirected())
                network_features['clustering_coefficient'] = np.mean(list(clustering.values()))
            except:
                network_features['clustering_coefficient'] = 0
            
            # Centrality analysis
            try:
                centrality = nx.betweenness_centrality(G)
                centrality_values = list(centrality.values())
                if centrality_values:
                    network_features['centrality_variance'] = np.var(centrality_values)
                else:
                    network_features['centrality_variance'] = 0
            except:
                network_features['centrality_variance'] = 0
        else:
            network_features['network_density'] = 0
            network_features['clustering_coefficient'] = 0
            network_features['centrality_variance'] = 0
        
        # Bot analysis in sharing network
        bot_shares = sum(1 for share in sharing_data if share.get('is_bot', False))
        network_features['bot_share_ratio'] = bot_shares / len(sharing_data) if sharing_data else 0
        
        # Rapid cascade detection
        timestamps = []
        for share in sharing_data:
            if 'timestamp' in share:
                try:
                    if isinstance(share['timestamp'], str):
                        timestamp = datetime.fromisoformat(share['timestamp'].replace('Z', '+00:00'))
                    else:
                        timestamp = share['timestamp']
                    timestamps.append(timestamp)
                except:
                    continue
        
        if len(timestamps) > 2:
            timestamps.sort()
            # Calculate time to reach certain share thresholds
            rapid_shares = 0
            for i in range(1, min(len(timestamps), 11)):  # First 10 shares
                time_diff = (timestamps[i] - timestamps[0]).total_seconds() / 60  # Minutes
                if time_diff < 60:  # Shared within first hour
                    rapid_shares += 1
            
            network_features['rapid_cascade_score'] = rapid_shares / min(len(timestamps), 10)
        else:
            network_features['rapid_cascade_score'] = 0
        
        return network_features
    
    def extract_content_features(self, content: str) -> Dict[str, float]:
        """
        Extract features from social media content text.
        
        Args:
            content: Social media post content
            
        Returns:
            Dictionary containing content analysis features
        """
        content_features = {}
        
        if not content:
            return {feature: 0.0 for feature in [
                'url_count', 'mention_count', 'hashtag_count', 'emoji_count',
                'caps_ratio', 'urgency_score', 'clickbait_score'
            ]}
        
        # Count different elements
        content_features['url_count'] = len(self.url_pattern.findall(content))
        content_features['mention_count'] = len(self.mention_pattern.findall(content))
        content_features['hashtag_count'] = len(self.hashtag_pattern.findall(content))
        content_features['emoji_count'] = len(self.emoji_pattern.findall(content))
        
        # Text characteristics
        if len(content) > 0:
            caps_count = sum(1 for c in content if c.isupper())
            content_features['caps_ratio'] = caps_count / len(content)
        else:
            content_features['caps_ratio'] = 0
        
        # Urgency indicators
        urgency_words = ['urgent', 'breaking', 'now', 'immediately', 'emergency', 'alert']
        urgency_count = sum(1 for word in urgency_words if word in content.lower())
        content_features['urgency_score'] = urgency_count / len(content.split()) if content.split() else 0
        
        # Clickbait indicators
        clickbait_phrases = [
            'you won\'t believe', 'shocking', 'this will change', 'doctors hate',
            'one simple trick', 'what happens next', 'number 7 will shock'
        ]
        clickbait_count = sum(1 for phrase in clickbait_phrases if phrase in content.lower())
        content_features['clickbait_score'] = clickbait_count
        
        return content_features
    
    def calculate_influence_score(self, user_data: Dict[str, any], 
                                 interaction_data: List[Dict[str, any]] = None) -> float:
        """
        Calculate user influence score based on reach and engagement.
        
        Args:
            user_data: User profile information
            interaction_data: Recent interactions and mentions
            
        Returns:
            Float influence score
        """
        influence_score = 0.0
        
        # Follower-based influence
        followers = user_data.get('follower_count', 0)
        if followers > 0:
            # Logarithmic scaling for follower count
            influence_score += min(np.log10(followers) / 6, 1.0)  # Max 1 point for 1M followers
        
        # Engagement-based influence
        if interaction_data:
            total_interactions = 0
            total_reach = 0
            
            for interaction in interaction_data:
                likes = interaction.get('likes', 0)
                shares = interaction.get('shares', 0)
                comments = interaction.get('comments', 0)
                views = interaction.get('views', 0)
                
                total_interactions += likes + shares + comments
                total_reach += views
            
            if total_reach > 0:
                engagement_rate = total_interactions / total_reach
                influence_score += min(engagement_rate * 2, 1.0)  # Max 1 point for 50% engagement
        
        # Verification bonus
        if user_data.get('verified', False):
            influence_score += 0.5
        
        # Account age factor (more established accounts have more influence)
        account_age_days = user_data.get('account_age_days', 0)
        if account_age_days > 365:
            age_factor = min(account_age_days / 1825, 1.0)  # Max factor at 5 years
            influence_score *= (1 + age_factor * 0.5)
        
        return min(influence_score, 5.0)  # Cap at 5.0
    
    def process_social_data(self, user_data: Dict[str, any], 
                           post_data: Dict[str, any] = None,
                           sharing_data: List[Dict[str, any]] = None,
                           activity_data: List[Dict[str, any]] = None) -> Dict[str, any]:
        """
        Complete social media processing pipeline.
        
        Args:
            user_data: User profile information
            post_data: Specific post information (optional)
            sharing_data: Sharing network data (optional)
            activity_data: User activity history (optional)
            
        Returns:
            Dictionary containing all processed social media features
        """
        processed_features = {}
        
        # User credibility analysis
        processed_features['user_credibility_score'] = self.calculate_user_credibility(user_data)
        
        # Bot detection
        processed_features['bot_probability'] = self.detect_bot_probability(user_data, activity_data)
        
        # Influence score
        processed_features['influence_score'] = self.calculate_influence_score(user_data, activity_data)
        
        # Basic user features
        processed_features['follower_count'] = user_data.get('follower_count', 0)
        processed_features['following_count'] = user_data.get('following_count', 0)
        processed_features['posts_count'] = user_data.get('posts_count', 0)
        processed_features['account_age_days'] = user_data.get('account_age_days', 0)
        processed_features['verified_status'] = 1 if user_data.get('verified', False) else 0
        
        # Calculate follower ratios
        followers = processed_features['follower_count']
        following = processed_features['following_count']
        if following > 0:
            processed_features['follower_following_ratio'] = followers / following
        else:
            processed_features['follower_following_ratio'] = followers  # If not following anyone
        
        # Post-specific analysis
        if post_data:
            engagement_features = self.analyze_engagement_patterns(post_data)
            processed_features.update(engagement_features)
            
            content_features = self.extract_content_features(post_data.get('content', ''))
            processed_features.update(content_features)
        else:
            # Default values for missing post data
            default_engagement = {
                'like_rate': 0, 'share_rate': 0, 'comment_rate': 0,
                'total_engagement_rate': 0, 'engagement_velocity': 0,
                'share_to_like_ratio': 0, 'controversy_engagement_score': 0
            }
            processed_features.update(default_engagement)
            
            default_content = {
                'url_count': 0, 'mention_count': 0, 'hashtag_count': 0,
                'emoji_count': 0, 'caps_ratio': 0, 'urgency_score': 0,
                'clickbait_score': 0
            }
            processed_features.update(default_content)
        
        # Network analysis
        if sharing_data:
            network_features = self.analyze_sharing_network(sharing_data)
            processed_features.update(network_features)
        else:
            # Default network features
            default_network = {
                'network_density': 0, 'clustering_coefficient': 0,
                'centrality_variance': 0, 'bot_share_ratio': 0,
                'rapid_cascade_score': 0
            }
            processed_features.update(default_network)
        
        return processed_features
    
    def batch_process(self, social_data_list: List[Dict[str, any]], 
                     show_progress: bool = True) -> List[Dict[str, any]]:
        """
        Process multiple social media data entries in batch.
        
        Args:
            social_data_list: List of social media data dictionaries
            show_progress: Whether to show progress bar
            
        Returns:
            List of processed social media features
        """
        from tqdm import tqdm
        
        results = []
        iterator = tqdm(social_data_list, desc="Processing social data") if show_progress else social_data_list
        
        for social_data in iterator:
            try:
                processed = self.process_social_data(
                    user_data=social_data.get('user_data', {}),
                    post_data=social_data.get('post_data'),
                    sharing_data=social_data.get('sharing_data'),
                    activity_data=social_data.get('activity_data')
                )
                results.append(processed)
            except Exception as e:
                print(f"Error processing social data: {e}")
                # Add empty result to maintain alignment
                results.append(self._get_empty_social_features())
        
        return results
    
    def _get_empty_social_features(self) -> Dict[str, float]:
        """Return empty feature dictionary for failed social media processing."""
        return {feature: 0.0 for feature in feature_config.SOCIAL_FEATURES}


class SyntheticSocialDataGenerator:
    """
    Generate synthetic social media data for testing and augmentation.
    """
    
    def __init__(self):
        """Initialize synthetic data generator."""
        self.fake_names = [
            'NewsBreaker123', 'TruthSeeker99', 'InfoWarrior2025', 'FactFinder88',
            'RealNewsNow', 'BreakingAlert', 'TruthTeller24', 'InfoSource'
        ]
        
        self.fake_bios = [
            'Bringing you the REAL news they don\'t want you to see!',
            'Independent journalist exposing the truth',
            'Follow for breaking news and exclusive content',
            'Click link below for shocking revelations',
            'Patriot fighting for freedom and truth'
        ]
    
    def generate_fake_user(self, credibility_level: str = 'low') -> Dict[str, any]:
        """
        Generate synthetic user data with specified credibility level.
        
        Args:
            credibility_level: 'low', 'medium', or 'high'
            
        Returns:
            Dictionary containing synthetic user data
        """
        import random
        
        if credibility_level == 'low':
            # Suspicious user characteristics
            user_data = {
                'username': random.choice(self.fake_names) + str(random.randint(1000, 9999)),
                'bio': random.choice(self.fake_bios),
                'verified': False,
                'follower_count': random.randint(10, 500),
                'following_count': random.randint(2000, 5000),
                'posts_count': random.randint(100, 1000),
                'account_age_days': random.randint(1, 30),  # Very new account
                'profile_image': False,
                'location': '',
                'website': ''
            }
        elif credibility_level == 'medium':
            # Moderately credible user
            user_data = {
                'username': f"User{random.randint(100, 999)}",
                'bio': 'Regular user sharing interesting content',
                'verified': False,
                'follower_count': random.randint(100, 2000),
                'following_count': random.randint(200, 1000),
                'posts_count': random.randint(50, 500),
                'account_age_days': random.randint(90, 1000),
                'profile_image': True,
                'location': 'City, State',
                'website': ''
            }
        else:  # high credibility
            # Credible user characteristics
            user_data = {
                'username': f"Journalist{random.randint(10, 99)}",
                'bio': 'Professional journalist reporting on current events',
                'verified': random.choice([True, False]),
                'follower_count': random.randint(5000, 50000),
                'following_count': random.randint(100, 1000),
                'posts_count': random.randint(200, 2000),
                'account_age_days': random.randint(365, 3000),
                'profile_image': True,
                'location': 'Major City',
                'website': 'news-website.com'
            }
        
        return user_data
    
    def generate_engagement_data(self, virality_level: str = 'normal') -> Dict[str, any]:
        """
        Generate synthetic engagement data.
        
        Args:
            virality_level: 'low', 'normal', 'viral', or 'suspicious'
            
        Returns:
            Dictionary containing synthetic engagement metrics
        """
        import random
        from datetime import datetime, timedelta
        
        base_views = {
            'low': random.randint(50, 200),
            'normal': random.randint(200, 2000),
            'viral': random.randint(10000, 100000),
            'suspicious': random.randint(50000, 500000)
        }
        
        views = base_views[virality_level]
        
        if virality_level == 'suspicious':
            # Unusual engagement patterns for suspicious content
            likes = int(views * random.uniform(0.1, 0.3))
            shares = int(views * random.uniform(0.05, 0.2))  # High share rate
            comments = int(views * random.uniform(0.001, 0.01))
        else:
            # Normal engagement patterns
            likes = int(views * random.uniform(0.01, 0.1))
            shares = int(views * random.uniform(0.001, 0.02))
            comments = int(views * random.uniform(0.001, 0.005))
        
        timestamp = datetime.now() - timedelta(hours=random.randint(1, 24))
        
        return {
            'views': views,
            'likes': likes,
            'shares': shares,
            'comments': comments,
            'timestamp': timestamp
        }


def create_social_features_dataframe(processed_social: List[Dict[str, any]]) -> pd.DataFrame:
    """
    Create a pandas DataFrame from processed social media features.
    
    Args:
        processed_social: List of processed social media data
        
    Returns:
        DataFrame with social media features
    """
    # Ensure all feature names are present in each dictionary
    all_features = set()
    for data in processed_social:
        all_features.update(data.keys())
    
    # Fill missing features with zeros
    normalized_data = []
    for data in processed_social:
        normalized_row = {feature: data.get(feature, 0.0) for feature in all_features}
        normalized_data.append(normalized_row)
    
    return pd.DataFrame(normalized_data)


# Example usage and testing
if __name__ == "__main__":
    # Initialize processor
    processor = SocialMediaProcessor()
    
    # Example user data
    sample_user = {
        'username': 'NewsBreaker2024',
        'bio': 'BREAKING news that they don\'t want you to see! Follow for truth!',
        'verified': False,
        'follower_count': 150,
        'following_count': 3000,
        'posts_count': 500,
        'account_age_days': 15,
        'profile_image': False
    }
    
    # Example post data
    sample_post = {
        'content': 'BREAKING: SHOCKING discovery that will change everything! You won\'t believe what scientists found! #Breaking #Truth',
        'likes': 100,
        'shares': 50,
        'comments': 5,
        'views': 1000,
        'timestamp': datetime.now()
    }
    
    # Process the data
    result = processor.process_social_data(sample_user, sample_post)
    
    print("Social Media Processing Results:")
    print(f"User credibility score: {result['user_credibility_score']:.3f}")
    print(f"Bot probability: {result['bot_probability']:.3f}")
    print(f"Influence score: {result['influence_score']:.3f}")
    print(f"Total engagement rate: {result['total_engagement_rate']:.3f}")
    print(f"Clickbait score: {result['clickbait_score']:.3f}")
    
    # Generate synthetic data
    generator = SyntheticSocialDataGenerator()
    fake_user = generator.generate_fake_user('low')
    print(f"\nSynthetic user credibility: {processor.calculate_user_credibility(fake_user):.3f}")
