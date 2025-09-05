import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.pipeline import Pipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Advanced feature engineering pipeline"""
    
    def __init__(self, create_interactions=True, create_polynomials=True):
        self.create_interactions = create_interactions
        self.create_polynomials = create_polynomials
        self.feature_names = None
        
    def fit(self, X, y=None):
        self.feature_names = X.columns.tolist()
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        
        # 1. Create interaction features
        if self.create_interactions:
            X_transformed = self._create_interaction_features(X_transformed)
        
        # 2. Create polynomial features
        if self.create_polynomials:
            X_transformed = self._create_polynomial_features(X_transformed)
        
        # 3. Create ratio features
        X_transformed = self._create_ratio_features(X_transformed)
        
        # 4. Create statistical features
        X_transformed = self._create_statistical_features(X_transformed)
        
        logger.info(f"Feature engineering completed. Original features: {len(self.feature_names)}, "
                   f"New features: {len(X_transformed.columns) - len(self.feature_names)}")
        
        return X_transformed
    
    def _create_interaction_features(self, X):
        """Create interaction features between highly correlated features"""
        # Select numeric features for interactions
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        
        # Create interactions for top features (you can customize this)
        interaction_pairs = [
            ('Energy', 'RhythmScore'),
            ('AudioLoudness', 'Energy'),
            ('MoodScore', 'Energy')
        ]
        
        for feat1, feat2 in interaction_pairs:
            if feat1 in numeric_features and feat2 in numeric_features:
                interaction_name = f"{feat1}_{feat2}_interaction"
                X[interaction_name] = X[feat1] * X[feat2]
        
        return X
    
    def _create_polynomial_features(self, X):
        """Create polynomial features for important features"""
        important_features = ['Energy', 'AudioLoudness', 'RhythmScore', 'MoodScore']
        
        for feature in important_features:
            if feature in X.columns:
                X[f"{feature}_squared"] = X[feature] ** 2
                X[f"{feature}_cubed"] = X[feature] ** 3
        
        return X
    
    def _create_ratio_features(self, X):
        """Create ratio features"""
        ratio_pairs = [
            ('Energy', 'AudioLoudness'),
            ('RhythmScore', 'MoodScore'),
            ('VocalContent', 'InstrumentalScore')
        ]
        
        for numerator, denominator in ratio_pairs:
            if numerator in X.columns and denominator in X.columns:
                # Avoid division by zero
                X[f"{numerator}_to_{denominator}"] = X[numerator] / (X[denominator] + 1e-8)
        
        return X
    
    def _create_statistical_features(self, X):
        """Create statistical features"""
        # Row-wise statistics
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X['mean_features'] = X[numeric_cols].mean(axis=1)
        X['std_features'] = X[numeric_cols].std(axis=1)
        X['max_features'] = X[numeric_cols].max(axis=1)
        X['min_features'] = X[numeric_cols].min(axis=1)
        
        return X

class DataPreprocessor(BaseEstimator, TransformerMixin):
    """Data preprocessing pipeline"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.transformer = PowerTransformer(method='yeo-johnson')
        self.numeric_features = None
        
    def fit(self, X, y=None):
        self.numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        self.scaler.fit(X[self.numeric_features])
        self.transformer.fit(X[self.numeric_features])
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        
        # Scale numeric features
        X_transformed[self.numeric_features] = self.scaler.transform(X_transformed[self.numeric_features])
        
        # Apply power transformation to reduce skewness
        X_transformed[self.numeric_features] = self.transformer.transform(X_transformed[self.numeric_features])
        
        return X_transformed

def create_feature_pipeline():
    """Create complete feature engineering pipeline"""
    return Pipeline([
        ('feature_engineering', FeatureEngineer()),
        ('preprocessing', DataPreprocessor())
    ])