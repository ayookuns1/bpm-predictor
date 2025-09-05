import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging
from typing import Dict, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Advanced model training and evaluation class"""
    
    def __init__(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
        self.X = X
        self.y = y
        self.test_size = test_size
        self.random_state = random_state
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        self.models = {}
        self.best_model = None
        
    def train_models(self):
        """Train multiple models with hyperparameter tuning"""
        
        # Define models and their parameter grids
        models_config = {
            'random_forest': {
                'model': RandomForestRegressor(random_state=self.random_state),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5]
                }
            },
            'xgboost': {
                'model': XGBRegressor(random_state=self.random_state),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2]
                }
            },
            'lightgbm': {
                'model': LGBMRegressor(random_state=self.random_state),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2]
                }
            },
            'ridge': {
                'model': Ridge(random_state=self.random_state),
                'params': {
                    'alpha': [0.1, 1.0, 10.0]
                }
            }
        }
        
        best_score = float('inf')
        
        for name, config in models_config.items():
            logger.info(f"Training {name}...")
            
            # Perform grid search
            grid_search = GridSearchCV(
                config['model'], 
                config['params'], 
                cv=5, 
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(self.X_train, self.y_train)
            
            # Store the best model
            self.models[name] = {
                'model': grid_search.best_estimator_,
                'best_params': grid_search.best_params_,
                'best_score': -grid_search.best_score_,
                'cv_results': grid_search.cv_results_
            }
            
            logger.info(f"Best parameters for {name}: {grid_search.best_params_}")
            logger.info(f"Best CV score for {name}: {-grid_search.best_score_:.4f}")
            
            # Update best model
            if -grid_search.best_score_ < best_score:
                best_score = -grid_search.best_score_
                self.best_model = grid_search.best_estimator_
        
        return self.models
    
    def evaluate_models(self) -> Dict:
        """Evaluate all trained models on test set"""
        evaluation_results = {}
        
        for name, model_info in self.models.items():
            model = model_info['model']
            y_pred = model.predict(self.X_test)
            
            metrics = {
                'mae': mean_absolute_error(self.y_test, y_pred),
                'mse': mean_squared_error(self.y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(self.y_test, y_pred)),
                'r2': r2_score(self.y_test, y_pred)
            }
            
            evaluation_results[name] = metrics
            logger.info(f"{name} Test Metrics: {metrics}")
        
        return evaluation_results
    
    def plot_feature_importance(self, model_name: str = 'xgboost'):
        """Plot feature importance for the specified model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found in trained models")
        
        model = self.models[model_name]['model']
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_names = self.X.columns
            
            # Create feature importance dataframe
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            # Plot
            plt.figure(figsize=(12, 8))
            sns.barplot(x='importance', y='feature', data=importance_df.head(20))
            plt.title(f'Feature Importance - {model_name}')
            plt.tight_layout()
            plt.savefig(f'../reports/figures/feature_importance_{model_name}.png', 
                       dpi=300, bbox_inches='tight')
            plt.show()
            
            return importance_df
        
        else:
            logger.warning(f"Model {model_name} doesn't have feature_importances_ attribute")
            return None
    
    def save_best_model(self, filepath: str):
        """Save the best model to disk"""
        if self.best_model is None:
            raise ValueError("No model has been trained yet")
        
        joblib.dump(self.best_model, filepath)
        logger.info(f"Best model saved to {filepath}")
    
    def create_performance_report(self, evaluation_results: Dict) -> pd.DataFrame:
        """Create a comprehensive performance report"""
        report_data = []
        
        for model_name, metrics in evaluation_results.items():
            report_data.append({
                'Model': model_name,
                'MAE': metrics['mae'],
                'MSE': metrics['mse'],
                'RMSE': metrics['rmse'],
                'R²': metrics['r2']
            })
        
        return pd.DataFrame(report_data)

def train_complete_pipeline(X: pd.DataFrame, y: pd.Series) -> Tuple[ModelTrainer, Dict]:
    """Complete training pipeline"""
    # Initialize trainer
    trainer = ModelTrainer(X, y)
    
    # Train models
    models = trainer.train_models()
    
    # Evaluate models
    evaluation_results = trainer.evaluate_models()
    
    # Plot feature importance for best model
    trainer.plot_feature_importance('xgboost')
    
    # Save best model
    trainer.save_best_model('../models/best_model.pkl')
    
    # Create performance report
    performance_report = trainer.create_performance_report(evaluation_results)
    performance_report.to_csv('../reports/model_performance.csv', index=False)
    
    return trainer, evaluation_results