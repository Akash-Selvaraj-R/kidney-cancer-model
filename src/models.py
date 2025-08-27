"""
Machine learning models for kidney cancer prediction.
Implements classification models: RandomForest, XGBoost, Logistic Regression, and Deep Learning MLP.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLModelPipeline:
    """Machine learning pipeline for kidney cancer classification."""
    
    def __init__(self, random_state=42):
        """Initialize ML pipeline."""
        self.random_state = random_state
        self.models = {}
        self.model_scores = {}
        
    def prepare_data(self, features, labels, test_size=0.2):
        """
        Prepare data for training and testing.
        
        Args:
            features (pd.DataFrame): Feature matrix
            labels (pd.DataFrame): Labels DataFrame
            test_size (float): Test set proportion
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        # Extract features (exclude sample_id)
        X = features.drop('sample_id', axis=1)
        
        # Prepare target variable
        y = labels['cancer_status'].map({'tumor': 1, 'normal': 0})
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        logger.info(f"Class distribution - Train: {y_train.value_counts().to_dict()}")
        logger.info(f"Class distribution - Test: {y_test.value_counts().to_dict()}")
        
        return X_train, X_test, y_train, y_test
    
    def train_random_forest(self, X_train, y_train, hyperparameter_tuning=True):
        """Train Random Forest classifier."""
        logger.info("Training Random Forest...")
        
        if hyperparameter_tuning:
            # Hyperparameter tuning
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            rf = RandomForestClassifier(random_state=self.random_state)
            grid_search = GridSearchCV(
                rf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)
            
            self.models['random_forest'] = grid_search.best_estimator_
            logger.info(f"Best RF parameters: {grid_search.best_params_}")
        else:
            # Default parameters
            self.models['random_forest'] = RandomForestClassifier(
                n_estimators=200, max_depth=20, random_state=self.random_state
            )
            self.models['random_forest'].fit(X_train, y_train)
    
    def train_xgboost(self, X_train, y_train, hyperparameter_tuning=True):
        """Train XGBoost classifier."""
        logger.info("Training XGBoost...")
        
        if hyperparameter_tuning:
            # Hyperparameter tuning
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
            
            xgb_model = xgb.XGBClassifier(random_state=self.random_state)
            grid_search = GridSearchCV(
                xgb_model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)
            
            self.models['xgboost'] = grid_search.best_estimator_
            logger.info(f"Best XGBoost parameters: {grid_search.best_params_}")
        else:
            # Default parameters
            self.models['xgboost'] = xgb.XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1, random_state=self.random_state
            )
            self.models['xgboost'].fit(X_train, y_train)
    
    def train_logistic_regression(self, X_train, y_train, hyperparameter_tuning=True):
        """Train Logistic Regression classifier."""
        logger.info("Training Logistic Regression...")
        
        if hyperparameter_tuning:
            # Hyperparameter tuning
            param_grid = {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
            
            lr = LogisticRegression(random_state=self.random_state, max_iter=1000)
            grid_search = GridSearchCV(
                lr, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)
            
            self.models['logistic_regression'] = grid_search.best_estimator_
            logger.info(f"Best LR parameters: {grid_search.best_params_}")
        else:
            # Default parameters
            self.models['logistic_regression'] = LogisticRegression(
                random_state=self.random_state, max_iter=1000
            )
            self.models['logistic_regression'].fit(X_train, y_train)
    
    def train_deep_learning(self, X_train, y_train, X_val=None, y_val=None):
        """Train Deep Learning MLP classifier."""
        logger.info("Training Deep Learning MLP...")
        
        # Prepare validation data
        if X_val is None or y_val is None:
            X_train_dl, X_val, y_train_dl, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=self.random_state, stratify=y_train
            )
        else:
            X_train_dl = X_train
            y_train_dl = y_train
        
        # Build model
        model = keras.Sequential([
            layers.Dense(512, activation='relu', input_shape=(X_train_dl.shape[1],)),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'auc']
        )
        
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.2, patience=5, min_lr=0.001
        )
        
        # Train model
        history = model.fit(
            X_train_dl, y_train_dl,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        self.models['deep_learning'] = model
        self.models['dl_history'] = history
    
    def evaluate_models(self, X_test, y_test):
        """
        Evaluate all trained models.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            dict: Model evaluation scores
        """
        logger.info("Evaluating models...")
        
        for model_name, model in self.models.items():
            if model_name == 'dl_history':
                continue
                
            logger.info(f"Evaluating {model_name}...")
            
            # Make predictions
            if model_name == 'deep_learning':
                y_pred_proba = model.predict(X_test).flatten()
                y_pred = (y_pred_proba > 0.5).astype(int)
            else:
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            scores = {
                'accuracy': accuracy_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred)
            }
            
            self.model_scores[model_name] = scores
            
            # Log results
            logger.info(f"{model_name} results:")
            for metric, score in scores.items():
                logger.info(f"  {metric}: {score:.4f}")
        
        return self.model_scores
    
    def get_feature_importance(self, feature_names):
        """Get feature importance from tree-based models."""
        importance_dict = {}
        
        for model_name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                importance_dict[model_name] = importance_df
        
        return importance_dict
    
    def save_models(self, save_dir='models/'):
        """Save trained models."""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            if model_name == 'dl_history':
                continue
            elif model_name == 'deep_learning':
                model.save(f"{save_dir}/{model_name}.h5")
            else:
                joblib.dump(model, f"{save_dir}/{model_name}.pkl")
        
        logger.info(f"Models saved to {save_dir}")

def train_classification_models(features, labels, hyperparameter_tuning=False):
    """
    Complete classification model training pipeline.
    
    Args:
        features (pd.DataFrame): Feature matrix
        labels (pd.DataFrame): Labels DataFrame
        hyperparameter_tuning (bool): Whether to perform hyperparameter tuning
        
    Returns:
        tuple: (pipeline, model_scores, X_test, y_test)
    """
    pipeline = MLModelPipeline()
    
    # Prepare data
    X_train, X_test, y_train, y_test = pipeline.prepare_data(features, labels)
    
    # Train models
    pipeline.train_random_forest(X_train, y_train, hyperparameter_tuning)
    pipeline.train_xgboost(X_train, y_train, hyperparameter_tuning)
    pipeline.train_logistic_regression(X_train, y_train, hyperparameter_tuning)
    pipeline.train_deep_learning(X_train, y_train)
    
    # Evaluate models
    model_scores = pipeline.evaluate_models(X_test, y_test)
    
    return pipeline, model_scores, X_test, y_test

if __name__ == "__main__":
    # Example usage
    from preprocess import load_and_preprocess_data
    from integrate import integrate_omics_data
    
    # Load and integrate data
    preprocessed_data, labels, _ = load_and_preprocess_data()
    
    if preprocessed_data:
        integrated_features, merged_labels, _ = integrate_omics_data(
            preprocessed_data, labels
        )
        
        # Train models
        pipeline, scores, X_test, y_test = train_classification_models(
            integrated_features, merged_labels, hyperparameter_tuning=False
        )
        
        print("Model training completed!")
        print("\nModel Performance Summary:")
        for model_name, metrics in scores.items():
            print(f"\n{model_name.upper()}:")
            for metric, score in metrics.items():
                print(f"  {metric}: {score:.4f}")
