"""
Multi-omics integration module for kidney cancer prediction.
Handles data merging, feature selection, and dimensionality reduction.
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow import keras
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiOmicsIntegrator:
    """Integrates multiple omics datasets with various feature selection methods."""
    
    def __init__(self, feature_selection_method='pca', n_components=100):
        """
        Initialize integrator.
        
        Args:
            feature_selection_method (str): 'pca', 'autoencoder', 'univariate', 'rf_importance'
            n_components (int): Number of components/features to select
        """
        self.feature_selection_method = feature_selection_method
        self.n_components = n_components
        self.feature_selectors = {}
        self.selected_features = {}
        
    def merge_omics_data(self, omics_data_dict, labels):
        """
        Merge multiple omics datasets on sample_id.
        
        Args:
            omics_data_dict (dict): Dictionary of preprocessed omics DataFrames
            labels (pd.DataFrame): Labels DataFrame with sample_id
            
        Returns:
            tuple: (merged_features, merged_labels)
        """
        logger.info("Merging omics datasets...")
        
        # Start with labels
        merged_data = labels.copy()
        
        # Merge each omics type
        for omics_type, data in omics_data_dict.items():
            logger.info(f"Merging {omics_type} data...")
            
            # Add prefix to feature names to avoid conflicts
            feature_cols = [col for col in data.columns if col != 'sample_id']
            data_renamed = data.copy()
            data_renamed.columns = ['sample_id'] + [f"{omics_type}_{col}" for col in feature_cols]
            
            # Merge on sample_id
            merged_data = merged_data.merge(data_renamed, on='sample_id', how='inner')
            logger.info(f"Shape after merging {omics_type}: {merged_data.shape}")
        
        # Separate features and labels
        feature_cols = [col for col in merged_data.columns 
                       if col not in ['sample_id', 'cancer_status', 'survival_time', 'event']]
        
        merged_features = merged_data[['sample_id'] + feature_cols]
        merged_labels = merged_data[['sample_id', 'cancer_status', 'survival_time', 'event']]
        
        logger.info(f"Final merged shape - Features: {merged_features.shape}, Labels: {merged_labels.shape}")
        return merged_features, merged_labels
    
    def apply_pca(self, features, omics_type, fit=True):
        """Apply PCA for dimensionality reduction."""
        if fit:
            self.feature_selectors[omics_type] = PCA(n_components=self.n_components)
            transformed = self.feature_selectors[omics_type].fit_transform(features)
        else:
            transformed = self.feature_selectors[omics_type].transform(features)
        
        # Create DataFrame with PCA component names
        component_names = [f"{omics_type}_PC{i+1}" for i in range(transformed.shape[1])]
        return pd.DataFrame(transformed, columns=component_names, index=features.index)
    
    def apply_autoencoder(self, features, omics_type, fit=True):
        """Apply autoencoder for dimensionality reduction."""
        input_dim = features.shape[1]
        encoding_dim = min(self.n_components, input_dim // 2)
        
        if fit:
            # Build autoencoder
            input_layer = keras.layers.Input(shape=(input_dim,))
            encoded = keras.layers.Dense(encoding_dim, activation='relu')(input_layer)
            decoded = keras.layers.Dense(input_dim, activation='linear')(encoded)
            
            autoencoder = keras.models.Model(input_layer, decoded)
            encoder = keras.models.Model(input_layer, encoded)
            
            autoencoder.compile(optimizer='adam', loss='mse')
            
            # Train autoencoder
            autoencoder.fit(features, features, epochs=50, batch_size=32, verbose=0)
            
            self.feature_selectors[omics_type] = encoder
        
        # Get encoded features
        encoded_features = self.feature_selectors[omics_type].predict(features)
        
        # Create DataFrame
        component_names = [f"{omics_type}_AE{i+1}" for i in range(encoded_features.shape[1])]
        return pd.DataFrame(encoded_features, columns=component_names, index=features.index)
    
    def apply_univariate_selection(self, features, labels, omics_type, fit=True):
        """Apply univariate feature selection."""
        if fit:
            self.feature_selectors[omics_type] = SelectKBest(
                score_func=f_classif, k=self.n_components
            )
            selected_features = self.feature_selectors[omics_type].fit_transform(features, labels)
        else:
            selected_features = self.feature_selectors[omics_type].transform(features)
        
        # Get selected feature names
        selected_indices = self.feature_selectors[omics_type].get_support(indices=True)
        selected_names = features.columns[selected_indices]
        
        return pd.DataFrame(selected_features, columns=selected_names, index=features.index)
    
    def apply_rf_importance(self, features, labels, omics_type, fit=True):
        """Apply Random Forest feature importance selection."""
        if fit:
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(features, labels)
            
            # Get feature importances
            importances = rf.feature_importances_
            indices = np.argsort(importances)[::-1][:self.n_components]
            
            self.feature_selectors[omics_type] = indices
        
        # Select top features
        selected_features = features.iloc[:, self.feature_selectors[omics_type]]
        return selected_features
    
    def select_features_by_omics(self, merged_features, labels, fit=True):
        """
        Apply feature selection separately for each omics type.
        
        Args:
            merged_features (pd.DataFrame): Merged features DataFrame
            labels (pd.Series): Target labels for supervised selection
            fit (bool): Whether to fit selectors
            
        Returns:
            pd.DataFrame: Features after selection
        """
        logger.info(f"Applying {self.feature_selection_method} feature selection...")
        
        # Separate features by omics type
        omics_features = {}
        for col in merged_features.columns:
            if col == 'sample_id':
                continue
            
            omics_type = col.split('_')[0]
            if omics_type not in omics_features:
                omics_features[omics_type] = []
            omics_features[omics_type].append(col)
        
        # Apply feature selection to each omics type
        selected_data_list = [merged_features[['sample_id']]]
        
        for omics_type, feature_cols in omics_features.items():
            logger.info(f"Selecting features for {omics_type} ({len(feature_cols)} features)...")
            
            omics_data = merged_features[feature_cols]
            
            if self.feature_selection_method == 'pca':
                selected = self.apply_pca(omics_data, omics_type, fit)
            elif self.feature_selection_method == 'autoencoder':
                selected = self.apply_autoencoder(omics_data, omics_type, fit)
            elif self.feature_selection_method == 'univariate':
                selected = self.apply_univariate_selection(omics_data, labels, omics_type, fit)
            elif self.feature_selection_method == 'rf_importance':
                selected = self.apply_rf_importance(omics_data, labels, omics_type, fit)
            else:
                raise ValueError(f"Unknown feature selection method: {self.feature_selection_method}")
            
            selected_data_list.append(selected)
            logger.info(f"Selected {selected.shape[1]} features for {omics_type}")
        
        # Combine all selected features
        result = pd.concat(selected_data_list, axis=1)
        logger.info(f"Total selected features: {result.shape[1] - 1}")  # -1 for sample_id
        
        return result
    
    def get_feature_importance_summary(self):
        """Get summary of feature selection results."""
        summary = {}
        
        for omics_type, selector in self.feature_selectors.items():
            if self.feature_selection_method == 'pca':
                summary[omics_type] = {
                    'explained_variance_ratio': selector.explained_variance_ratio_,
                    'cumulative_variance': np.cumsum(selector.explained_variance_ratio_)
                }
            elif self.feature_selection_method == 'univariate':
                summary[omics_type] = {
                    'scores': selector.scores_,
                    'selected_features': selector.get_support(indices=True)
                }
        
        return summary

def integrate_omics_data(omics_data_dict, labels, feature_selection_method='pca', n_components=100):
    """
    Complete multi-omics integration pipeline.
    
    Args:
        omics_data_dict (dict): Dictionary of preprocessed omics DataFrames
        labels (pd.DataFrame): Labels DataFrame
        feature_selection_method (str): Feature selection method
        n_components (int): Number of components to select
        
    Returns:
        tuple: (integrated_features, labels, integrator)
    """
    integrator = MultiOmicsIntegrator(feature_selection_method, n_components)
    
    # Merge omics data
    merged_features, merged_labels = integrator.merge_omics_data(omics_data_dict, labels)
    
    # Apply feature selection
    target_labels = merged_labels['cancer_status'].map({'tumor': 1, 'normal': 0})
    integrated_features = integrator.select_features_by_omics(
        merged_features, target_labels, fit=True
    )
    
    return integrated_features, merged_labels, integrator

if __name__ == "__main__":
    # Example usage
    from preprocess import load_and_preprocess_data
    
    # Load preprocessed data
    preprocessed_data, labels, _ = load_and_preprocess_data()
    
    if preprocessed_data:
        # Integrate omics data
        integrated_features, merged_labels, integrator = integrate_omics_data(
            preprocessed_data, labels, feature_selection_method='pca', n_components=50
        )
        
        print("Integration completed successfully!")
        print(f"Integrated features shape: {integrated_features.shape}")
        print(f"Labels shape: {merged_labels.shape}")
