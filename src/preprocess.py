"""
Data preprocessing module for multi-omics kidney cancer prediction.
Handles missing values, normalization, and quality control.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OmicsPreprocessor:
    """Preprocessor for multi-omics data with various normalization and imputation methods."""
    
    def __init__(self, imputation_method='knn', scaling_method='standard'):
        """
        Initialize preprocessor.
        
        Args:
            imputation_method (str): 'mean', 'median', 'knn'
            scaling_method (str): 'standard', 'robust', 'minmax'
        """
        self.imputation_method = imputation_method
        self.scaling_method = scaling_method
        self.imputers = {}
        self.scalers = {}
        
    def _get_imputer(self, method):
        """Get imputer based on method."""
        if method == 'mean':
            return SimpleImputer(strategy='mean')
        elif method == 'median':
            return SimpleImputer(strategy='median')
        elif method == 'knn':
            return KNNImputer(n_neighbors=5)
        else:
            raise ValueError(f"Unknown imputation method: {method}")
    
    def _get_scaler(self, method):
        """Get scaler based on method."""
        if method == 'standard':
            return StandardScaler()
        elif method == 'robust':
            return RobustScaler()
        elif method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            return MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
    
    def preprocess_omics_data(self, data, data_type, fit=True):
        """
        Preprocess omics data with imputation and scaling.
        
        Args:
            data (pd.DataFrame): Input data with sample_id as first column
            data_type (str): Type of omics data ('genomics', 'transcriptomics', 'proteomics')
            fit (bool): Whether to fit transformers or use existing ones
            
        Returns:
            pd.DataFrame: Preprocessed data
        """
        logger.info(f"Preprocessing {data_type} data...")
        
        # Separate sample IDs and features
        sample_ids = data['sample_id'].copy()
        features = data.drop('sample_id', axis=1)
        
        # Log data statistics
        logger.info(f"Original shape: {features.shape}")
        logger.info(f"Missing values: {features.isnull().sum().sum()}")
        
        # Remove features with too many missing values (>50%)
        missing_threshold = 0.5
        missing_ratio = features.isnull().sum() / len(features)
        features_to_keep = missing_ratio[missing_ratio <= missing_threshold].index
        features = features[features_to_keep]
        
        logger.info(f"Shape after removing high-missing features: {features.shape}")
        
        # Imputation
        if fit:
            self.imputers[data_type] = self._get_imputer(self.imputation_method)
            features_imputed = pd.DataFrame(
                self.imputers[data_type].fit_transform(features),
                columns=features.columns,
                index=features.index
            )
        else:
            features_imputed = pd.DataFrame(
                self.imputers[data_type].transform(features),
                columns=features.columns,
                index=features.index
            )
        
        # Scaling
        if fit:
            self.scalers[data_type] = self._get_scaler(self.scaling_method)
            features_scaled = pd.DataFrame(
                self.scalers[data_type].fit_transform(features_imputed),
                columns=features_imputed.columns,
                index=features_imputed.index
            )
        else:
            features_scaled = pd.DataFrame(
                self.scalers[data_type].transform(features_imputed),
                columns=features_imputed.columns,
                index=features_imputed.index
            )
        
        # Combine with sample IDs
        result = pd.concat([sample_ids, features_scaled], axis=1)
        
        logger.info(f"Final preprocessed shape: {result.shape}")
        return result
    
    def quality_control(self, data, data_type):
        """
        Perform quality control checks on omics data.
        
        Args:
            data (pd.DataFrame): Input data
            data_type (str): Type of omics data
            
        Returns:
            dict: QC metrics
        """
        features = data.drop('sample_id', axis=1)
        
        qc_metrics = {
            'n_samples': len(data),
            'n_features': len(features.columns),
            'missing_values_total': features.isnull().sum().sum(),
            'missing_values_percent': (features.isnull().sum().sum() / features.size) * 100,
            'features_with_zero_variance': (features.var() == 0).sum(),
            'mean_feature_correlation': np.abs(features.corr()).mean().mean()
        }
        
        logger.info(f"QC metrics for {data_type}:")
        for metric, value in qc_metrics.items():
            logger.info(f"  {metric}: {value}")
        
        return qc_metrics

def load_and_preprocess_data(data_dir='data/', imputation_method='knn', scaling_method='standard'):
    """
    Load and preprocess all omics datasets.
    
    Args:
        data_dir (str): Directory containing CSV files
        imputation_method (str): Imputation method
        scaling_method (str): Scaling method
        
    Returns:
        tuple: (preprocessed_data_dict, labels, preprocessor)
    """
    preprocessor = OmicsPreprocessor(imputation_method, scaling_method)
    
    # Load raw data
    omics_types = ['genomics', 'transcriptomics', 'proteomics']
    raw_data = {}
    
    for omics_type in omics_types:
        try:
            filepath = f"{data_dir}/{omics_type}.csv"
            raw_data[omics_type] = pd.read_csv(filepath)
            logger.info(f"Loaded {omics_type} data: {raw_data[omics_type].shape}")
        except FileNotFoundError:
            logger.warning(f"File not found: {filepath}")
            continue
    
    # Load labels
    try:
        labels = pd.read_csv(f"{data_dir}/labels.csv")
        logger.info(f"Loaded labels: {labels.shape}")
    except FileNotFoundError:
        logger.error("Labels file not found!")
        return None, None, None
    
    # Preprocess each omics type
    preprocessed_data = {}
    for omics_type, data in raw_data.items():
        # Quality control
        qc_metrics = preprocessor.quality_control(data, omics_type)
        
        # Preprocess
        preprocessed_data[omics_type] = preprocessor.preprocess_omics_data(
            data, omics_type, fit=True
        )
    
    return preprocessed_data, labels, preprocessor

if __name__ == "__main__":
    # Example usage
    preprocessed_data, labels, preprocessor = load_and_preprocess_data()
    
    if preprocessed_data:
        print("Preprocessing completed successfully!")
        for omics_type, data in preprocessed_data.items():
            print(f"{omics_type}: {data.shape}")
