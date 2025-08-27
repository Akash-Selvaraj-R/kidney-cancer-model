"""
Exploratory Data Analysis notebook for multi-omics kidney cancer data.
This script can be converted to a Jupyter notebook for interactive analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_data(data_dir='data/'):
    """Load all datasets for exploration."""
    datasets = {}
    
    # Load each dataset
    for dataset_name in ['labels', 'genomics', 'transcriptomics', 'proteomics']:
        try:
            datasets[dataset_name] = pd.read_csv(f"{data_dir}/{dataset_name}.csv")
            print(f"Loaded {dataset_name}: {datasets[dataset_name].shape}")
        except FileNotFoundError:
            print(f"Warning: {dataset_name}.csv not found")
    
    return datasets

def explore_labels(labels_df):
    """Explore the labels dataset."""
    print("="*50)
    print("LABELS EXPLORATION")
    print("="*50)
    
    # Basic statistics
    print("\nBasic Statistics:")
    print(labels_df.describe())
    
    # Cancer status distribution
    print(f"\nCancer Status Distribution:")
    print(labels_df['cancer_status'].value_counts())
    
    # Survival statistics
    print(f"\nSurvival Statistics:")
    print(f"Mean survival time: {labels_df['survival_time'].mean():.1f} days")
    print(f"Median survival time: {labels_df['survival_time'].median():.1f} days")
    print(f"Event rate: {labels_df['event'].mean():.2%}")
    
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Cancer status distribution
    labels_df['cancer_status'].value_counts().plot(kind='bar', ax=axes[0,0])
    axes[0,0].set_title('Cancer Status Distribution')
    axes[0,0].set_ylabel('Count')
    
    # Survival time distribution
    axes[0,1].hist(labels_df['survival_time'], bins=30, alpha=0.7)
    axes[0,1].set_title('Survival Time Distribution')
    axes[0,1].set_xlabel('Survival Time (days)')
    axes[0,1].set_ylabel('Frequency')
    
    # Event distribution
    labels_df['event'].value_counts().plot(kind='bar', ax=axes[1,0])
    axes[1,0].set_title('Event Distribution')
    axes[1,0].set_ylabel('Count')
    
    # Survival time by cancer status
    for status in labels_df['cancer_status'].unique():
        subset = labels_df[labels_df['cancer_status'] == status]
        axes[1,1].hist(subset['survival_time'], alpha=0.6, label=status, bins=20)
    axes[1,1].set_title('Survival Time by Cancer Status')
    axes[1,1].set_xlabel('Survival Time (days)')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.show()

def explore_omics_data(omics_dict, labels_df):
    """Explore omics datasets."""
    print("="*50)
    print("OMICS DATA EXPLORATION")
    print("="*50)
    
    for omics_name, omics_df in omics_dict.items():
        if omics_name == 'labels':
            continue
            
        print(f"\n{omics_name.upper()} Dataset:")
        print(f"Shape: {omics_df.shape}")
        
        # Feature columns (exclude sample_id)
        feature_cols = [col for col in omics_df.columns if col != 'sample_id']
        features = omics_df[feature_cols]
        
        # Missing values
        missing_percent = (features.isnull().sum().sum() / features.size) * 100
        print(f"Missing values: {missing_percent:.2f}%")
        
        # Basic statistics
        print(f"Mean feature value: {features.mean().mean():.3f}")
        print(f"Std feature value: {features.std().mean():.3f}")
        
        # Distribution plots
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Feature value distribution
        sample_features = features.iloc[:, :100].values.flatten()  # Sample first 100 features
        axes[0].hist(sample_features, bins=50, alpha=0.7)
        axes[0].set_title(f'{omics_name.title()} - Feature Value Distribution')
        axes[0].set_xlabel('Feature Value')
        axes[0].set_ylabel('Frequency')
        
        # Missing values per feature
        missing_per_feature = features.isnull().sum()
        axes[1].hist(missing_per_feature, bins=30, alpha=0.7)
        axes[1].set_title(f'{omics_name.title()} - Missing Values per Feature')
        axes[1].set_xlabel('Missing Values')
        axes[1].set_ylabel('Number of Features')
        
        # Feature variance
        feature_variance = features.var()
        axes[2].hist(feature_variance, bins=50, alpha=0.7)
        axes[2].set_title(f'{omics_name.title()} - Feature Variance Distribution')
        axes[2].set_xlabel('Variance')
        axes[2].set_ylabel('Number of Features')
        
        plt.tight_layout()
        plt.show()

def perform_dimensionality_reduction(omics_dict, labels_df):
    """Perform PCA and t-SNE on integrated data."""
    print("="*50)
    print("DIMENSIONALITY REDUCTION")
    print("="*50)
    
    # Merge all omics data
    merged_data = labels_df[['sample_id', 'cancer_status']].copy()
    
    for omics_name, omics_df in omics_dict.items():
        if omics_name == 'labels':
            continue
        
        # Select top 100 features by variance
        feature_cols = [col for col in omics_df.columns if col != 'sample_id']
        features = omics_df[feature_cols]
        top_features = features.var().nlargest(100).index
        
        # Rename columns to avoid conflicts
        omics_subset = omics_df[['sample_id'] + list(top_features)].copy()
        omics_subset.columns = ['sample_id'] + [f"{omics_name}_{col}" for col in top_features]
        
        # Merge
        merged_data = merged_data.merge(omics_subset, on='sample_id', how='inner')
    
    print(f"Merged data shape: {merged_data.shape}")
    
    # Prepare feature matrix
    feature_cols = [col for col in merged_data.columns if col not in ['sample_id', 'cancer_status']]
    X = merged_data[feature_cols].fillna(0)  # Simple imputation for exploration
    y = merged_data['cancer_status']
    
    # PCA
    print("Performing PCA...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    print(f"PCA explained variance: {pca.explained_variance_ratio_}")
    
    # t-SNE
    print("Performing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X)
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # PCA plot
    colors = {'tumor': 'red', 'normal': 'blue'}
    for status in y.unique():
        mask = y == status
        axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1], 
                       c=colors[status], label=status, alpha=0.6)
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    axes[0].set_title('PCA - Integrated Omics Data')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # t-SNE plot
    for status in y.unique():
        mask = y == status
        axes[1].scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                       c=colors[status], label=status, alpha=0.6)
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')
    axes[1].set_title('t-SNE - Integrated Omics Data')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def correlation_analysis(omics_dict):
    """Analyze correlations within and between omics types."""
    print("="*50)
    print("CORRELATION ANALYSIS")
    print("="*50)
    
    # Sample features from each omics type
    sampled_features = {}
    
    for omics_name, omics_df in omics_dict.items():
        if omics_name == 'labels':
            continue
        
        feature_cols = [col for col in omics_df.columns if col != 'sample_id']
        features = omics_df[feature_cols]
        
        # Sample 50 features with highest variance
        top_features = features.var().nlargest(50).index
        sampled_features[omics_name] = features[top_features]
    
    # Calculate correlations
    fig, axes = plt.subplots(1, len(sampled_features), figsize=(5*len(sampled_features), 4))
    if len(sampled_features) == 1:
        axes = [axes]
    
    for idx, (omics_name, features) in enumerate(sampled_features.items()):
        # Calculate correlation matrix
        corr_matrix = features.corr()
        
        # Plot heatmap
        sns.heatmap(corr_matrix, cmap='coolwarm', center=0, 
                   square=True, ax=axes[idx], cbar_kws={'shrink': 0.8})
        axes[idx].set_title(f'{omics_name.title()} - Feature Correlations')
    
    plt.tight_layout()
    plt.show()
    
    # Cross-omics correlations
    if len(sampled_features) > 1:
        print("\nCross-omics correlation analysis...")
        omics_names = list(sampled_features.keys())
        
        for i in range(len(omics_names)):
            for j in range(i+1, len(omics_names)):
                omics1, omics2 = omics_names[i], omics_names[j]
                
                # Calculate cross-correlation
                features1 = sampled_features[omics1]
                features2 = sampled_features[omics2]
                
                # Sample correlation between mean values
                mean1 = features1.mean(axis=1)
                mean2 = features2.mean(axis=1)
                cross_corr = np.corrcoef(mean1, mean2)[0, 1]
                
                print(f"{omics1} vs {omics2} correlation: {cross_corr:.3f}")

def main():
    """Main exploration function."""
    # Generate sample data if not exists
    if not any(Path(f"data/{f}.csv").exists() for f in ['labels', 'genomics', 'transcriptomics', 'proteomics']):
        print("Sample data not found. Generating sample datasets...")
        from scripts.generate_sample_data import generate_sample_data
        generate_sample_data()
    
    # Load data
    datasets = load_data()
    
    if 'labels' not in datasets:
        print("Error: Labels dataset not found!")
        return
    
    # Explore labels
    explore_labels(datasets['labels'])
    
    # Explore omics data
    omics_data = {k: v for k, v in datasets.items() if k != 'labels'}
    if omics_data:
        explore_omics_data(omics_data, datasets['labels'])
        perform_dimensionality_reduction(omics_data, datasets['labels'])
        correlation_analysis(omics_data)
    
    print("="*50)
    print("EXPLORATORY ANALYSIS COMPLETED!")
    print("="*50)

if __name__ == "__main__":
    from pathlib import Path
    main()
