"""
Model evaluation and visualization module.
Generates comprehensive evaluation metrics and plots for all models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    classification_report
)
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluation and visualization."""
    
    def __init__(self, output_dir='results/'):
        """Initialize evaluator."""
        self.output_dir = output_dir
        import os
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_confusion_matrices(self, models, X_test, y_test, save_path=None):
        """
        Plot confusion matrices for all classification models.
        
        Args:
            models (dict): Dictionary of trained models
            X_test: Test features
            y_test: Test labels
            save_path (str): Path to save plot
        """
        n_models = len([m for m in models.keys() if m != 'dl_history'])
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        model_idx = 0
        for model_name, model in models.items():
            if model_name == 'dl_history':
                continue
                
            # Make predictions
            if model_name == 'deep_learning':
                y_pred = (model.predict(X_test).flatten() > 0.5).astype(int)
            else:
                y_pred = model.predict(X_test)
            
            # Compute confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # Plot
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Tumor'],
                yticklabels=['Normal', 'Tumor'],
                ax=axes[model_idx]
            )
            axes[model_idx].set_title(f'{model_name.replace("_", " ").title()}')
            axes[model_idx].set_xlabel('Predicted')
            axes[model_idx].set_ylabel('Actual')
            
            model_idx += 1
        
        # Hide unused subplots
        for i in range(model_idx, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_roc_curves(self, models, X_test, y_test, save_path=None):
        """
        Plot ROC curves for all classification models.
        
        Args:
            models (dict): Dictionary of trained models
            X_test: Test features
            y_test: Test labels
            save_path (str): Path to save plot
        """
        plt.figure(figsize=(10, 8))
        
        for model_name, model in models.items():
            if model_name == 'dl_history':
                continue
                
            # Get prediction probabilities
            if model_name == 'deep_learning':
                y_pred_proba = model.predict(X_test).flatten()
            else:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Compute ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            # Plot
            plt.plot(
                fpr, tpr,
                label=f'{model_name.replace("_", " ").title()} (AUC = {roc_auc:.3f})',
                linewidth=2
            )
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Classification Models')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_precision_recall_curves(self, models, X_test, y_test, save_path=None):
        """
        Plot Precision-Recall curves for all classification models.
        
        Args:
            models (dict): Dictionary of trained models
            X_test: Test features
            y_test: Test labels
            save_path (str): Path to save plot
        """
        plt.figure(figsize=(10, 8))
        
        for model_name, model in models.items():
            if model_name == 'dl_history':
                continue
                
            # Get prediction probabilities
            if model_name == 'deep_learning':
                y_pred_proba = model.predict(X_test).flatten()
            else:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Compute Precision-Recall curve
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            pr_auc = auc(recall, precision)
            
            # Plot
            plt.plot(
                recall, precision,
                label=f'{model_name.replace("_", " ").title()} (AUC = {pr_auc:.3f})',
                linewidth=2
            )
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves - Classification Models')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_feature_importance(self, importance_dict, top_n=20, save_path=None):
        """
        Plot feature importance for tree-based models.
        
        Args:
            importance_dict (dict): Dictionary of feature importance DataFrames
            top_n (int): Number of top features to show
            save_path (str): Path to save plot
        """
        n_models = len(importance_dict)
        if n_models == 0:
            logger.warning("No feature importance data available")
            return
        
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 8))
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, importance_df) in enumerate(importance_dict.items()):
            top_features = importance_df.head(top_n)
            
            # Plot horizontal bar chart
            y_pos = np.arange(len(top_features))
            axes[idx].barh(y_pos, top_features['importance'])
            axes[idx].set_yticks(y_pos)
            axes[idx].set_yticklabels(top_features['feature'])
            axes[idx].invert_yaxis()
            axes[idx].set_xlabel('Importance')
            axes[idx].set_title(f'{model_name.replace("_", " ").title()}\nTop {top_n} Features')
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_pca_scatter(self, features, labels, pca_components=None, save_path=None):
        """
        Plot PCA scatter plot of integrated features.
        
        Args:
            features (pd.DataFrame): Feature matrix
            labels (pd.DataFrame): Labels
            pca_components (np.array): PCA components (optional)
            save_path (str): Path to save plot
        """
        from sklearn.decomposition import PCA
        
        # Prepare data
        X = features.drop('sample_id', axis=1)
        y = labels['cancer_status'].map({'tumor': 1, 'normal': 0})
        
        # Apply PCA if components not provided
        if pca_components is None:
            pca = PCA(n_components=2)
            pca_components = pca.fit_transform(X)
            explained_var = pca.explained_variance_ratio_
        else:
            explained_var = [0, 0]  # Placeholder
        
        # Create scatter plot
        plt.figure(figsize=(10, 8))
        
        colors = ['blue', 'red']
        labels_text = ['Normal', 'Tumor']
        
        for i, (color, label) in enumerate(zip(colors, labels_text)):
            mask = y == i
            plt.scatter(
                pca_components[mask, 0],
                pca_components[mask, 1],
                c=color, label=label, alpha=0.6, s=50
            )
        
        plt.xlabel(f'PC1 ({explained_var[0]:.1%} variance)')
        plt.ylabel(f'PC2 ({explained_var[1]:.1%} variance)')
        plt.title('PCA Scatter Plot - Integrated Omics Features')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_model_comparison_table(self, classification_scores, survival_scores=None):
        """
        Create a comprehensive model comparison table.
        
        Args:
            classification_scores (dict): Classification model scores
            survival_scores (dict): Survival model scores (optional)
            
        Returns:
            pd.DataFrame: Comparison table
        """
        # Classification models
        classification_df = pd.DataFrame(classification_scores).T
        classification_df.index.name = 'Model'
        
        # Add model type
        classification_df['Model_Type'] = 'Classification'
        
        # Survival models
        if survival_scores:
            survival_df = pd.DataFrame(survival_scores).T
            survival_df.index.name = 'Model'
            survival_df['Model_Type'] = 'Survival'
            
            # Combine tables
            comparison_df = pd.concat([classification_df, survival_df], sort=False)
        else:
            comparison_df = classification_df
        
        # Round numeric columns
        numeric_cols = comparison_df.select_dtypes(include=[np.number]).columns
        comparison_df[numeric_cols] = comparison_df[numeric_cols].round(4)
        
        return comparison_df
    
    def generate_evaluation_report(self, classification_pipeline, survival_pipeline=None,
                                 X_test=None, y_test=None, features=None, labels=None):
        """
        Generate comprehensive evaluation report with all visualizations.
        
        Args:
            classification_pipeline: Trained classification pipeline
            survival_pipeline: Trained survival pipeline (optional)
            X_test: Test features for classification
            y_test: Test labels for classification
            features: Full feature matrix
            labels: Full labels DataFrame
        """
        logger.info("Generating comprehensive evaluation report...")
        
        # 1. Confusion Matrices
        if X_test is not None and y_test is not None:
            self.plot_confusion_matrices(
                classification_pipeline.models, X_test, y_test,
                save_path=f"{self.output_dir}/confusion_matrices.png"
            )
        
        # 2. ROC Curves
        if X_test is not None and y_test is not None:
            self.plot_roc_curves(
                classification_pipeline.models, X_test, y_test,
                save_path=f"{self.output_dir}/roc_curves.png"
            )
        
        # 3. Precision-Recall Curves
        if X_test is not None and y_test is not None:
            self.plot_precision_recall_curves(
                classification_pipeline.models, X_test, y_test,
                save_path=f"{self.output_dir}/precision_recall_curves.png"
            )
        
        # 4. Feature Importance
        if hasattr(classification_pipeline, 'get_feature_importance'):
            feature_names = [col for col in features.columns if col != 'sample_id']
            importance_dict = classification_pipeline.get_feature_importance(feature_names)
            if importance_dict:
                self.plot_feature_importance(
                    importance_dict,
                    save_path=f"{self.output_dir}/feature_importance.png"
                )
        
        # 5. PCA Scatter Plot
        if features is not None and labels is not None:
            self.plot_pca_scatter(
                features, labels,
                save_path=f"{self.output_dir}/pca_scatter.png"
            )
        
        # 6. Survival Analysis Plots
        if survival_pipeline is not None:
            # Kaplan-Meier curves
            survival_data = labels[['survival_time', 'event']].copy()
            survival_pipeline.plot_kaplan_meier(
                survival_data,
                save_path=f"{self.output_dir}/kaplan_meier.png"
            )
        
        # 7. Model Comparison Table
        comparison_table = self.create_model_comparison_table(
            classification_pipeline.model_scores,
            survival_pipeline.model_scores if survival_pipeline else None
        )
        
        # Save comparison table
        comparison_table.to_csv(f"{self.output_dir}/model_comparison.csv")
        
        logger.info(f"Evaluation report saved to {self.output_dir}")
        
        return comparison_table

def evaluate_all_models(classification_pipeline, survival_pipeline=None,
                       X_test=None, y_test=None, features=None, labels=None):
    """
    Complete model evaluation pipeline.
    
    Args:
        classification_pipeline: Trained classification pipeline
        survival_pipeline: Trained survival pipeline (optional)
        X_test: Test features
        y_test: Test labels
        features: Full feature matrix
        labels: Full labels DataFrame
        
    Returns:
        tuple: (evaluator, comparison_table)
    """
    evaluator = ModelEvaluator()
    
    comparison_table = evaluator.generate_evaluation_report(
        classification_pipeline, survival_pipeline,
        X_test, y_test, features, labels
    )
    
    return evaluator, comparison_table

if __name__ == "__main__":
    # Example usage
    from preprocess import load_and_preprocess_data
    from integrate import integrate_omics_data
    from models import train_classification_models
    from survival import train_survival_models
    
    # Load and process data
    preprocessed_data, labels, _ = load_and_preprocess_data()
    
    if preprocessed_data:
        # Integrate data
        integrated_features, merged_labels, _ = integrate_omics_data(
            preprocessed_data, labels
        )
        
        # Train models
        classification_pipeline, class_scores, X_test, y_test = train_classification_models(
            integrated_features, merged_labels
        )
        
        survival_pipeline, surv_scores, _ = train_survival_models(
            integrated_features, merged_labels
        )
        
        # Evaluate models
        evaluator, comparison_table = evaluate_all_models(
            classification_pipeline, survival_pipeline,
            X_test, y_test, integrated_features, merged_labels
        )
        
        print("Model evaluation completed!")
        print("\nModel Comparison Table:")
        print(comparison_table)
