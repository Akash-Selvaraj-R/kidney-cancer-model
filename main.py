"""
Main pipeline script for Multi-Omics Kidney Cancer Prediction.
Orchestrates the complete workflow from data loading to model evaluation.
"""

import argparse
import logging
import sys
from pathlib import Path

# Import custom modules
from src.preprocess import load_and_preprocess_data
from src.integrate import integrate_omics_data
from src.models import train_classification_models
from src.survival import train_survival_models
from src.evaluate import evaluate_all_models

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('kidney_cancer_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main pipeline execution."""
    parser = argparse.ArgumentParser(description='Multi-Omics Kidney Cancer Prediction Pipeline')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data/',
                       help='Directory containing CSV files')
    parser.add_argument('--output_dir', type=str, default='results/',
                       help='Directory for output files')
    
    # Preprocessing arguments
    parser.add_argument('--imputation_method', type=str, default='knn',
                       choices=['mean', 'median', 'knn'],
                       help='Imputation method for missing values')
    parser.add_argument('--scaling_method', type=str, default='standard',
                       choices=['standard', 'robust', 'minmax'],
                       help='Scaling method for features')
    
    # Integration arguments
    parser.add_argument('--feature_selection', type=str, default='pca',
                       choices=['pca', 'autoencoder', 'univariate', 'rf_importance'],
                       help='Feature selection method')
    parser.add_argument('--n_components', type=int, default=100,
                       help='Number of components/features to select')
    
    # Model training arguments
    parser.add_argument('--hyperparameter_tuning', action='store_true',
                       help='Enable hyperparameter tuning (slower but better results)')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set proportion')
    
    # Pipeline control
    parser.add_argument('--skip_classification', action='store_true',
                       help='Skip classification models')
    parser.add_argument('--skip_survival', action='store_true',
                       help='Skip survival analysis')
    parser.add_argument('--skip_evaluation', action='store_true',
                       help='Skip comprehensive evaluation')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info("MULTI-OMICS KIDNEY CANCER PREDICTION PIPELINE")
    logger.info("="*60)
    
    try:
        # Step 1: Data Loading and Preprocessing
        logger.info("Step 1: Loading and preprocessing data...")
        preprocessed_data, labels, preprocessor = load_and_preprocess_data(
            data_dir=args.data_dir,
            imputation_method=args.imputation_method,
            scaling_method=args.scaling_method
        )
        
        if not preprocessed_data:
            logger.error("Failed to load data. Please check your data directory and files.")
            return
        
        logger.info(f"Successfully loaded {len(preprocessed_data)} omics datasets")
        
        # Step 2: Multi-Omics Integration
        logger.info("Step 2: Integrating multi-omics data...")
        integrated_features, merged_labels, integrator = integrate_omics_data(
            preprocessed_data, labels,
            feature_selection_method=args.feature_selection,
            n_components=args.n_components
        )
        
        logger.info(f"Integration completed. Final feature matrix: {integrated_features.shape}")
        
        # Initialize variables for later use
        classification_pipeline = None
        survival_pipeline = None
        X_test, y_test = None, None
        
        # Step 3: Classification Models
        if not args.skip_classification:
            logger.info("Step 3: Training classification models...")
            classification_pipeline, class_scores, X_test, y_test = train_classification_models(
                integrated_features, merged_labels,
                hyperparameter_tuning=args.hyperparameter_tuning
            )
            
            logger.info("Classification model training completed!")
            logger.info("Classification Results Summary:")
            for model_name, metrics in class_scores.items():
                logger.info(f"  {model_name.upper()}:")
                for metric, score in metrics.items():
                    logger.info(f"    {metric}: {score:.4f}")
        
        # Step 4: Survival Analysis
        if not args.skip_survival:
            logger.info("Step 4: Training survival models...")
            survival_pipeline, surv_scores, _ = train_survival_models(
                integrated_features, merged_labels
            )
            
            logger.info("Survival analysis completed!")
            logger.info("Survival Results Summary:")
            for model_name, metrics in surv_scores.items():
                logger.info(f"  {model_name.upper()}:")
                for metric, score in metrics.items():
                    logger.info(f"    {metric}: {score:.4f}")
        
        # Step 5: Comprehensive Evaluation
        if not args.skip_evaluation and (classification_pipeline or survival_pipeline):
            logger.info("Step 5: Generating comprehensive evaluation report...")
            evaluator, comparison_table = evaluate_all_models(
                classification_pipeline, survival_pipeline,
                X_test, y_test, integrated_features, merged_labels
            )
            
            # Save comparison table
            comparison_table.to_csv(f"{args.output_dir}/model_comparison.csv")
            
            logger.info("Comprehensive evaluation completed!")
            logger.info(f"Results saved to {args.output_dir}")
            
            # Display final comparison table
            logger.info("\nFINAL MODEL COMPARISON:")
            logger.info("\n" + comparison_table.to_string())
        
        # Step 6: Save Models
        if classification_pipeline:
            classification_pipeline.save_models(f"{args.output_dir}/models/")
        
        logger.info("="*60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}")
        logger.error("Please check the logs for detailed error information.")
        raise

if __name__ == "__main__":
    main()
