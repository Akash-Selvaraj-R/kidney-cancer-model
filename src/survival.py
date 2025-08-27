"""
Survival analysis module for kidney cancer prediction.
Implements Cox Proportional Hazards and Random Survival Forest models.
"""

import pandas as pd
import numpy as np
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored, integrated_brier_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SurvivalAnalysisPipeline:
    """Survival analysis pipeline for kidney cancer prognosis."""
    
    def __init__(self, random_state=42):
        """Initialize survival analysis pipeline."""
        self.random_state = random_state
        self.models = {}
        self.model_scores = {}
        
    def prepare_survival_data(self, features, labels, test_size=0.2):
        """
        Prepare data for survival analysis.
        
        Args:
            features (pd.DataFrame): Feature matrix
            labels (pd.DataFrame): Labels with survival_time and event
            test_size (float): Test set proportion
            
        Returns:
            tuple: Training and test data for survival analysis
        """
        # Extract features (exclude sample_id)
        X = features.drop('sample_id', axis=1)
        
        # Prepare survival data
        survival_data = labels[['survival_time', 'event']].copy()
        survival_data['event'] = survival_data['event'].astype(bool)
        
        # Create structured array for scikit-survival
        y_structured = np.array(
            [(event, time) for event, time in zip(survival_data['event'], survival_data['survival_time'])],
            dtype=[('event', bool), ('time', float)]
        )
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_structured, test_size=test_size, random_state=self.random_state
        )
        
        # Also prepare data for lifelines (Cox model)
        survival_train = pd.concat([
            X_train.reset_index(drop=True),
            pd.DataFrame({
                'survival_time': [y['time'] for y in y_train],
                'event': [y['event'] for y in y_train]
            })
        ], axis=1)
        
        survival_test = pd.concat([
            X_test.reset_index(drop=True),
            pd.DataFrame({
                'survival_time': [y['time'] for y in y_test],
                'event': [y['event'] for y in y_test]
            })
        ], axis=1)
        
        logger.info(f"Survival training set: {X_train.shape}")
        logger.info(f"Survival test set: {X_test.shape}")
        logger.info(f"Event rate - Train: {np.mean([y['event'] for y in y_train]):.3f}")
        logger.info(f"Event rate - Test: {np.mean([y['event'] for y in y_test]):.3f}")
        
        return (X_train, X_test, y_train, y_test, survival_train, survival_test)
    
    def train_cox_model(self, survival_train, feature_selection=True, n_features=50):
        """
        Train Cox Proportional Hazards model.
        
        Args:
            survival_train (pd.DataFrame): Training data with survival info
            feature_selection (bool): Whether to perform feature selection
            n_features (int): Number of features to select
        """
        logger.info("Training Cox Proportional Hazards model...")
        
        # Feature selection using univariate Cox regression
        if feature_selection and len(survival_train.columns) > n_features + 2:  # +2 for survival_time and event
            feature_cols = [col for col in survival_train.columns if col not in ['survival_time', 'event']]
            
            # Univariate Cox regression for feature selection
            univariate_scores = {}
            for feature in feature_cols:
                try:
                    temp_data = survival_train[['survival_time', 'event', feature]].dropna()
                    if temp_data[feature].var() > 0:  # Check for non-zero variance
                        cph_temp = CoxPHFitter()
                        cph_temp.fit(temp_data, duration_col='survival_time', event_col='event')
                        univariate_scores[feature] = abs(cph_temp.summary['coef'].iloc[0])
                except:
                    continue
            
            # Select top features
            selected_features = sorted(univariate_scores.items(), key=lambda x: x[1], reverse=True)[:n_features]
            selected_feature_names = [f[0] for f in selected_features]
            
            # Prepare data with selected features
            cox_data = survival_train[['survival_time', 'event'] + selected_feature_names]
            logger.info(f"Selected {len(selected_feature_names)} features for Cox model")
        else:
            cox_data = survival_train
        
        # Train Cox model
        self.models['cox'] = CoxPHFitter(penalizer=0.1)
        self.models['cox'].fit(cox_data, duration_col='survival_time', event_col='event')
        
        # Store selected features
        self.cox_features = [col for col in cox_data.columns if col not in ['survival_time', 'event']]
        
        logger.info("Cox model training completed")
        logger.info(f"Cox model summary:\n{self.models['cox'].summary}")
    
    def train_random_survival_forest(self, X_train, y_train, n_estimators=100):
        """
        Train Random Survival Forest model.
        
        Args:
            X_train: Training features
            y_train: Training survival data (structured array)
            n_estimators (int): Number of trees
        """
        logger.info("Training Random Survival Forest...")
        
        self.models['rsf'] = RandomSurvivalForest(
            n_estimators=n_estimators,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        self.models['rsf'].fit(X_train, y_train)
        logger.info("Random Survival Forest training completed")
    
    def evaluate_survival_models(self, X_test, y_test, survival_test):
        """
        Evaluate survival models.
        
        Args:
            X_test: Test features
            y_test: Test survival data (structured array)
            survival_test: Test data for Cox model
            
        Returns:
            dict: Model evaluation scores
        """
        logger.info("Evaluating survival models...")
        
        # Evaluate Cox model
        if 'cox' in self.models:
            try:
                # Prepare test data with selected features
                cox_test_data = survival_test[['survival_time', 'event'] + self.cox_features]
                
                # Predict risk scores
                risk_scores = self.models['cox'].predict_partial_hazard(cox_test_data)
                
                # Calculate C-index
                c_index_cox = concordance_index(
                    cox_test_data['survival_time'],
                    -risk_scores,  # Negative because higher risk = lower survival
                    cox_test_data['event']
                )
                
                self.model_scores['cox'] = {'c_index': c_index_cox}
                logger.info(f"Cox model C-index: {c_index_cox:.4f}")
                
            except Exception as e:
                logger.error(f"Error evaluating Cox model: {e}")
                self.model_scores['cox'] = {'c_index': np.nan}
        
        # Evaluate Random Survival Forest
        if 'rsf' in self.models:
            try:
                # Calculate C-index
                c_index_rsf = self.models['rsf'].score(X_test, y_test)
                
                self.model_scores['rsf'] = {'c_index': c_index_rsf}
                logger.info(f"Random Survival Forest C-index: {c_index_rsf:.4f}")
                
            except Exception as e:
                logger.error(f"Error evaluating RSF model: {e}")
                self.model_scores['rsf'] = {'c_index': np.nan}
        
        return self.model_scores
    
    def plot_kaplan_meier(self, survival_data, groupby_feature=None, save_path=None):
        """
        Plot Kaplan-Meier survival curves.
        
        Args:
            survival_data (pd.DataFrame): Survival data
            groupby_feature (str): Feature to group by (optional)
            save_path (str): Path to save plot
        """
        plt.figure(figsize=(10, 6))
        
        if groupby_feature and groupby_feature in survival_data.columns:
            # Group by feature (e.g., high/low risk)
            groups = survival_data[groupby_feature].unique()
            
            for group in groups:
                group_data = survival_data[survival_data[groupby_feature] == group]
                kmf = KaplanMeierFitter()
                kmf.fit(
                    group_data['survival_time'],
                    group_data['event'],
                    label=f'{groupby_feature}={group}'
                )
                kmf.plot_survival_function()
        else:
            # Overall survival curve
            kmf = KaplanMeierFitter()
            kmf.fit(survival_data['survival_time'], survival_data['event'])
            kmf.plot_survival_function()
        
        plt.title('Kaplan-Meier Survival Curves')
        plt.xlabel('Time')
        plt.ylabel('Survival Probability')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_survival_risk_groups(self, survival_test, model_name='cox', n_groups=3, save_path=None):
        """
        Plot survival curves for different risk groups.
        
        Args:
            survival_test (pd.DataFrame): Test survival data
            model_name (str): Model to use for risk prediction
            n_groups (int): Number of risk groups
            save_path (str): Path to save plot
        """
        if model_name not in self.models:
            logger.error(f"Model {model_name} not found")
            return
        
        # Predict risk scores
        if model_name == 'cox':
            risk_scores = self.models['cox'].predict_partial_hazard(
                survival_test[['survival_time', 'event'] + self.cox_features]
            )
        else:
            logger.error(f"Risk prediction not implemented for {model_name}")
            return
        
        # Create risk groups
        risk_quantiles = np.quantile(risk_scores, np.linspace(0, 1, n_groups + 1))
        risk_groups = pd.cut(risk_scores, bins=risk_quantiles, labels=[f'Group_{i+1}' for i in range(n_groups)])
        
        # Add risk groups to survival data
        survival_with_risk = survival_test.copy()
        survival_with_risk['risk_group'] = risk_groups
        
        # Plot Kaplan-Meier curves by risk group
        self.plot_kaplan_meier(survival_with_risk, 'risk_group', save_path)
    
    def get_feature_importance_survival(self):
        """Get feature importance from survival models."""
        importance_dict = {}
        
        # Cox model coefficients
        if 'cox' in self.models:
            cox_summary = self.models['cox'].summary
            importance_dict['cox'] = cox_summary[['coef', 'p']].sort_values('coef', key=abs, ascending=False)
        
        # RSF feature importance
        if 'rsf' in self.models:
            feature_names = [f'feature_{i}' for i in range(len(self.models['rsf'].feature_importances_))]
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': self.models['rsf'].feature_importances_
            }).sort_values('importance', ascending=False)
            importance_dict['rsf'] = importance_df
        
        return importance_dict

def train_survival_models(features, labels):
    """
    Complete survival analysis pipeline.
    
    Args:
        features (pd.DataFrame): Feature matrix
        labels (pd.DataFrame): Labels with survival information
        
    Returns:
        tuple: (pipeline, model_scores, test_data)
    """
    pipeline = SurvivalAnalysisPipeline()
    
    # Prepare survival data
    X_train, X_test, y_train, y_test, survival_train, survival_test = pipeline.prepare_survival_data(
        features, labels
    )
    
    # Train models
    pipeline.train_cox_model(survival_train)
    pipeline.train_random_survival_forest(X_train, y_train)
    
    # Evaluate models
    model_scores = pipeline.evaluate_survival_models(X_test, y_test, survival_test)
    
    return pipeline, model_scores, (X_test, y_test, survival_test)

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
        
        # Train survival models
        pipeline, scores, test_data = train_survival_models(
            integrated_features, merged_labels
        )
        
        print("Survival analysis completed!")
        print("\nSurvival Model Performance:")
        for model_name, metrics in scores.items():
            print(f"\n{model_name.upper()}:")
            for metric, score in metrics.items():
                print(f"  {metric}: {score:.4f}")
