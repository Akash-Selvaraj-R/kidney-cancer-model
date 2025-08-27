# Multi-Omics Driven Predictive Modeling for Kidney Cancer

A comprehensive machine learning pipeline for predicting kidney cancer outcomes using integrated multi-omics data (genomics, transcriptomics, and proteomics).

## Project Overview

This project implements a complete pipeline for:
- Multi-omics data preprocessing and integration
- Feature selection and dimensionality reduction
- Classification and survival prediction models
- Comprehensive model evaluation and visualization

## Dataset Format

### Required CSV Files (place in `data/` directory):

1. **genomics.csv**: Genomic features
   - `sample_id`: Unique sample identifier
   - Feature columns: `gene_1`, `gene_2`, ..., `gene_n`

2. **transcriptomics.csv**: Gene expression data
   - `sample_id`: Unique sample identifier  
   - Feature columns: `transcript_1`, `transcript_2`, ..., `transcript_n`

3. **proteomics.csv**: Protein abundance data
   - `sample_id`: Unique sample identifier
   - Feature columns: `protein_1`, `protein_2`, ..., `protein_n`

4. **labels.csv**: Clinical outcomes
   - `sample_id`: Unique sample identifier
   - `cancer_status`: Binary (tumor/normal)
   - `survival_time`: Time to event (days/months)
   - `event`: Event indicator (1=death, 0=censored)

## Installation

\`\`\`bash
# Clone the repository
git clone <repository-url>
cd kidney_cancer_project

# Install dependencies
pip install -r requirements.txt
\`\`\`

## Usage

### Quick Start
\`\`\`bash
# Run the complete pipeline
python main.py

# Run specific components
python src/preprocess.py
python src/integrate.py
python src/models.py
python src/survival.py
python src/evaluate.py
\`\`\`

### Custom Configuration
Modify parameters in `main.py` or individual modules for:
- Feature selection methods
- Model hyperparameters
- Evaluation metrics
- Visualization options

## Project Structure

\`\`\`
kidney_cancer_project/
├── data/                    # Raw CSV datasets
├── src/                     # Source code modules
│   ├── preprocess.py       # Data preprocessing
│   ├── integrate.py        # Multi-omics integration
│   ├── models.py           # ML models
│   ├── survival.py         # Survival analysis
│   └── evaluate.py         # Model evaluation
├── notebooks/              # Jupyter notebooks for exploration
├── results/                # Output files and plots
├── main.py                 # Main pipeline script
├── requirements.txt        # Dependencies
└── README.md              # This file
\`\`\`

## Models Implemented

### Classification Models:
- Random Forest
- XGBoost
- Logistic Regression
- Deep Learning MLP

### Survival Models:
- Cox Proportional Hazards
- Random Survival Forest

## Evaluation Metrics

- **Classification**: Accuracy, ROC-AUC, Precision, Recall, F1-score
- **Survival**: C-index, Integrated Brier Score
- **Visualizations**: ROC curves, confusion matrices, Kaplan-Meier plots, feature importance

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details.
