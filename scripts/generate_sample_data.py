"""
Generate sample multi-omics datasets for testing the pipeline.
Creates synthetic genomics, transcriptomics, proteomics, and labels data.
"""

import pandas as pd
import numpy as np
import os

def generate_sample_data(n_samples=200, output_dir='data/'):
    """
    Generate synthetic multi-omics datasets for testing.
    
    Args:
        n_samples (int): Number of samples to generate
        output_dir (str): Output directory for CSV files
    """
    np.random.seed(42)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate sample IDs
    sample_ids = [f"SAMPLE_{i:03d}" for i in range(1, n_samples + 1)]
    
    # Generate labels first (to create realistic correlations)
    cancer_status = np.random.choice(['tumor', 'normal'], n_samples, p=[0.6, 0.4])
    survival_time = np.random.exponential(scale=500, size=n_samples)  # Days
    event = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])  # 70% event rate
    
    # Create labels DataFrame
    labels_df = pd.DataFrame({
        'sample_id': sample_ids,
        'cancer_status': cancer_status,
        'survival_time': survival_time.astype(int),
        'event': event
    })
    
    # Generate genomics data (SNPs, CNVs, etc.)
    n_genomic_features = 1000
    genomics_data = np.random.normal(0, 1, (n_samples, n_genomic_features))
    
    # Add some correlation with cancer status
    tumor_mask = cancer_status == 'tumor'
    genomics_data[tumor_mask, :100] += np.random.normal(0.5, 0.2, (tumor_mask.sum(), 100))
    
    genomics_df = pd.DataFrame(
        genomics_data,
        columns=[f"gene_{i:04d}" for i in range(n_genomic_features)]
    )
    genomics_df.insert(0, 'sample_id', sample_ids)
    
    # Generate transcriptomics data (gene expression)
    n_transcript_features = 2000
    transcriptomics_data = np.random.lognormal(0, 1, (n_samples, n_transcript_features))
    
    # Add correlation with cancer status
    transcriptomics_data[tumor_mask, :200] *= np.random.uniform(1.5, 3.0, (tumor_mask.sum(), 200))
    
    transcriptomics_df = pd.DataFrame(
        transcriptomics_data,
        columns=[f"transcript_{i:04d}" for i in range(n_transcript_features)]
    )
    transcriptomics_df.insert(0, 'sample_id', sample_ids)
    
    # Generate proteomics data (protein abundance)
    n_protein_features = 500
    proteomics_data = np.random.gamma(2, 2, (n_samples, n_protein_features))
    
    # Add correlation with cancer status
    proteomics_data[tumor_mask, :50] *= np.random.uniform(0.5, 2.0, (tumor_mask.sum(), 50))
    
    proteomics_df = pd.DataFrame(
        proteomics_data,
        columns=[f"protein_{i:04d}" for i in range(n_protein_features)]
    )
    proteomics_df.insert(0, 'sample_id', sample_ids)
    
    # Add some missing values to make it realistic
    for df in [genomics_df, transcriptomics_df, proteomics_df]:
        # Randomly set 5% of values to NaN
        mask = np.random.random(df.iloc[:, 1:].shape) < 0.05
        df.iloc[:, 1:] = df.iloc[:, 1:].mask(mask)
    
    # Save datasets
    labels_df.to_csv(f"{output_dir}/labels.csv", index=False)
    genomics_df.to_csv(f"{output_dir}/genomics.csv", index=False)
    transcriptomics_df.to_csv(f"{output_dir}/transcriptomics.csv", index=False)
    proteomics_df.to_csv(f"{output_dir}/proteomics.csv", index=False)
    
    print(f"Sample datasets generated successfully!")
    print(f"Files saved to {output_dir}:")
    print(f"  - labels.csv: {labels_df.shape}")
    print(f"  - genomics.csv: {genomics_df.shape}")
    print(f"  - transcriptomics.csv: {transcriptomics_df.shape}")
    print(f"  - proteomics.csv: {proteomics_df.shape}")
    
    # Print data summary
    print(f"\nData Summary:")
    print(f"  - Total samples: {n_samples}")
    print(f"  - Cancer status distribution: {labels_df['cancer_status'].value_counts().to_dict()}")
    print(f"  - Event rate: {labels_df['event'].mean():.2%}")
    print(f"  - Mean survival time: {labels_df['survival_time'].mean():.1f} days")

if __name__ == "__main__":
    generate_sample_data()
