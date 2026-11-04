from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns


def resolve_data_path(file_path):
    """Resolve project-relative paths regardless of the current working directory."""
    path = Path(file_path)
    if path.is_absolute():
        return path

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    candidates = [script_dir / path, project_root / path]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return (script_dir / path).resolve()


def perform_rfm_segmentation(file_path="../Data_set/cleaned_retail_sales.csv"):
    """
    Computes RFM (Recency, Frequency, Monetary) metrics from transactions
    and applies K-Means clustering to segment customers.
    """
    dataset_path = resolve_data_path(file_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at '{dataset_path}'.")

    df = pd.read_csv(dataset_path)
    df['OrderDate'] = pd.to_datetime(df['OrderDate'])
    
    potential_id_cols = ['CustomerID', 'Customer ID', 'Customer_ID', 'Client_ID']
    customer_col = next((col for col in potential_id_cols if col in df.columns), None)
    
    if not customer_col:
        print("Notice: No explicit 'CustomerID' column found in cleaned data.")
        print("Generating synthetic unique Customer IDs for RFM clustering demonstration...")
        np.random.seed(42)
        df['CustomerID'] = [f"CUST-{np.random.randint(10000, 10500):05d}" for _ in range(len(df))]
        customer_col = 'CustomerID'
        
    snapshot_date = df['OrderDate'].max() + pd.Timedelta(days=1)
    
    rfm_df = df.groupby(customer_col).agg(
        Recency=('OrderDate', lambda x: (snapshot_date - x.max()).days),
        Frequency=('OrderID', 'nunique') if 'OrderID' in df.columns else ('OrderDate', 'count'),
        Monetary=('Calculated_Revenue', 'sum')
    ).reset_index()
    
    n_samples = len(rfm_df)
    print(f"Calculated RFM metrics for {n_samples} unique customer entities.")
    
    if n_samples < 2:
        print("Error: Not enough unique customer samples to perform clustering.")
        return rfm_df

    n_clusters = min(4, n_samples)
    rfm_features = ['Recency', 'Frequency', 'Monetary']
    rfm_log = np.log1p(rfm_df[rfm_features])
    
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm_df['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    cluster_summary = rfm_df.groupby('Cluster')[rfm_features].mean()
    print("\n--- Cluster Centroids (Behavioral Profiles) ---")
    print(cluster_summary)
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=rfm_df, 
        x='Frequency', 
        y='Monetary', 
        hue='Cluster', 
        palette='viridis', 
        s=80, 
        alpha=0.8,
        legend='full'
    )
    plt.title(f'Customer Segments via K-Means Clustering (k={n_clusters})', fontsize=13, fontweight='bold')
    plt.xlabel('Frequency (Total Orders)', fontsize=10)
    plt.ylabel('Monetary Value ($)', fontsize=10)
    plt.savefig('customer_segments.png', dpi=300, bbox_inches='tight')
    print("Customer segments plot saved to 'customer_segments.png'")
    plt.show()
    
    return rfm_df

if __name__ == "__main__":
    rfm_results = perform_rfm_segmentation()