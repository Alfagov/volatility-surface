
import pandas as pd
import numpy as np
import torch
from dataset import OptionsDataModule

# Constants
DATA_DIR = "./data/108105"
BATCH_SIZE = 128

def analyze_data():
    print("Loading Data Module...")
    data_module = OptionsDataModule(DATA_DIR, batch_size=BATCH_SIZE)
    data_module.setup("fit")
    
    # Access test data
    # data_module.test is an OptionsDataset
    # x_data is a tensor
    
    x_test = data_module.test.x_data.numpy()
    y_test = data_module.test.y_data.numpy()
    
    # Columns: S, K, T (scaled), vix (scaled)
    S = x_test[:, 0]
    K = x_test[:, 1]
    T_scaled = x_test[:, 2]
    vix_scaled = x_test[:, 3]
    
    # Inverse transform T and vix
    # Scaler expects (n_samples, 2)
    to_inverse = np.column_stack((T_scaled, vix_scaled))
    inversed = data_module.x_scaler.inverse_transform(to_inverse)
    T = inversed[:, 0]
    vix = inversed[:, 1]
    
    # Inverse transform prices
    prices = data_module.y_scaler.inverse_transform(y_test).flatten()
    
    df = pd.DataFrame({
        'S': S,
        'K': K,
        'T': T,
        'vix': vix,
        'Price': prices
    })
    
    print("Dataframe constructed. Shape:", df.shape)
    print(df.head())
    
    # Group by S and count
    s_counts = df['S'].value_counts()
    print("\nTop 10 Spot Prices by count:")
    print(s_counts.head(10))
    
    # Pick the top S
    top_s = s_counts.index[0]
    subset = df[df['S'] == top_s]
    
    print(f"\nSubset for S={top_s}:")
    print(subset.head())
    print("Unique VIX in subset:", subset['vix'].unique())
    print("Unique T in subset:", subset['T'].unique())
    print("Number of points:", len(subset))
    
if __name__ == "__main__":
    analyze_data()
