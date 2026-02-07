import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm
from scipy.optimize import brentq
import warnings

from dataset import OptionsDataModule
from model import OptionNetModule

warnings.filterwarnings("ignore")

# --- Constants ---
CHECKPOINT_PATH = "logs/my_experiment/version_57/checkpoints/epoch=19-step=27720.ckpt"
DATA_DIR = "./data/108105"
BATCH_SIZE = 128
RISK_FREE_RATE = 0.04  # Assumed r

def black_scholes_call_price(S, K, T, r, sigma):
    """Calculates Black-Scholes Call Option Price"""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def calculate_implied_volatility(price, S, K, T, r):
    """Calculates Implied Volatility using Brent's method"""
    def obj_func(sigma):
        return black_scholes_call_price(S, K, T, r, sigma) - price

    # Bounds for sigma
    low, high = 1e-4, 5.0
    
    # Check if solution is within bounds
    if obj_func(low) * obj_func(high) > 0:
         return np.nan # Return NaN if no root in range

    try:
        iv = brentq(obj_func, low, high)
        return iv
    except Exception:
        return np.nan

def main():
    print("Loading Data Module...")
    # Initialize DataModule to get the fitted scalers
    data_module = OptionsDataModule(DATA_DIR, batch_size=BATCH_SIZE)
    data_module.setup("fit") # This triggers fitting of scalers
    
    print(f"Loading Model from {CHECKPOINT_PATH}...")
    model = OptionNetModule.load_from_checkpoint(CHECKPOINT_PATH)
    model.eval()
    
    # --- Generate Grid ---
    print("Generating Grid...")
    # Fixed parameters
    S_fixed = 5868.55 # Example Spot Price
    vix_fixed = 17.93 # Example VIX
    
    # Variable parameters
    k_min = 0.8 * S_fixed
    k_max = 1.2 * S_fixed
    K_values = np.linspace(k_min, k_max, 30) # 30 steps for Strike
    
    T_days = np.linspace(30, 730, 30) # 30 steps for Time (days)
    T_years = T_days / 365.0 # T in years for BS
    
    # Create meshgrid
    K_grid, T_grid = np.meshgrid(K_values, T_years)
    
    # Flatten for model input
    K_flat = K_grid.flatten()
    T_flat = T_grid.flatten()
    
    # Prepare Model Inputs
    # Inputs: S, K, T, vix (depending on dataset columns used in training)
    # dataset.py: X = final_df.loc[:, ["S", "K", "T", "vix"]]
    # T in dataset seems to be in days based on head command output (260, 351, etc.)
    # So we should use T_days_flat for the model input if dataset used raw days.
    # dataset.py: X_train["T"] = train_scaled_t_v[:, 0] -> Scaled!
    
    T_days_flat = (T_flat * 365.0).reshape(-1, 1)
    vix_flat = np.full((len(T_flat), 1), vix_fixed)
    
    # Scale T and VIX using the data_module's scaler
    # expected input for scaler is a DF or array with columns ["T", "vix"]
    import pandas as pd
    to_scale = pd.DataFrame({'T': T_days_flat.flatten(), 'vix': vix_flat.flatten()})
    
    scaled_t_v = data_module.x_scaler.transform(to_scale)
    T_scaled = scaled_t_v[:, 0]
    vix_scaled = scaled_t_v[:, 1]
    
    # Construct tensor input X
    # Model expects: S, K, T, vix (based on dataset.py columns)
    # BUT, wait, model.py forward:
    # s = x[:, 0:1]
    # k = x[:, 1:2]
    # t = x[:, 2:3]
    # rest = x[:, 3:] (vix)
    
    S_tensor = torch.full((len(K_flat), 1), S_fixed, dtype=torch.float32)
    K_tensor = torch.tensor(K_flat, dtype=torch.float32).reshape(-1, 1)
    T_tensor = torch.tensor(T_scaled, dtype=torch.float32).reshape(-1, 1)
    vix_tensor = torch.tensor(vix_scaled, dtype=torch.float32).reshape(-1, 1)
    
    # Concatenate [S, K, T, vix]
    x_input = torch.cat([S_tensor, K_tensor, T_tensor, vix_tensor], dim=1)
    
    # Predict
    print("Predicting Prices...")
    with torch.no_grad():
        preds, _ = model(x_input.to("mps"))
        
    # Inverse transform predictions?
    # dataset.py: Y_train = self.y_scaler.fit_transform(...)
    # So we need to inverse transform the output
    predicted_prices_scaled = preds.cpu().numpy()
    predicted_prices = data_module.y_scaler.inverse_transform(predicted_prices_scaled).flatten()
    
    # --- Calculate Implied Volatility ---
    print("Calculating Implied Volatility...")
    iv_results = []
    
    for i in range(len(predicted_prices)):
        price = predicted_prices[i]
        K_val = K_flat[i]
        T_val = T_flat[i] # In Years
        
        iv = calculate_implied_volatility(price, S_fixed, K_val, T_val, RISK_FREE_RATE)
        iv_results.append(iv)
        
    iv_grid = np.array(iv_results).reshape(K_grid.shape)
    
    # --- Plotting ---
    print("Plotting Surface...")
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Normalize for colormap
    # Filter nan for plotting if necessary, but surface plot handles some
    
    surf = ax.plot_surface(K_grid, T_grid, iv_grid, cmap='viridis', edgecolor='none')
    
    ax.set_xlabel('Strike Price (K)')
    ax.set_ylabel('Time to Maturity (Years)')
    ax.set_zlabel('Implied Volatility')
    ax.set_title(f'Volatility Surface (S={S_fixed}, VIX={vix_fixed})')
    
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    plt.savefig('volatility_surface.png')
    print("Volatility Surface saved to 'volatility_surface.png'")

if __name__ == "__main__":
    main()
