import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from scipy.optimize import brentq
from scipy.stats import norm

from dataset import OptionsDataModule
from model import OptionNetModule


def black_scholes_price(cp_flag: str, s: float, k: float, t_years: float, r: float, sigma: float) -> float:
    if t_years <= 0.0 or sigma <= 0.0 or s <= 0.0 or k <= 0.0:
        return np.nan
    d1 = (np.log(s / k) + (r + 0.5 * sigma**2) * t_years) / (sigma * np.sqrt(t_years))
    d2 = d1 - sigma * np.sqrt(t_years)
    if cp_flag == "P":
        return k * np.exp(-r * t_years) * norm.cdf(-d2) - s * norm.cdf(-d1)
    return s * norm.cdf(d1) - k * np.exp(-r * t_years) * norm.cdf(d2)


def implied_volatility(cp_flag: str, price: float, s: float, k: float, t_years: float, r: float) -> float:
    def obj_func(sigma: float) -> float:
        return black_scholes_price(cp_flag, s, k, t_years, r, sigma) - price

    if any(not np.isfinite(x) for x in [price, s, k, t_years, r]) or price <= 0.0:
        return np.nan

    low, high = 1e-4, 5.0
    if obj_func(low) * obj_func(high) > 0:
        return np.nan
    try:
        return float(brentq(obj_func, low, high))
    except Exception:
        return np.nan


def choose_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def expected_raw_feature_count(model: OptionNetModule) -> int:
    # model_input = [m, t, rest], where m = S/K, rest starts at index 3 in raw x.
    # First linear in_features = 2 + (raw_feature_count - 3) => raw_feature_count = in_features + 1
    first_linear = model.model.model[0]
    return int(first_linear.in_features) + 1


def build_input_frame(df: pd.DataFrame, raw_feature_count: int, default_vix: float | None) -> pd.DataFrame:
    working = df.copy()

    if "spot_price" not in working.columns:
        raise ValueError("Surface CSV must contain 'spot_price' for comparison.")
    if "impl_strike" not in working.columns:
        raise ValueError("Surface CSV must contain 'impl_strike'.")
    if "days" not in working.columns:
        raise ValueError("Surface CSV must contain 'days'.")

    working["vix"] = working["vix"].fillna(default_vix) if "vix" in working.columns else default_vix
    if working["vix"].isna().any():
        raise ValueError("Missing VIX values. Provide --default-vix or include 'vix' in CSV.")

    features = pd.DataFrame(
        {
            "S": working["spot_price"].astype(float),
            "K": working["impl_strike"].astype(float),
            "T": working["days"].astype(float),
            "vix": working["vix"].astype(float),
        }
    )

    if raw_feature_count == 4:
        return features

    if raw_feature_count == 9:
        for hv_col in ["hv_10", "hv_14", "hv_30", "hv_60", "hv_91"]:
            if hv_col not in working.columns:
                working[hv_col] = np.nan
            working[hv_col] = working[hv_col].astype(float)
            working[hv_col] = working[hv_col].fillna(working[hv_col].median())
            working[hv_col] = working[hv_col].fillna(0.0)
            features[hv_col] = working[hv_col]
        return features

    raise ValueError(
        f"Unsupported model raw input feature count={raw_feature_count}. "
        "This script currently supports raw x with 4 or 9 columns."
    )


def save_surface_plot(
    df: pd.DataFrame,
    value_col: str,
    output_path: Path,
    title: str,
    z_label: str,
) -> None:
    pivot = (
        df.pivot_table(index="days", columns="delta", values=value_col, aggfunc="mean")
        .sort_index()
        .sort_index(axis=1)
    )
    fig = go.Figure(
        data=[
            go.Surface(
                x=pivot.columns.values,
                y=pivot.index.values,
                z=pivot.values,
                colorscale="Viridis",
                hovertemplate=(
                    "Delta: %{x:.3f}<br>"
                    "Days: %{y:.0f}<br>"
                    f"{z_label}: %{{z:.5f}}<extra></extra>"
                ),
            )
        ]
    )
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="Delta",
            yaxis_title="Days to Expiration",
            zaxis_title=z_label,
            aspectratio=dict(x=1, y=1, z=0.6),
        ),
        width=1000,
        height=700,
    )
    fig.write_html(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare model IV surface vs WRDS downloaded surface.")
    parser.add_argument("--surface-csv", required=True, help="CSV from downloader surface mode.")
    parser.add_argument("--checkpoint", required=True, help="Lightning checkpoint path.")
    parser.add_argument("--train-data-dir", required=True, help="Folder used to fit scalers.")
    parser.add_argument("--cp-flag", choices=["C", "P"], default="C")
    parser.add_argument("--date", default=None, help="Optional date filter (YYYY-MM-DD). Defaults to latest date.")
    parser.add_argument("--risk-free-rate", type=float, default=0.04)
    parser.add_argument("--default-vix", type=float, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--output-dir", default="./data/comparisons")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.surface_csv)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["cp_flag"] == args.cp_flag].copy()
    if df.empty:
        raise ValueError(f"No rows found for cp_flag={args.cp_flag}.")

    target_date = pd.to_datetime(args.date) if args.date else df["date"].max()
    daily = df[df["date"] == target_date].copy()
    if daily.empty:
        raise ValueError(f"No rows found for date={target_date.date()} and cp_flag={args.cp_flag}.")

    data_module = OptionsDataModule(args.train_data_dir, batch_size=128)
    data_module.setup("fit")

    model = OptionNetModule.load_from_checkpoint(args.checkpoint)
    model.eval()

    device = choose_device(args.device)
    model.to(device)

    raw_feature_count = expected_raw_feature_count(model)
    features = build_input_frame(daily, raw_feature_count=raw_feature_count, default_vix=args.default_vix)

    scaled_t_v = data_module.x_scaler.transform(features.loc[:, ["T", "vix"]])
    features_scaled = features.copy()
    features_scaled["T"] = scaled_t_v[:, 0]
    features_scaled["vix"] = scaled_t_v[:, 1]

    x_input = torch.tensor(features_scaled.values, dtype=torch.float32, device=device)

    pred_scaled, _ = model(x_input)
    pred_prices = data_module.y_scaler.inverse_transform(pred_scaled.detach().cpu().numpy()).flatten()

    t_years = daily["days"].astype(float).values / 365.0
    cp_flags = daily["cp_flag"].astype(str).values
    s_vals = daily["spot_price"].astype(float).values
    k_vals = daily["impl_strike"].astype(float).values

    model_iv = np.array(
        [
            implied_volatility(cp, px, s, k, t, args.risk_free_rate)
            for cp, px, s, k, t in zip(cp_flags, pred_prices, s_vals, k_vals, t_years, strict=True)
        ]
    )

    compared = daily.copy()
    compared["model_price"] = pred_prices
    compared["model_impl_volatility"] = model_iv
    compared["iv_error"] = compared["model_impl_volatility"] - compared["impl_volatility"]
    compared["abs_iv_error"] = compared["iv_error"].abs()

    valid = compared[np.isfinite(compared["impl_volatility"]) & np.isfinite(compared["model_impl_volatility"])].copy()
    if valid.empty:
        raise ValueError("No valid points after implied-vol inversion. Try another date or risk-free rate.")

    rmse = float(np.sqrt(np.mean((valid["iv_error"]) ** 2)))
    mae = float(np.mean(valid["abs_iv_error"]))
    mean_abs_pct = float(np.mean(valid["abs_iv_error"] / valid["impl_volatility"].clip(lower=1e-8)) * 100.0)

    stamp = f"{str(valid['secid'].iloc[0])}_{args.cp_flag}_{target_date.date()}"
    compared_csv = output_dir / f"surface_comparison_{stamp}.csv"
    metrics_json = output_dir / f"surface_metrics_{stamp}.json"
    market_html = output_dir / f"market_surface_{stamp}.html"
    model_html = output_dir / f"model_surface_{stamp}.html"
    error_html = output_dir / f"error_surface_{stamp}.html"

    compared.to_csv(compared_csv, index=False)
    save_surface_plot(valid, "impl_volatility", market_html, "WRDS Market Surface (Delta x Days)", "Market IV")
    save_surface_plot(
        valid,
        "model_impl_volatility",
        model_html,
        "Model-Implied Surface (Delta x Days)",
        "Model IV",
    )
    save_surface_plot(valid, "iv_error", error_html, "Model - Market IV Error Surface", "IV Error")

    metrics = {
        "secid": str(valid["secid"].iloc[0]),
        "cp_flag": args.cp_flag,
        "date": str(target_date.date()),
        "points_total": int(len(compared)),
        "points_valid": int(len(valid)),
        "rmse_iv": rmse,
        "mae_iv": mae,
        "mape_percent_iv": mean_abs_pct,
        "outputs": {
            "compared_csv": str(compared_csv),
            "market_surface_html": str(market_html),
            "model_surface_html": str(model_html),
            "error_surface_html": str(error_html),
        },
    }
    metrics_json.write_text(json.dumps(metrics, indent=2))

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
