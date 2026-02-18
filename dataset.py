import glob

import lightning as pl
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


class OptionsDataset(Dataset):
    def __init__(self, x, y):
        self.x_data = x
        self.y_data = y

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, item):
        x = self.x_data[item]
        y = self.y_data[item]
        return x, y

class OptionsDataModule(pl.LightningDataModule):
    def __init__(self, folder, sofr_path = None, batch_size = 64):
        super().__init__()
        self.folder = folder
        self.sofr_path = sofr_path
        self.train = None
        self.test = None
        self.val = None
        self.batch_size = batch_size
        self.x_scaler = MinMaxScaler()
        self.y_scaler = MinMaxScaler()
        self.train_sampler = None

    @staticmethod
    def _moneyness_bucket_indices(moneyness: np.ndarray) -> np.ndarray:
        # Bucket 0 intentionally groups deep OTM and deep ITM together.
        buckets = np.zeros_like(moneyness, dtype=np.int64)
        buckets[(moneyness >= 0.90) & (moneyness < 0.97)] = 1  # OTM
        buckets[(moneyness >= 0.97) & (moneyness < 1.03)] = 2  # ATM
        buckets[(moneyness >= 1.03) & (moneyness < 1.10)] = 3  # ITM
        return buckets

    def _build_balanced_train_sampler(self, x_train: pd.DataFrame) -> WeightedRandomSampler | None:
        moneyness = x_train["S"].to_numpy(dtype=np.float64) / np.clip(
            x_train["K"].to_numpy(dtype=np.float64), 1e-8, None
        )
        bucket_indices = self._moneyness_bucket_indices(moneyness)

        bucket_names = ["deep", "otm", "atm", "itm"]
        bucket_counts = np.bincount(bucket_indices, minlength=len(bucket_names))
        available_bucket_count = int(np.sum(bucket_counts > 0))

        if available_bucket_count == 0:
            return None

        per_bucket_weights = np.zeros_like(bucket_counts, dtype=np.float64)
        per_bucket_weights[bucket_counts > 0] = 1.0 / bucket_counts[bucket_counts > 0]
        sample_weights = per_bucket_weights[bucket_indices]

        print(
            "Train moneyness bucket rows -> "
            + " | ".join(f"{name.upper()}: {int(count)}" for name, count in zip(bucket_names, bucket_counts))
        )
        print(f"Train sampler -> balanced across {available_bucket_count} available moneyness buckets per epoch")

        return WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )

    def setup(self, stage: str) -> None:
        files = glob.glob(self.folder + "/*.csv")

        df_list = []
        for file in files:
            df = pd.read_csv(file)

            df_list.append(df)

        final_df = pd.concat(df_list, ignore_index=True)
        final_df["date"] = pd.to_datetime(final_df["date"])
        final_df = final_df.sort_values("date")
        final_df = final_df[final_df["T"] < 365]
        final_df = final_df[final_df["T"] > 14]
        final_df["M"] = final_df["S"] / final_df["K"]
        final_df = final_df[final_df["M"] >= 0.5]
        final_df = final_df[final_df["M"] <= 1.15]

        final_df.dropna(inplace=True)

        if self.sofr_path is not None:
            sofr_df = pd.read_csv(self.sofr_path)
            sofr_df["date"] = pd.to_datetime(sofr_df["date"])
            sofr_df["sofr"] = sofr_df["sofr"] / 100.0
            final_df = pd.merge(final_df, sofr_df, on="date", how="inner")

        final_df["Price"] = final_df["Price"]
        final_df["vix"] = final_df["vix"] / 100
        final_df["T"] = final_df["T"] / 365

        X = final_df.loc[:, ["S", "K", "T", "vix", 'sofr', "dividend_yield", "rate"]]
        Y = final_df.loc[:, "Price"]

        unique_dates = np.sort(final_df["date"].dt.normalize().unique())
        n_dates = len(unique_dates)
        if n_dates < 3:
            raise ValueError(f"Need at least 3 unique dates for train/val/test split, got {n_dates}.")

        train_date_count = max(1, int(n_dates * 0.85))
        val_date_count = max(1, int(n_dates * 0.05))

        if train_date_count + val_date_count >= n_dates:
            if val_date_count > 1:
                val_date_count -= 1
            else:
                train_date_count = max(1, train_date_count - 1)

        train_dates = unique_dates[:train_date_count]
        val_dates = unique_dates[train_date_count: train_date_count + val_date_count]
        test_dates = unique_dates[train_date_count + val_date_count:]

        train_mask = final_df["date"].dt.normalize().isin(train_dates)
        val_mask = final_df["date"].dt.normalize().isin(val_dates)
        test_mask = final_df["date"].dt.normalize().isin(test_dates)

        X_train = X.loc[train_mask]
        Y_train = Y.loc[train_mask]

        X_val = X.loc[val_mask]
        Y_val = Y.loc[val_mask]

        X_test = X.loc[test_mask]
        Y_test = Y.loc[test_mask]

        Y_train = Y_train.values.reshape(-1, 1)
        Y_val = Y_val.values.reshape(-1, 1)
        Y_test = Y_test.values.reshape(-1, 1)

        self.train_sampler = self._build_balanced_train_sampler(X_train)

        self.train = OptionsDataset(torch.tensor(X_train.values, dtype=torch.float32),
                                    torch.tensor(Y_train, dtype=torch.float32))
        self.val = OptionsDataset(torch.tensor(X_val.values, dtype=torch.float32),
                                  torch.tensor(Y_val, dtype=torch.float32))
        self.test = OptionsDataset(torch.tensor(X_test.values, dtype=torch.float32),
                                   torch.tensor(Y_test, dtype=torch.float32))

        print(
            f"Date split -> Train: {len(train_dates)} | Validation: {len(val_dates)} | Test: {len(test_dates)}"
        )
        print(f"Rows -> Train: {len(self.train)} | Validation: {len(self.val)} | Test: {len(self.test)}")

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=self.train_sampler is None,
            sampler=self.train_sampler,
            num_workers=4,
            persistent_workers=True,
        )
    def val_dataloader(self):
         return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=4, persistent_workers=True)
    def test_dataloader(self):
         return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers=4, persistent_workers=True)
