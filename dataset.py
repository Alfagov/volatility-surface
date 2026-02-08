import glob

import lightning as pl
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset, DataLoader


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

    def setup(self, stage: str) -> None:
        files = glob.glob(self.folder + "/*.csv")

        df_list = []
        for file in files:
            df = pd.read_csv(file)

            df_list.append(df)

        final_df = pd.concat(df_list, ignore_index=True)
        final_df["date"] = pd.to_datetime(final_df["date"])
        final_df = final_df.sort_values("date")

        if self.sofr_path is not None:
            sofr_df = pd.read_csv(self.sofr_path)
            sofr_df["date"] = pd.to_datetime(sofr_df["date"])
            sofr_df["sofr"] = sofr_df["sofr"] / 100.0
            final_df = pd.merge(final_df, sofr_df, on="date", how="inner")

        final_df["Price"] = final_df["Price"] / final_df["K"]
        final_df["vix"] = final_df["vix"] / 100
        final_df["T"] = final_df["T"] / 365

        X = final_df.loc[:, ["S", "K", "T", "vix", 'hv_10', 'hv_14', 'hv_30', 'hv_60', 'hv_91', 'sofr']]
        Y = final_df.loc[:, "Price"]

        train_size = int(len(X) * 0.85)
        val_size = int(len(X) * 0.05)

        X_train = X.iloc[:train_size]
        Y_train = Y.iloc[:train_size]

        X_val = X.iloc[train_size: train_size + val_size]
        Y_val = Y.iloc[train_size: train_size + val_size]

        X_test = X.iloc[train_size + val_size:]
        Y_test = Y.iloc[train_size + val_size:]

        #train_scaled_t_v = self.x_scaler.fit_transform(X_train.loc[:, ["T"]])
        #X_train.loc[:, "T"] = train_scaled_t_v[:, 0]
        #X_train.loc[:, "vix"] = train_scaled_t_v[:, 1]

        #test_scaled_t_v = self.x_scaler.transform(X_test.loc[:, ["T"]])
        #X_test.loc[:, "T"] = test_scaled_t_v[:, 0]
        #X_test.loc[:, "vix"] = test_scaled_t_v[:, 1]

        #val_scaled_t_v = self.x_scaler.transform(X_val.loc[:, ["T"]])
        #X_val.loc[:, "T"] = val_scaled_t_v[:, 0]
        #X_val.loc[:, "vix"] = val_scaled_t_v[:, 1]

        Y_train = Y_train.values.reshape(-1, 1)
        Y_val = Y_val.values.reshape(-1, 1)
        Y_test = Y_test.values.reshape(-1, 1)

        self.train = OptionsDataset(torch.tensor(X_train.values, dtype=torch.float32),
                                    torch.tensor(Y_train, dtype=torch.float32))
        self.val = OptionsDataset(torch.tensor(X_val.values, dtype=torch.float32),
                                  torch.tensor(Y_val, dtype=torch.float32))
        self.test = OptionsDataset(torch.tensor(X_test.values, dtype=torch.float32),
                                   torch.tensor(Y_test, dtype=torch.float32))

        print(f"Train: {len(self.train)} | Validation: {len(self.val)} | Test: {len(self.test)}")

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=4, persistent_workers=True)
    def val_dataloader(self):
         return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=4, persistent_workers=True)
    def test_dataloader(self):
         return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers=4, persistent_workers=True)



