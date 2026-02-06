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
    def __init__(self, folder, batch_size = 64):
        super().__init__()
        self.folder = folder
        self.train = None
        self.test = None
        self.val = None
        self.batch_size = batch_size
        self.scaler = MinMaxScaler()
        self.y_scaler = MinMaxScaler()

    def setup(self, stage: str) -> None:
        files = glob.glob(self.folder + "/*.csv")

        df_list = []
        for file in files:
            df = pd.read_csv(file)

            df_list.append(df)

        final_df = pd.concat(df_list, ignore_index=True)
        final_df = final_df[final_df["Price"] > 20]
        final_df = final_df[final_df["T"] > 30]
        final_df = final_df[final_df["T"] < 366]
        X = final_df.loc[:, ["S", "K", "T", "vix"]]
        Y = final_df.loc[:, "Price"]

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.9, random_state=42, shuffle=True)

        t_v_transf = self.scaler.fit_transform(X_train.loc[:, ["T", "vix"]])
        X_train["T"] = t_v_transf[:, 0]
        X_train["vix"] = t_v_transf[:, 1]

        t_v_transf_test = self.scaler.transform(X_test.loc[:, ["T", "vix"]])
        X_test["T"] = t_v_transf_test[:, 0]
        X_test["vix"] = t_v_transf_test[:, 1]

        X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, train_size=0.7, random_state=42, shuffle=True)

        train = X_train.values.astype(np.float32) #self.scaler.fit_transform(X_train).astype(np.float32)
        test = X_test.values.astype(np.float32)#self.scaler.transform(X_test).astype(np.float32)
        val = X_val.values.astype(np.float32)#self.scaler.transform(X_val).astype(np.float32)

        Y_train = self.y_scaler.fit_transform(Y_train.values.reshape(-1, 1))
        Y_test = self.y_scaler.transform(Y_test.values.reshape(-1, 1))
        Y_val = self.y_scaler.transform(Y_val.values.reshape(-1, 1))

        print(f"Train: {len(train)} | Validation: {len(val)} | Test: {len(test)}")

        X_tr = torch.from_numpy(train).to(torch.float32)
        y_tr = torch.from_numpy(Y_train).to(torch.float32)
        X_te = torch.from_numpy(test).to(torch.float32)
        y_te = torch.from_numpy(Y_test).to(torch.float32)
        X_va = torch.from_numpy(val).to(torch.float32)
        y_va = torch.from_numpy(Y_val).to(torch.float32)

        self.train = OptionsDataset(X_tr, y_tr)
        self.test = OptionsDataset(X_te, y_te)
        self.val = OptionsDataset(X_va, y_va)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=4, persistent_workers=True)
    def val_dataloader(self):
         return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=4, persistent_workers=True)
    def test_dataloader(self):
         return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers=4, persistent_workers=True)



