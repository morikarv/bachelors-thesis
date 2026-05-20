import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    # prefer mps when available on macs
    if getattr(torch, 'has_mps', False) and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

class SingleOutputDataset(Dataset):
    def __init__(self, X_np: np.ndarray, y_np: np.ndarray,
                 x_scaler: StandardScaler = None, y_scaler: StandardScaler = None,
                 fit_scaler: bool = False, original_df: pd.DataFrame = None, feature_names=None):
        assert X_np.ndim == 2 and y_np.ndim in (1,2)
        if y_np.ndim == 2:
            y_np = y_np.reshape(-1)
        if fit_scaler:
            x_scaler = StandardScaler().fit(X_np)
            y_scaler = StandardScaler().fit(y_np.reshape(-1,1))
        assert x_scaler is not None and y_scaler is not None, "Scalers must be provided or fit first"

        self.X = torch.tensor(x_scaler.transform(X_np).astype(np.float32), dtype=torch.float32)
        self.y = torch.tensor(y_scaler.transform(y_np.reshape(-1,1)).astype(np.float32)).squeeze(-1)
        self._y_raw = y_np
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.original_df = original_df.reset_index(drop=True) if original_df is not None else None
        self.feature_names = feature_names

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def get_original_row(self, idx):
        if self.original_df is None:
            raise RuntimeError("original_df not provided to dataset")
        return self.original_df.iloc[idx]

def fit_transform_dfs(df_train, df_val, df_test,
                      target_col,
                      cat_cols=None,
                      num_cols=None):

    # Автоматически выделяем числовые признаки
    if num_cols is None:
        num_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in num_cols:
            num_cols.remove(target_col)

    if cat_cols is None:
        cat_cols = []

    needed = num_cols + cat_cols + [target_col]

    df_train = df_train.dropna(subset=needed).reset_index(drop=True)
    df_val   = df_val.dropna(subset=needed).reset_index(drop=True)
    df_test  = df_test.dropna(subset=needed).reset_index(drop=True)

    if len(cat_cols) > 0:
        ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")
        ohe.fit(df_train[cat_cols].astype(str))

        X_train_cat = ohe.transform(df_train[cat_cols].astype(str))
        X_val_cat   = ohe.transform(df_val[cat_cols].astype(str))
        X_test_cat  = ohe.transform(df_test[cat_cols].astype(str))

        cat_feature_names = [
            f"{col}__{cat}"
            for col, cats in zip(cat_cols, ohe.categories_)
            for cat in cats
        ]
    else:
        ohe = None
        X_train_cat = np.zeros((len(df_train), 0))
        X_val_cat   = np.zeros((len(df_val), 0))
        X_test_cat  = np.zeros((len(df_test), 0))
        cat_feature_names = []

    X_train_num = df_train[num_cols].values.astype(np.float32)
    X_val_num   = df_val[num_cols].values.astype(np.float32)
    X_test_num  = df_test[num_cols].values.astype(np.float32)

    X_train = np.hstack([X_train_num, X_train_cat])
    X_val   = np.hstack([X_val_num,   X_val_cat])
    X_test  = np.hstack([X_test_num,  X_test_cat])

    y_train = df_train[target_col].values.astype(np.float32)
    y_val   = df_val[target_col].values.astype(np.float32)
    y_test  = df_test[target_col].values.astype(np.float32)

    x_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))

    feature_names = num_cols + cat_feature_names

    encs = {
        "ohe": ohe,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "num_cols": num_cols,
        "cat_cols": cat_cols
    }

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_names, encs

def load_data(path_csv: str, target_col: str, cat_cols: list = None,
              test_frac=0.2, val_frac=0.5, random_state=42):
    df = pd.read_csv(path_csv)
    df_train, df_temp = train_test_split(df, test_size=test_frac, random_state=random_state)
    df_val, df_test = train_test_split(df_temp, test_size=val_frac, random_state=random_state)

    X_tr, y_tr, X_val, y_val, X_test, y_test, feat_names, encs = fit_transform_dfs(df_train, df_val, df_test,
                                                                                  target_col, cat_cols)
    train_ds = SingleOutputDataset(X_tr, y_tr, fit_scaler=False,
                                   x_scaler=encs['x_scaler'], y_scaler=encs['y_scaler'],
                                   original_df=df_train, feature_names=feat_names)
    val_ds   = SingleOutputDataset(X_val, y_val, x_scaler=encs['x_scaler'], y_scaler=encs['y_scaler'],
                                   original_df=df_val, feature_names=feat_names)
    test_ds  = SingleOutputDataset(X_test, y_test, x_scaler=encs['x_scaler'], y_scaler=encs['y_scaler'],
                                   original_df=df_test, feature_names=feat_names)
    return train_ds, val_ds, test_ds, encs

def load_data_train_test(train_csv, test_csv,
                         target_col, cat_cols=None,
                         val_frac=0.1, random_state=42):

    df_train = pd.read_csv(train_csv)
    df_test  = pd.read_csv(test_csv)

    df_train, df_val = train_test_split(df_train, test_size=val_frac, random_state=random_state)

    X_tr, y_tr, X_val, y_val, X_test, y_test, feat_names, encs = \
        fit_transform_dfs(df_train, df_val, df_test, target_col, cat_cols)

    train_ds = SingleOutputDataset(
        X_tr, y_tr, fit_scaler=False,
        x_scaler=encs["x_scaler"], y_scaler=encs["y_scaler"],
        original_df=df_train, feature_names=feat_names)

    val_ds = SingleOutputDataset(
        X_val, y_val,
        x_scaler=encs["x_scaler"], y_scaler=encs["y_scaler"],
        original_df=df_val, feature_names=feat_names)

    test_ds = SingleOutputDataset(
        X_test, y_test,
        x_scaler=encs["x_scaler"], y_scaler=encs["y_scaler"],
        original_df=df_test, feature_names=feat_names)

    return train_ds, val_ds, test_ds, encs