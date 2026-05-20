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
from models.utils import get_device
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# SWAG (swa_gaussian)
from swa_gaussian.swag.posteriors import SWAG

class MLP(nn.Module):
    def __init__(self, in_dim, hidden=[50, 20]):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers.append(nn.Linear(d, h))
            layers.append(nn.ReLU())
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x).squeeze(-1)

def make_model(in_dim = 17):
    return MLP(in_dim=in_dim, hidden=[50, 20])

def train_swag_model(X_train_np, y_train_np,
                     in_dim,
                     hidden_dims=[128,64],
                     lr=1e-3,
                     epochs=150,
                     swag_start=80,
                     batch_size=64,
                     device='cpu'):
    # создаём базовую модель
    device = torch.device(device)

    batch_size = 64
    lr = 1e-3
    epochs = 100
    swag_start = 50

    ds = TensorDataset(torch.tensor(X_train_np), torch.tensor(y_train_np))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    base_model = make_model(X_train_np.shape[1])
    optimizer = optim.Adam(base_model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    swag_model = SWAG(make_model, max_num_models=20)

    for ep in tqdm.tqdm(range(epochs)):
        base_model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = base_model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
        if ep >= swag_start:
            swag_model.collect_model(base_model)

    return swag_model

def predict_swag_model(swag_model, X_np, n_samples=50, device='cpu'):
    swag_model.to(device)
    swag_model.eval()

    n_samples = 30
    preds = []
    for _ in range(n_samples):
        swag_model.sample()
        with torch.no_grad():
            preds.append(swag_model(X_np.to('mps')).cpu().numpy())
    # return preds
    preds = np.stack(preds, axis=0)
    mean_pred = preds.mean(axis=0)
    std_pred  = preds.std(axis=0)

    arr = np.stack(preds, axis=0)
    return arr, mean_pred, std_pred