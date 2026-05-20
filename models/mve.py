import os
import numpy as np
import pandas as pd
import tqdm
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

class MLP_MVE(nn.Module):
    def __init__(self, in_dim, hidden_dims=[256,128], dropout=0.1):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        self.backbone = nn.Sequential(*layers)
        self.mu_head = nn.Linear(d, 1)
        self.logvar_head = nn.Linear(d, 1)

    def forward(self, x):
        h = self.backbone(x)
        mu = self.mu_head(h).squeeze(-1)
        logvar = self.logvar_head(h).squeeze(-1)
        return mu, logvar
    
def train_mve_ensemble(train_ds, val_ds, ensemble_size=5,
                       hidden_dims=[256,128], dropout=0.1,
                       lr=1e-3, epochs=20, batch_size=128, device=None, verbose=False):
    device = device or get_device()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    models = []
    histories = []
    criterion = nn.GaussianNLLLoss()
    for e in tqdm.tqdm(range(ensemble_size)):
        m = MLP_MVE(train_ds.X.shape[1], hidden_dims, dropout).to(device)
        optimizer = optim.Adam(m.parameters(), lr=lr)
        best_val = float('inf'); best_state = None
        history = {'train_loss': [], 'val_loss': []}
        for epoch in range(epochs):
            m.train()
            total = 0.0; cnt = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                mu, logvar = m(xb)
                var = torch.exp(logvar).clamp(min=1e-6)
                loss = criterion(mu, yb, var)
                loss.backward(); optimizer.step()
                total += loss.item() * xb.size(0); cnt += xb.size(0)
            train_loss = total / cnt
            # val
            m.eval()
            total = 0.0; cnt = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    mu, logvar = m(xb)
                    var = torch.exp(logvar).clamp(min=1e-6)
                    total += criterion(mu, yb, var).item() * xb.size(0)
                    cnt += xb.size(0)
            val_loss = total / cnt
            history['train_loss'].append(train_loss); history['val_loss'].append(val_loss)
            if val_loss < best_val:
                best_val = val_loss; best_state = m.state_dict()
        m.load_state_dict(best_state)
        models.append(m); histories.append(history)
        if verbose:
            print(f"Trained ensemble member {e+1}/{ensemble_size}, best_val={best_val:.6f}")
    return models, histories

def predict_mve_ensemble(models, X_np, device=None):
    device = device or get_device()
    X = torch.tensor(X_np.astype(np.float32)).to(device)
    mu_list = []
    var_list = []
    for m in models:
        m.eval()
        with torch.no_grad():
            mu, logvar = m(X)
            mu_list.append(mu.cpu().numpy())
            var_list.append(np.exp(logvar.cpu().numpy()))
    mu_stack = np.stack(mu_list, axis=0)
    var_stack = np.stack(var_list, axis=0)
    mean = mu_stack.mean(axis=0)
    epistemic = mu_stack.var(axis=0)
    aleatoric = var_stack.mean(axis=0)
    total_var = epistemic + aleatoric
    std = np.sqrt(total_var)
    return mean, std, mu_stack, epistemic, aleatoric