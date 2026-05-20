import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.parametrizations import spectral_norm
from models.utils import get_device

class DNN(nn.Module):
    def __init__(self, in_dim, hidden_dims=[64,32], dropout=0.1, use_spectral_norm=False):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden_dims:
            lin = nn.Linear(d, h)
            if use_spectral_norm:
                lin = spectral_norm(lin)
            layers += [lin, nn.ReLU(), nn.Dropout(dropout)]
            d = h

        out = nn.Linear(d, 1)
        if use_spectral_norm:
            out = spectral_norm(out)
        layers.append(out)

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_simple_dnn(train_ds, val_ds, hidden_dims=[64,32], dropout=0.1, lr=1e-3,
                     epochs=50, batch_size=64, device=None, verbose=False,
                     use_spectral_norm=False):
    device = device or get_device()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = DNN(
        in_dim=train_ds.X.shape[1],
        hidden_dims=hidden_dims,
        dropout=dropout,
        use_spectral_norm=use_spectral_norm
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    history = {'train_loss': [], 'val_loss': []}
    best_state = None
    best_val = float('inf')

    for epoch in range(epochs):
        model.train()
        total=0.0; cnt=0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total += loss.item()*xb.size(0); cnt += xb.size(0)

        train_loss = total/cnt

        model.eval()
        total=0.0; cnt=0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                total += criterion(pred, yb).item()*xb.size(0); cnt += xb.size(0)

        val_loss = total/cnt
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, history