import torch
import torch.nn as nn
import torch.nn.functional as F


class DUQRegressor(nn.Module):
    def __init__(self, input_dim, latent_dim=8, n_centroids=20):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )

        self.centroids = nn.Parameter(
            torch.randn(n_centroids, latent_dim)
        )

        self.values = nn.Parameter(
            torch.randn(n_centroids)
        )

    def forward(self, x):
        z = self.encoder(x) # (B, D)
        dist = torch.cdist(z, self.centroids) # (B, K)

        weights = torch.softmax(-dist**2, dim=1)

        y_pred = (weights * self.values).sum(dim=1)
        uncertainty = dist.min(dim=1).values

        return y_pred, uncertainty
