import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import yfinance as yf
from typing import Tuple
import matplotlib.pyplot as plt


class VAE(nn.Module):

    def __init__(self, input_dim = 11, latent_dim = 4):
        super(VAE, self).__init__()

        self.encoder_hidden = nn.Sequential(
            nn.Linear(input_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(400, latent_dim)
        self.fc_var = nn.Linear(400, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, input_dim),

        )

    def encode(self, x):
        h = self.encoder_hidden(x)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstructed = self.decode(z)
        return reconstructed, mu, log_var

    def extract_features(self, x):
        mu, _ = self.encode(x)
        return mu


def get_data(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period='max')['Close']
    df = df.rename({ticker: 'price'}, axis = 'columns')

    return df


def features(dataset):
    dataset['returns'] = ((dataset['price'] / dataset['price'].shift(1)) - 1) * 100

    return dataset


def create_returns_sequence(data: pd.DataFrame, window_size: int, forecast_horizon: int) -> Tuple:
    returns = data['returns'].values

    X = []
    y = []
    valid_indices = []

    for i in range(window_size, len(returns) - forecast_horizon + 1):
        returns_window = returns[i - window_size:i]

        future_return = returns[i + forecast_horizon - 1]

        if not (np.isnan(returns_window).any() or np.isnan(future_return)):
            X.append(returns_window)
            y.append(future_return)
            valid_indices.append(i)

    return np.array(X), np.array(y), valid_indices


def prepare_data(data, window_size: int, forecast_horizon: int, train_ratio: str = 0.7) -> pd.DataFrame:
    X, y, valid_indices = create_returns_sequence(data, window_size, forecast_horizon)

    n_samples = len(X)



def main():
    ticker = 'GS'
    window_size = 11
    forecast_horizon = 1

    data = get_data(ticker)
    print(f'There are {data.shape[0]} number of days in the dataset.')

    dataset_TI_df = features(data)

    data_splits = ...


if __name__ == '__main__':
    main()