import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import yfinance as yf
from torch import optim
from typing import Tuple, Any
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


class VAE(nn.Module):

    def __init__(self, input_dim = 11, latent_dim = 4):
        super(VAE, self).__init__()

        self.encoder_hidden = nn.Sequential(
            nn.Linear(input_dim, 40),
            nn.ReLU(),
            nn.Linear(40, 40),
            nn.ReLU(),
            nn.Linear(40, 40),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(40, latent_dim)
        self.fc_var = nn.Linear(40, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 40),
            nn.ReLU(),
            nn.Linear(40, 40),
            nn.ReLU(),
            nn.Linear(40, 40),
            nn.ReLU(),
            nn.Linear(40, input_dim),

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


def vae_loss_function(recon_x, x, mu, log_var, beta: float = 1.0):
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    kl_div = kl_div / x.size(0)
    return recon_loss + beta * kl_div


def train_vae(model, train_loader, val_loader, epochs = 100, lr = 0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_x, _ in train_loader:
            optimizer.zero_grad()

            recon_x, mu, log_var = model(batch_x)
            loss = vae_loss_function(recon_x, batch_x, mu, log_var)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, _ in val_loader:
                recon_x, mu, log_var = model(batch_x)
                loss = vae_loss_function(recon_x, batch_x, mu, log_var)
                val_loss += loss.item()

        train_losses.append(train_loss / len(train_loader.dataset))
        val_losses.append(val_loss / len(val_loader.dataset))

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}')

    return train_losses, val_losses


def extract_vae_features(model, data_loader):
    model.eval()
    features = []
    returns = []

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_features = model.extract_features(batch_x)
            features.append(batch_features.numpy())
            returns.append(batch_y.numpy())

    return np.vstack(features), np.concatenate(returns)


def prepare_data(
        data,
        window_size: int,
        forecast_horizon: int,
        train_ratio: str = 0.7
) -> dict[str | Any, StandardScaler | Any]:
    X, y, valid_indices = create_returns_sequence(data, window_size, forecast_horizon)

    n_samples = len(X)
    train_size = int(n_samples * train_ratio)
    val_size = int(n_samples * 0.15)

    train_end = train_size
    val_end = train_end + val_size

    X_train = X[:train_end]
    y_train = y[:train_end]

    X_val = X[train_end:val_end]
    y_val = y[train_end:val_end]

    X_test = X[val_end:]
    y_test = y[val_end:]

    returns_scaler = StandardScaler()
    X_train_scaled = returns_scaler.fit_transform(X_train)
    X_val_scaled = returns_scaler.transform(X_val)
    X_test_scaled = returns_scaler.transform(X_test)

    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = target_scaler.transform(y_val.reshape(-1, 1)).flatten()
    y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten()

    return {
        'X_train': X_train_scaled, 'y_train': y_train_scaled,
        'X_val': X_val_scaled, 'y_val': y_val_scaled,
        'X_test': X_test_scaled, 'y_test': y_test_scaled,
        'returns_scaler': returns_scaler,
        'target_scaler': target_scaler,
        'y_train_original': y_train,
        'y_val_original': y_val,
        'y_test_original': y_test
    }


def main():
    ticker = 'GS'
    window_size = 11
    forecast_horizon = 1

    data = get_data(ticker)
    print(f'There are {data.shape[0]} number of days in the dataset.')

    dataset_TI_df = features(data)

    data_splits = prepare_data(
        dataset_TI_df,
        window_size,
        forecast_horizon
    )

    X_train_tensor = torch.FloatTensor(data_splits['X_train'])
    y_train_tensor = torch.FloatTensor(data_splits['y_train'])
    X_val_tensor = torch.FloatTensor(data_splits['X_val'])
    y_val_tensor = torch.FloatTensor(data_splits['y_val'])
    X_test_tensor = torch.FloatTensor(data_splits['X_test'])
    y_test_tensor = torch.FloatTensor(data_splits['y_test'])

    batch_size = 64
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("\nInitializing VAE...")
    vae_model = VAE(input_dim=window_size, latent_dim=4)

    print("Training VAE on returns sequences...")
    train_losses, val_losses = train_vae(
        vae_model, train_loader, val_loader, epochs=100, lr=0.001
    )

    train_features, train_returns = extract_vae_features(vae_model, train_loader)
    val_features, val_returns = extract_vae_features(vae_model, val_loader)
    test_features, test_returns = extract_vae_features(vae_model, test_loader)

    print(f"Training features shape: {train_features.shape}")
    print(f"Validation features shape: {val_features.shape}")
    print(f"Test features shape: {test_features.shape}")

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('VAE Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.yscale('log')

    plt.subplot(1, 3, 2)
    plt.plot(data_splits['y_train_original'][:100], label='Actual Future Returns')
    plt.title('Sample Future Returns (Training Set)')
    plt.xlabel('Sample')
    plt.ylabel('Returns (%)')
    plt.legend()

    plt.subplot(1, 3, 3)
    with torch.no_grad():
        sample_input = X_train_tensor[:1]
        reconstructed, _, _ = vae_model(sample_input)
        original = data_splits['returns_scaler'].inverse_transform(sample_input.numpy())[0]
        recon = data_splits['returns_scaler'].inverse_transform(reconstructed.numpy())[0]

        plt.plot(original, 'o-', label='Original Returns', alpha=0.7)
        plt.plot(recon, 's-', label='Reconstructed Returns', alpha=0.7)
        plt.title('Sample VAE Reconstruction')
        plt.xlabel('Day')
        plt.ylabel('Returns (%)')
        plt.legend()

    plt.tight_layout()
    plt.show()

    torch.save(vae_model.state_dict(), f'{ticker}_returns_vae_model.pth')

    prediction_data = {
        'train_features': train_features,
        'train_returns': data_splits['y_train_original'],
        'val_features': val_features,
        'val_returns': data_splits['y_val_original'],
        'test_features': test_features,
        'test_returns': data_splits['y_test_original'],
        'returns_scaler': data_splits['returns_scaler'],
        'target_scaler': data_splits['target_scaler']
    }

    np.savez(f'{ticker}_returns_vae_features.npz', **prediction_data)


if __name__ == '__main__':
    main()