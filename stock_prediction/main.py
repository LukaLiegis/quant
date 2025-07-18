import torch
import numpy as np
import pandas as pd
import torch.nn as nn
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


class LSTMGenerator(nn.Module):
    def __init__(self, input_dim, hidden_dim = 128, num_layers = 2):
        super(LSTMGenerator, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size = input_dim, hidden_size = hidden_dim, num_layers = num_layers, batch_first = True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        lstm_out = self.dropout(lstm_out)
        output = self.fc(lstm_out)
        return output


class CNNDiscriminator(nn.Module):
    def __init__(self, input_dim, seq_length):
        super(CNNDiscriminator, self).__init__()

        self.conv1 = nn.Conv1d(input_dim, 32, kernel_size = 3, stride = 1, padding = 1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size = 3, stride = 1, padding = 1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size = 3, stride = 1, padding = 1)

        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.2)

        conv_output_size = seq_length // 8 * 128

        self.fc = nn.Sequential(
            nn.Linear(conv_output_size, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.transpose(1, 2)

        x = self.pool(F.leaky_relu(self.conv1(x), 0.2))
        x = self.pool(F.leaky_relu(self.conv2(x), 0.2))
        x = self.pool(F.leaky_relu(self.conv3(x), 0.2))

        x = x.flatten(1)
        x = self.fc(x)
        return x


def get_technical_indicators(data):
    df = data.copy()

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window = 14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window = 14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    df['volatility'] = df['close'].pct_change().rolling(window = 20).std()

    return df


def create_sequence(
        data: pd.DataFrame,
        seq_len: int = 10,
        target_col: str = 'returns',
) -> Tuple:

    sequences = []
    targets = []

    for i in range(len(data) - seq_len):
        seq = data.iloc[i:i + seq_len].values
        target = data.iloc[i + seq_len][target_col]

        if not (np.isnan(seq).any() or np.isnan(target)):
            sequences.append(seq)
            targets.append(target)

    return np.array(sequences), np.array(targets)


def vae_loss_function(recon_x, x, mu, log_var):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kl_loss


def train_vae(
        vae,
        train_loader,
        val_loader,
        epochs = 100,
        lr = 0.001
):
    optimizer = optim.Adam(vae.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        vae.train()
        train_loss = 0

        for batch_data, _ in train_loader:
            optimizer.zero_grad()
            recon_batch, mu, log_var = vae(batch_data)
            loss = vae_loss_function(recon_batch, batch_data, mu, log_var)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        vae.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_data, _ in val_loader:
                recon_batch, mu, log_var = vae(batch_data)
                loss = vae_loss_function(recon_batch, batch_data, mu, log_var)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_val_loss = val_loss / len(val_loader.dataset)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        if epoch % 10 == 0:
            print(f'Epoch [{epoch} / {epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

    return train_losses, val_losses


def train_gan(
        generator,
        discriminator,
        train_loader,
        epochs = 100,
        lr = 0.0002,
):
    criterion = nn.BCELoss()

    optimizer_g = optim.Adam(generator.parameters(), lr=lr)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr)

    g_losses = []
    d_losses = []

    for epoch in range(epochs):
        for i, (real_data, real_targets) in enumerate(train_loader):
            batch_size = real_data.size(0)

            # Train discriminator
            optimizer_d.zero_grad()

            real_labels = torch.ones(batch_size, 1)
            real_outputs = discriminator(real_data)
            d_loss_real = criterion(real_outputs, real_labels)

            fake_outputs = generator(real_data)
            fake_data = torch.cat([real_data[:, 1:], fake_outputs.unsqueeze(1)], dim=1)
            fake_labels = torch.zeros(batch_size, 1)
            fake_outputs_d = discriminator(fake_data.detach())
            d_loss_fake = criterion(fake_outputs_d, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_d.step()

            # Train Generator
            optimizer_g.zero_grad()

            fake_outputs_d = discriminator(fake_data)
            g_loss = criterion(fake_outputs_d, real_labels)
            g_loss.backward()
            optimizer_g.step()

            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())

        if epoch % 10 == 0:
            print(f'Epoch [{epoch} / {epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}')

    return g_losses, d_losses


def main():
    df = pd.read_csv('xnas-itch-20180501-20250430.ohlcv-1s.csv.zst', compression='zstd')

    df['returns'] = df['close'].pct_change() * 100

    df = get_technical_indicators(df)

    df = df.dropna()

    feature_columns = ['returns', 'rsi', 'volatility']

    feature_columns = [col for col in feature_columns if col in df.columns]

    print(f"Using features: {feature_columns}")
    print(f"Data shape after preprocessing: {df.shape}")

    feature_data = df[feature_columns].values

    train_size = int(len(feature_data) * 0.7)
    val_size = int(len(feature_data) * 0.15)

    train_data = feature_data[:train_size]
    val_data = feature_data[train_size:train_size + val_size]
    test_data = feature_data[train_size + val_size:]

    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data)
    val_data_scaled = scaler.transform(val_data)
    test_data_scaled = scaler.transform(test_data)

    print(f"Train data shape: {train_data_scaled.shape}")
    print(f"Val data shape: {val_data_scaled.shape}")
    print(f"Test data shape: {test_data_scaled.shape}")

    seq_len = 10

    X_train, y_train = create_sequence(pd.DataFrame(train_data_scaled, columns = feature_columns),
                                       seq_len=seq_len, target_col='returns')
    X_val, y_val = create_sequence(pd.DataFrame(val_data_scaled, columns = feature_columns),
                                   seq_len=seq_len, target_col='returns')
    X_test, y_test = create_sequence(pd.DataFrame(test_data_scaled, columns = feature_columns),
                                     seq_len=seq_len, target_col='returns')

    print(f"Sequence shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)

    batch_size = 32
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print('\nTraining VAE...')
    vae = VAE(input_dim = len(feature_columns), latent_dim = 4)

    vae_train_data = X_train.reshape(-1, X_train.shape[-1])
    vae_val_data = X_val.reshape(-1, X_val.shape[-1])

    vae_train_tensor = torch.FloatTensor(vae_train_data)
    vae_val_tensor = torch.FloatTensor(vae_val_data)

    vae_train_dataset = TensorDataset(vae_train_tensor, vae_train_tensor)
    vae_val_dataset = TensorDataset(vae_val_tensor, vae_val_tensor)

    vae_train_loader = DataLoader(vae_train_dataset, batch_size=batch_size, shuffle=True)
    vae_val_loader = DataLoader(vae_val_dataset, batch_size=batch_size, shuffle=False)

    vae_train_losses, vae_val_losses = train_vae(vae, vae_train_loader, vae_val_loader, epochs = 50)

    print("Extracting VAE features...")
    vae.eval()
    with torch.no_grad():
        vae_features_train = vae().extract_features(X_train_tensor.reshape(-1, X_train_tensor.shape[-1]))
        vae_features_val = vae().extract_features(X_val_tensor.reshape(-1, X_val_tensor.shape[-1]))
        vae_features_test = vae().extract_features(X_test_tensor.reshape(-1, X_test_tensor.shape[-1]))

    vae_features_train = vae_features_train.reshape(X_train.shape[0], seq_len, -1)
    vae_features_val = vae_features_val.reshape(X_val.shape[0], seq_len, -1)
    vae_features_test = vae_features_test.reshape(X_test.shape[0], seq_len, -1)

    print(f"VAE features shape: {vae_features_train.shape}")

    print("\nInitializing GAN models...")
    input_dim = vae_features_train.shape[-1]
    generator = LSTMGenerator(input_dim = input_dim, hidden_dim = 128, num_layers = 2)
    discriminator = CNNDiscriminator(input_dim = input_dim, seq_length = seq_len)

    gan_train_dataset = TensorDataset(vae_features_train, y_train_tensor)
    gan_train_loader = DataLoader(gan_train_dataset, batch_size=batch_size, shuffle=True)

    print("Training GAN...")
    g_losses, d_losses = train_gan(generator, gan_train_loader, epochs = 100)

    print("\nMaking predictions...")
    generator.eval()
    with torch.no_grad():
        predictions = generator(vae_features_test)
        predictions = predictions.cpu().numpy().flatten()

    actual_returns = y_test

    dummy_features = np.zeros((len(predictions), len(feature_columns)))
    dummy_features[:, 0] = predictions
    predictions_original = scaler.inverse_transform(dummy_features)[:, 0]

    dummy_actual = np.zeros((len(actual_returns), len(feature_columns)))
    dummy_actual[:, 0] = actual_returns
    actual_original = scaler.inverse_transform(dummy_actual)[:, 0]

    mse = np.mean((predictions_original - actual_original) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions_original - actual_original))

    print(f"\nPerformance Metrics:")
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")

    plt.figure(figsize=(15, 10))

    # Plot 1: VAE training loss
    plt.subplot(2, 3, 1)
    plt.plot(vae_train_losses, label='Train Loss')
    plt.plot(vae_val_losses, label='Val Loss')
    plt.title('VAE Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.yscale('log')

    # Plot 2: GAN losses
    plt.subplot(2, 3, 2)
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(d_losses, label='Discriminator Loss')
    plt.title('GAN Training Losses')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()

    # Plot 3: Predictions vs Actual (last 100 points)
    plt.subplot(2, 3, 3)
    n_points = min(100, len(predictions_original))
    plt.plot(actual_original[-n_points:], label='Actual Returns', alpha=0.7)
    plt.plot(predictions_original[-n_points:], label='Predicted Returns', alpha=0.7)
    plt.title('Returns Prediction (Last 100 Points)')
    plt.xlabel('Time')
    plt.ylabel('Returns (%)')
    plt.legend()

    # Plot 4: Scatter plot of predictions vs actual
    plt.subplot(2, 3, 4)
    plt.scatter(actual_original, predictions_original, alpha=0.5)
    plt.plot([actual_original.min(), actual_original.max()],
             [actual_original.min(), actual_original.max()], 'r--', lw=2)
    plt.xlabel('Actual Returns')
    plt.ylabel('Predicted Returns')
    plt.title('Predictions vs Actual')

    # Plot 5: Error distribution
    plt.subplot(2, 3, 5)
    errors = predictions_original - actual_original
    plt.hist(errors, bins=50, alpha=0.7)
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')

    # Plot 6: Cumulative returns
    plt.subplot(2, 3, 6)
    cumulative_actual = np.cumsum(actual_original)
    cumulative_predicted = np.cumsum(predictions_original)
    plt.plot(cumulative_actual, label='Actual Cumulative Returns')
    plt.plot(cumulative_predicted, label='Predicted Cumulative Returns')
    plt.title('Cumulative Returns')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Returns (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()

    print("\nTraining completed successfully!")
    print(f"Final model predicts returns with RMSE: {rmse:.6f}")


if __name__ == '__main__':
    main()