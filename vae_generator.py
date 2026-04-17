import argparse
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats import ks_2samp, ttest_ind
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

class BetaVAE(nn.Module):
    def __init__(self, input_size, latent_size=8, hidden_sizes=(128, 64)):
        super().__init__()
        # Encoder
        encoder_layers = []
        last_size = input_size
        for size in hidden_sizes:
            encoder_layers.extend([nn.Linear(last_size, size), nn.ReLU()])
            last_size = size
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space
        self.mu_layer = nn.Linear(last_size, latent_size)
        self.logvar_layer = nn.Linear(last_size, latent_size)
        
        # Decoder
        decoder_layers = []
        last_size = latent_size
        for size in reversed(hidden_sizes):
            decoder_layers.extend([nn.Linear(last_size, size), nn.ReLU()])
            last_size = size
        decoder_layers.append(nn.Linear(last_size, input_size))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        h = self.encoder(x)
        return self.mu_layer(h), self.logvar_layer(h)

    def reparameterize(self, mu, logvar, noise_scale=1.0):
        std = torch.exp(0.5 * logvar) * noise_scale
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, noise_scale=1.0):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar, noise_scale)
        return self.decoder(z), mu, logvar

def vae_loss(recon_x, x, mu, logvar, beta):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (recon_loss + beta * kld) / x.size(0)

def train_model(model, dataloader, epochs, lr, beta, device):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        
        for (batch,) in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(batch)
            loss = vae_loss(recon, batch, mu, logvar, beta)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.size(0)
        
        if epoch % 50 == 0 or epoch == 1:
            avg_loss = total_loss / len(dataloader.dataset)
            print(f"Epoch {epoch:4d} | Loss: {avg_loss:.6f}")

def generate_data(model, num_samples, scaler, noise_scale, device):
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, model.mu_layer.out_features).to(device) * noise_scale
        synthetic = model.decoder(z).cpu().numpy()
    return scaler.inverse_transform(synthetic)

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic torsion data using VAE")
    parser.add_argument("--csv_path", required=True, help="Path to original CSV file")
    parser.add_argument("--output_path", default="synthetic_torsion_data.csv", help="Output file path")
    parser.add_argument("--epochs", type=int, default=400, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--latent_dim", type=int, default=8, help="Latent space size")
    parser.add_argument("--hidden", nargs="+", type=int, default=[128, 64], help="Hidden layer sizes")
    parser.add_argument("--beta", type=float, default=0.5, help="KL divergence weight")
    parser.add_argument("--noise_scale", type=float, default=1.2, help="Sampling noise scale")
    parser.add_argument("--synthetic_rows", type=int, default=500, help="Synthetic samples to generate")
    
    args = parser.parse_args()

    # Set random seeds
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    
    # Load data
    df = pd.read_csv(args.csv_path)
    numeric_df = df.select_dtypes(include=np.number)
    
    # Scale data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df.values)
    
    # Create dataloader
    dataset = TensorDataset(torch.tensor(scaled_data, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Initialize and train model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = BetaVAE(
        input_size=scaled_data.shape[1],
        latent_size=args.latent_dim,
        hidden_sizes=args.hidden
    )
    
    train_model(
        model=model,
        dataloader=dataloader,
        epochs=args.epochs,
        lr=1e-3,
        beta=args.beta,
        device=device
    )
    
    # Generate synthetic data
    synthetic = generate_data(
        model=model,
        num_samples=args.synthetic_rows,
        scaler=scaler,
        noise_scale=args.noise_scale,
        device=device
    )
    
    # Clip to physical range and save
    synthetic = np.clip(synthetic, 0, 360)
    synthetic_df = pd.DataFrame(synthetic, columns=numeric_df.columns)
    synthetic_df.to_csv(args.output_path, index=False)
    print(f"\nSynthetic data saved to: {args.output_path}")

if __name__ == "__main__":
    main()