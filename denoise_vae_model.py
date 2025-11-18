# denoise_vae_model.py
import os
from glob import glob

from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms


class DenoisePairDataset(Dataset):

    def __init__(self, root_dir, split='train', img_size=128, max_pairs=None):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size

        clean_dir = os.path.join(root_dir, split, "imgs")
        noisy_dir = os.path.join(root_dir, split, "noisy")

        noisy_paths = []
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff","*.ppm"]:
            noisy_paths.extend(glob(os.path.join(noisy_dir, ext)))
        noisy_paths = sorted(noisy_paths)

        if len(noisy_paths) == 0:
            raise RuntimeError(f"No noisy images found in {noisy_dir}")

        self.pairs = []
        for n_path in noisy_paths:
            
            name = os.path.basename(n_path)
            c_name = name.replace("_noise", "")
            c_path = os.path.join(clean_dir, c_name)

            if not os.path.exists(c_path):
                print(f"[WARN] clean image not found for {name}, skip.")
                continue

            self.pairs.append((n_path, c_path))
            if max_pairs is not None:
                self.pairs = self.pairs[:max_pairs]

        if len(self.pairs) == 0:
            raise RuntimeError("No (noisy, clean) pairs matched!")

        print(f"[{split}] Found {len(self.pairs)} pairs.")

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),  # [0,1], (C,H,W)
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        noisy_path, clean_path = self.pairs[idx]
        noisy_img = Image.open(noisy_path).convert("RGB")
        clean_img = Image.open(clean_path).convert("RGB")

        noisy = self.transform(noisy_img)
        clean = self.transform(clean_img)
        return noisy, clean


# =====================
# VAE
# =====================

class ConvDenoiseVAE(nn.Module):
    def __init__(self, img_channels=3, img_size=64, latent_dim=128,nb_channels_base = 32):
        super().__init__()
        self.img_channels = img_channels
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.nb_channels_base = nb_channels_base



        # Encoder: 3 x 128 x 128 -> 128 x 16 x 16
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, nb_channels_base, kernel_size=4, stride=2, padding=1),  # 64x64
            nn.ReLU(inplace=True),
            nn.Conv2d(nb_channels_base, nb_channels_base*2, kernel_size=4, stride=2, padding=1),           # 32x32
            nn.ReLU(inplace=True),
            nn.Conv2d(nb_channels_base*2, nb_channels_base*4, kernel_size=4, stride=2, padding=1),          # 16x16
            nn.ReLU(inplace=True),
        )

        self.enc_feat_h = img_size // 8
        self.enc_feat_w = img_size // 8
        enc_flat_dim = nb_channels_base* 4 * self.enc_feat_h * self.enc_feat_w

        # latent: mu, logvar
        self.fc_mu = nn.Linear(enc_flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(enc_flat_dim, latent_dim)

        # Decoder: latent -> 128 x 16 x 16
        self.fc_dec = nn.Linear(latent_dim, enc_flat_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(nb_channels_base*4, nb_channels_base*2, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nb_channels_base*2, nb_channels_base, kernel_size=4, stride=2, padding=1),   # 64x64
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nb_channels_base, img_channels, kernel_size=4, stride=2, padding=1),  # 128x128
            nn.Sigmoid(),  # [0,1]
        )

    def encode(self, x):
        h = self.encoder(x)                # [B,128,16,16]
        h = h.view(h.size(0), -1)          # [B, enc_flat_dim]
        mu = self.fc_mu(h)                 # [B, latent_dim]
        logvar = self.fc_logvar(h)         # [B, latent_dim]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_dec(z)
        h = h.view(h.size(0), self.nb_channels_base*4, self.enc_feat_h, self.enc_feat_w)
        x_recon = self.decoder(h)
        return x_recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


# =====================
# 3. loss + train/eval per epoch
# =====================

def vae_loss(recon, target, mu, logvar, beta=1e-4):

    recon_loss = nn.functional.mse_loss(recon, target, reduction='mean')
    kl = -0.5 * torch.mean(torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp(), dim=1
    ))
    return recon_loss + beta * kl, recon_loss, kl


def train_epoch(model, loader, optimizer, device, beta=1e-4):
    model.train()
    total_loss = total_recon = total_kl = 0.0

    for noisy, clean in loader:
        noisy = noisy.to(device)
        clean = clean.to(device)

        optimizer.zero_grad()
        recon, mu, logvar = model(noisy)
        loss, recon_loss, kl_loss = vae_loss(recon, clean, mu, logvar, beta=beta)
        loss.backward()
        optimizer.step()

        bs = noisy.size(0)
        total_loss  += loss.item()        * bs
        total_recon += recon_loss.item()  * bs
        total_kl    += kl_loss.item()     * bs

    n = len(loader.dataset)
    return total_loss / n, total_recon / n, total_kl / n


@torch.no_grad()
def eval_epoch(model, loader, device, beta=1e-4):
    model.eval()
    total_loss = total_recon = total_kl = 0.0

    for noisy, clean in loader:
        noisy = noisy.to(device)
        clean = clean.to(device)

        recon, mu, logvar = model(noisy)
        loss, recon_loss, kl_loss = vae_loss(recon, clean, mu, logvar, beta=beta)

        bs = noisy.size(0)
        total_loss  += loss.item()        * bs
        total_recon += recon_loss.item()  * bs
        total_kl    += kl_loss.item()     * bs

    n = len(loader.dataset)
    return total_loss / n, total_recon / n, total_kl / n
