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
    def __init__(self, img_channels=3, img_size=64, latent_dim=128, nb_channels_base=32):
        super().__init__()
        self.img_channels = img_channels
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.nb_channels_base = nb_channels_base

        # -------------------------
        # Encoder: downsampling + store skip features
        # -------------------------
        self.enc1 = nn.Sequential(
            nn.Conv2d(img_channels, nb_channels_base, 4, 2, 1),  # 64 -> 32
            nn.ReLU(inplace=True),
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(nb_channels_base, nb_channels_base * 2, 4, 2, 1),  # 32 -> 16
            nn.ReLU(inplace=True),
        )

        self.enc3 = nn.Sequential(
            nn.Conv2d(nb_channels_base * 2, nb_channels_base * 4, 4, 2, 1),  # 16 -> 8
            nn.ReLU(inplace=True),
        )

        # Bottleneck feature size: img_size / 2^3
        self.enc_feat_h = img_size // 8
        self.enc_feat_w = img_size // 8
        enc_flat_dim = nb_channels_base * 4 * self.enc_feat_h * self.enc_feat_w

        # VAE latent mapping
        self.fc_mu = nn.Linear(enc_flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(enc_flat_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, enc_flat_dim)

        # -------------------------
        # Decoder: upsampling + skip + conv
        # -------------------------

        # up1: 8 -> 16, keeping nb*4 channels
        self.up1 = nn.ConvTranspose2d(
            nb_channels_base * 4, nb_channels_base * 4, 4, 2, 1
        )
        # After concatenating f2 (nb*2 channels), reduce channels to nb*2
        self.dec1 = nn.Sequential(
            nn.Conv2d(nb_channels_base * 4 + nb_channels_base * 2,
                      nb_channels_base * 2,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        # up2: 16 -> 32
        self.up2 = nn.ConvTranspose2d(
            nb_channels_base * 2, nb_channels_base * 2, 4, 2, 1
        )
        # After concatenating f1 (nb channels), reduce to nb
        self.dec2 = nn.Sequential(
            nn.Conv2d(nb_channels_base * 2 + nb_channels_base,
                      nb_channels_base,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        # up3: 32 -> 64
        self.up3 = nn.ConvTranspose2d(
            nb_channels_base, nb_channels_base, 4, 2, 1
        )
        # Final 3-channel output layer
        self.dec3 = nn.Sequential(
            nn.Conv2d(nb_channels_base, img_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),  # Output range [0,1]
        )

    # -------------------------
    # Encoder: return latent + skip features
    # -------------------------
    def encode(self, x):
        f1 = self.enc1(x)      # 32x32
        f2 = self.enc2(f1)     # 16x16
        f3 = self.enc3(f2)     # 8x8

        h = f3.view(f3.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar, (f1, f2, f3)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # -------------------------
    # Decoder: using skip connections
    # -------------------------
    def decode(self, z, skips):
        f1, f2, f3 = skips

        h = self.fc_dec(z)
        h = h.view(h.size(0),
                   self.nb_channels_base * 4,
                   self.enc_feat_h,
                   self.enc_feat_w)  # 8x8

        # 8 -> 16
        x = self.up1(h)
        x = torch.cat([x, f2], dim=1)   # concatenate skip2
        x = self.dec1(x)

        # 16 -> 32
        x = self.up2(x)
        x = torch.cat([x, f1], dim=1)   # concatenate skip1
        x = self.dec2(x)

        # 32 -> 64
        x = self.up3(x)
        x = self.dec3(x)
        return x

    def forward(self, x):
        mu, logvar, skips = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, skips)
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
