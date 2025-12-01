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
        noisy_dir = os.path.join(root_dir, split, "noisy_gauss_fort")

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
# GAN
# =====================

adversarial_loss = torch.nn.BCELoss()

class ConvoGan(nn.Module):
    def __init__(self, img_channels=6, img_size=64, nb_channels_base=32):

        super().__init__()
        self.img_channels = img_channels
        self.img_size = img_size
        self.nb_channels_base = nb_channels_base

        self.features = nn.Sequential(
            nn.Conv2d(img_channels, nb_channels_base, kernel_size=4, stride=2, padding=1),  # 128 -> 64
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nb_channels_base, nb_channels_base * 2, kernel_size=4, stride=2, padding=1),  # 64 -> 32
            nn.BatchNorm2d(nb_channels_base * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nb_channels_base * 2, nb_channels_base * 4, kernel_size=4, stride=2, padding=1),  # 32 -> 16
            nn.BatchNorm2d(nb_channels_base * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.enc_feat_h = img_size // 8
        self.enc_feat_w = img_size // 8
        enc_flat_dim = nb_channels_base * 4 * self.enc_feat_h * self.enc_feat_w

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(enc_flat_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, pair):
        """
        pair: [B, 6, H, W] = concat(noisy, target) on channel dim
        """
        feat = self.features(pair)
        validity = self.classifier(feat)  # [B,1]
        return validity


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
        total_loss  += loss.item()       * bs
        total_recon += recon_loss.item() * bs
        total_kl    += kl_loss.item()    * bs

    n = len(loader.dataset)
    return total_loss / n, total_recon / n, total_kl / n


def train_epoch_withGan(
    vae_model,
    discriminator,
    optimizer_G,
    optimizer_D,
    loader,
    device,
    beta=1e-4,
    lambda_gan=1e-4,
    k_G=1,
    k_D=5, 
):

    vae_model.train()
    discriminator.train()

    total_G = total_recon = total_kl = total_gan_G = 0.0
    total_D = 0.0
    N = len(loader.dataset)
    D_count = 0

    for noisy, clean in loader:
        noisy = noisy.to(device)
        clean = clean.to(device)
        bs = noisy.size(0)


        for _ in range(k_D):
            optimizer_D.zero_grad()

            with torch.no_grad():
                recon, _, _ = vae_model(noisy)

            real_pair = torch.cat([noisy, clean], dim=1)   # [B, 6, H, W]
            fake_pair = torch.cat([noisy, recon], dim=1)   # [B, 6, H, W]

            valid = torch.ones(bs, 1, device=device)
            fake  = torch.zeros(bs, 1, device=device)

            pred_real = discriminator(real_pair)
            pred_fake = discriminator(fake_pair)

            loss_D_real = adversarial_loss(pred_real, valid)
            loss_D_fake = adversarial_loss(pred_fake, fake)
            loss_D = 0.5 * (loss_D_real + loss_D_fake)

            loss_D.backward()
            optimizer_D.step()

            total_D += loss_D.item() * bs
            D_count += bs


        for _ in range(k_G):
            optimizer_G.zero_grad()

            recon, mu, logvar = vae_model(noisy)
            vae_total, recon_loss, kl_loss = vae_loss(
                recon, clean, mu, logvar, beta=beta
            )

            fake_pair_for_G = torch.cat([noisy, recon], dim=1)
            valid = torch.ones(bs, 1, device=device)
            pred_fake_for_G = discriminator(fake_pair_for_G)
            gan_loss_G = adversarial_loss(pred_fake_for_G, valid)

            loss_total = vae_total + lambda_gan * gan_loss_G
            loss_total.backward()
            optimizer_G.step()

            total_G     += loss_total.item()  * bs
            total_recon += recon_loss.item()  * bs
            total_kl    += kl_loss.item()     * bs
            total_gan_G += gan_loss_G.item()  * bs

    if D_count > 0:
        D_epoch = total_D / D_count
    else:
        D_epoch = 0.0

    return {
        "G_total": total_G / N,
        "recon":   total_recon / N,
        "kl":      total_kl / N,
        "gan_G":   total_gan_G / N,
        "D_loss":  D_epoch,
    }


@torch.no_grad()
def eval_epoch(
    vae_model,
    loader,
    device,
    beta=1e-4,
    discriminator=None,
    lambda_gan=0.0
):

    vae_model.eval()
    if discriminator is not None:
        discriminator.eval()

    total_loss = total_recon = total_kl = 0.0
    total_adv  = 0.0
    N = len(loader.dataset)

    for noisy, clean in loader:
        noisy = noisy.to(device)
        clean = clean.to(device)

        recon, mu, logvar = vae_model(noisy)
        vae_total, recon_loss, kl_loss = vae_loss(
            recon, clean, mu, logvar, beta=beta
        )

        if discriminator is not None and lambda_gan > 0.0:
            fake_pair = torch.cat([noisy, recon], dim=1)
            valid = torch.ones(noisy.size(0), 1, device=device)
            pred_fake = discriminator(fake_pair)
            gan_loss_G = adversarial_loss(pred_fake, valid)
        else:
            gan_loss_G = torch.tensor(0.0, device=device)

        loss = vae_total + lambda_gan * gan_loss_G

        bs = noisy.size(0)
        total_loss  += loss.item()       * bs
        total_recon += recon_loss.item() * bs
        total_kl    += kl_loss.item()    * bs
        total_adv   += gan_loss_G.item() * bs

    return (
        total_loss / N,
        total_recon / N,
        total_kl / N,
        total_adv / N,
    )
