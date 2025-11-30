# inference_denoise.py
import os
from glob import glob

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from denoise_vae_model import ConvDenoiseVAE 


def load_model(checkpoint_path, img_size=64, latent_dim=256, nb_channels_base=32, device="cpu"):
    model = ConvDenoiseVAE(
        img_channels=3,
        img_size=img_size,
        latent_dim=latent_dim,
        nb_channels_base=nb_channels_base
    )
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def denoise_folder(model, in_dir, out_dir, img_size=64, device="cpu"):
    os.makedirs(out_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),     # [0,1]
    ])

    exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.ppm"]
    img_paths = []
    for e in exts:
        img_paths.extend(glob(os.path.join(in_dir, e)))
    img_paths = sorted(img_paths)

    if not img_paths:
        print(f"[WARN] No images found in {in_dir}")
        return

    print(f"[Info] Found {len(img_paths)} noisy images in {in_dir}")

    for p in img_paths:
        img = Image.open(p).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)   # [1,3,H,W]

        with torch.no_grad():
            recon, _, _ = model(x)      # recon: [1,3,H,W]

        recon_img = recon.squeeze(0).cpu()  # [3,H,W]
        recon_pil = transforms.ToPILImage()(recon_img)

        name = os.path.basename(p)
        save_path = os.path.join(out_dir, name.replace("_noise", "_denoised"))
        recon_pil.save(save_path)

        print(f"[OK] {p}  ->  {save_path}")


def main():
    # ====== setting ======
    checkpoint_path = "checkpoints/11-30_10-53/denoise_vae_epoch  3.pth" 
     
    img_size = 128
    latent_dim = 256
    nb_channels_base = 32

    noisy_dir = "./inference_noisy" 
    output_dir = "./inference_output" 
    # ================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Use device:", device)

    model = load_model(
        checkpoint_path,
        img_size=img_size,
        latent_dim=latent_dim,
        nb_channels_base=nb_channels_base,
        device=device
    )

    denoise_folder(model, noisy_dir, output_dir, img_size=img_size, device=device)
    print("Done.")


if __name__ == "__main__":
    main()
