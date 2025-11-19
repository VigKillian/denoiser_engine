# train_denoise_vae.py
import os
import torch
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from datetime import datetime
import time

from denoise_vae_model import (
    DenoisePairDataset,
    ConvDenoiseVAE,
    train_epoch,
    eval_epoch,
)

from math import log10

def compute_psnr(recon, target):
    mse = torch.mean((recon - target) ** 2).item()
    if mse == 0:
        return float("inf")
    return 10 * log10(1.0 / mse)  # data in [0,1]


@torch.no_grad()
def save_recon_examples(model, loader, device, epoch, out_dir="results", n_show=8):
    model.eval()
    os.makedirs(out_dir, exist_ok=True)

    dataset = loader.dataset
    N = len(dataset)
    if N == 0:
        print("[Vis] Dataset is empty, skip visualization.")
        return

    n_show = min(n_show, N)

    indices = torch.linspace(0, N - 1, steps=n_show).long().tolist()

    noisy_list = []
    clean_list = []
    for idx in indices:
        noisy_img, clean_img = dataset[idx] 
        noisy_list.append(noisy_img)
        clean_list.append(clean_img)

    noisy = torch.stack(noisy_list, dim=0).to(device)   # [n_show, C, H, W]
    clean = torch.stack(clean_list, dim=0).to(device)

    recon, _, _ = model(noisy)

    to_save = torch.cat([noisy, recon, clean], dim=0).cpu()

    grid = vutils.make_grid(
        to_save,
        nrow=n_show, 
        normalize=True,
        value_range=(0, 1)
    )

    save_path = os.path.join(out_dir, f"recon_epoch_{epoch:03d}.png")
    vutils.save_image(grid, save_path)
    print(f"[Vis] Saved reconstruction examples to: {save_path}")


def main():
    # ===== config =====
    root_dir   = "./dataset"  
    max_train = None           
    max_val   = None           
    img_size   = 64
    nb_channels_base = 32   # if 32 : 3->32->32*2->32*4 --> 32*2->32->3
    latent_dim = 256
    batch_size = 12
    epochs     = 50
    lr         = 1e-3
    beta       = 1e-5          # KL
    # =================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Use device:", device)

    # ========= run time =========
    run_id = datetime.now().strftime("%m-%d_%H-%M")
    results_root = os.path.join("results", run_id)
    ckpt_root    = os.path.join("checkpoints", run_id)
    dat_root     = "histos"

    os.makedirs(results_root, exist_ok=True)
    os.makedirs(ckpt_root, exist_ok=True)

    print(f"[Run] results will be saved to: {results_root}")
    print(f"[Run] checkpoints will be saved to: {ckpt_root}")
    # ==========================================

    train_set = DenoisePairDataset(
        root_dir, split="train_color", img_size=img_size, max_pairs=max_train
    )
    test_set  = DenoisePairDataset(
        root_dir, split="val_color",   img_size=img_size, max_pairs=max_val
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader  = DataLoader(
        test_set,  batch_size=batch_size, shuffle=False, num_workers=2
    )

    model = ConvDenoiseVAE(
        img_channels=3, img_size=img_size, latent_dim=latent_dim, nb_channels_base = nb_channels_base
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # histoPath = os.path.join(dat_root, run_id)
    # os.makedirs(histoPath, exist_ok=True)

    for epoch in range(1, epochs + 1):

        timer_start = time.time()

        tr_loss, tr_rec, tr_kl = train_epoch(
            model, train_loader, optimizer, device, beta=beta
        )
        vl_loss, vl_rec, vl_kl = eval_epoch(
            model, test_loader, device, beta=beta
        )

        # compute PSNR on validation set
        model.eval()
        psnr_vals = []

        with torch.no_grad():
            for noisy, clean in test_loader:
                noisy = noisy.to(device)
                clean = clean.to(device)
                recon, _, _ = model(noisy)
                psnr_vals.append(compute_psnr(recon, clean))

        val_psnr = sum(psnr_vals) / len(psnr_vals)

        print(
            f"[Epoch {epoch:03d}] "
            f"Train loss={tr_loss:.6f} (recon={tr_rec:.6f}, KL={tr_kl:.6f}) | "
            f"Val loss={vl_loss:.6f} (recon={vl_rec:.6f}, KL={vl_kl:.6f})"
        )

        if epoch % 2 == 0 or epoch == 1 or epoch == epochs:
            save_recon_examples(
                model, test_loader, device,
                epoch, out_dir=results_root, n_show=8
            )

        if epoch % 2 == 0 or epoch == epochs:
            ckpt_path = os.path.join(
                ckpt_root, f"denoise_vae_epoch{epoch:3d}.pth"
            )
            torch.save(model.state_dict(), ckpt_path)
            print("Saved:", ckpt_path)

        timer_end = time.time()
        print("Time : ", timer_end-timer_start)
        print(f"  >> PSNR = {val_psnr:.3f} dB")

        log_path = os.path.join(dat_root, f"{run_id}.dat")

        with open(log_path, "a") as f:
            f.write(f"{epoch} {val_psnr:.3f}\n")


    print("Training finished.")


if __name__ == "__main__":
    main()
