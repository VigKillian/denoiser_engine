#!/usr/bin/env python3
import os
import glob
import argparse

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# =========================
#  Bruit "expansion dynamique"
# =========================

def simulate_expansion_noise(img,
                             quant_bits=6,
                             noise_std=0.008,
                             color_noise_std=0.01):
    """
    img : numpy float32 [H, W, 3] dans [0,1], RGB (déjà gamma/sRGB)
    Renvoie une image bruitée + expansion dynamique, également dans [0,1].
    """
    noisy = img.astype(np.float32).copy()

    # 1) petit bruit gaussien pixel-wise
    noisy += np.random.normal(0.0, noise_std, noisy.shape).astype(np.float32)

    # 2) bruit de couleur (offset différent par canal)
    color_noise = np.random.normal(0.0, color_noise_std, (1, 1, 3)).astype(np.float32)
    noisy += color_noise

    noisy = np.clip(noisy, 0.0, 1.0)

    # 3) quantification grossière (simulateur d’ADC à N bits)
    levels = 2 ** quant_bits          # ex. 64 niveaux pour 6 bits
    noisy_q = np.round(noisy * (levels - 1)) / (levels - 1)

    # 4) "expansion dynamique" = ré-étirer sur [0,1]
    vmin = noisy_q.min()
    vmax = noisy_q.max()
    if vmax <= vmin + 1e-8:
        expanded = np.zeros_like(noisy_q, dtype=np.float32)
    else:
        expanded = (noisy_q - vmin) / (vmax - vmin)

    return np.clip(expanded, 0.0, 1.0)


# =========================
#  Traitement d’une image
# =========================

def process_one_image(path_in,
                      path_clean_out,
                      path_noisy_out,
                      target_size=None):
    """
    Charge une image, la normalise, applique le bruit, et
    sauve les versions clean / noisy.
    """
    img = Image.open(path_in).convert("RGB")

    if target_size is not None:
        img = img.resize(target_size, Image.BILINEAR)

    arr = np.asarray(img).astype(np.float32) / 255.0    # [0,1]

    noisy = simulate_expansion_noise(arr,
                                     quant_bits=6,
                                     noise_std=0.008,
                                     color_noise_std=0.01)

    # Sauvegarde clean
    clean_uint8 = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    noisy_uint8 = (np.clip(noisy, 0.0, 1.0) * 255.0).astype(np.uint8)

    Image.fromarray(clean_uint8).save(path_clean_out)
    Image.fromarray(noisy_uint8).save(path_noisy_out)


# =========================
#  Debug : histogramme
# =========================

def show_debug_example(path_clean, path_noisy):
    clean = np.asarray(Image.open(path_clean).convert("RGB")).astype(np.uint8)
    noisy = np.asarray(Image.open(path_noisy).convert("RGB")).astype(np.uint8)

    # On regarde par exemple le canal bleu comme dans ton graphe
    blue = noisy[:, :, 2].flatten()

    plt.figure()
    plt.hist(blue, bins=256, range=(0, 255))
    plt.title("Histogramme canal bleu (image bruitée)")
    plt.xlabel("Intensité")
    plt.ylabel("Nombre de pixels")
    plt.show()


# =========================
#  Main
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Dossier d’images propres (jpg/png/...)")
    parser.add_argument("--clean_dir", type=str, required=True,
                        help="Dossier de sortie pour les images clean")
    parser.add_argument("--noisy_dir", type=str, required=True,
                        help="Dossier de sortie pour les images noisy")
    parser.add_argument("--size", type=int, default=128,
                        help="Taille (carrée) de sortie, ex: 128")
    parser.add_argument("--debug", action="store_true",
                        help="Afficher l’histogramme d’un exemple")
    args = parser.parse_args()

    os.makedirs(args.clean_dir, exist_ok=True)
    os.makedirs(args.noisy_dir, exist_ok=True)

    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(args.input_dir, e)))
    paths = sorted(paths)

    print(f"{len(paths)} images trouvées.")

    for i, p in enumerate(paths):
        filename = os.path.splitext(os.path.basename(p))[0] + ".png"
        out_clean = os.path.join(args.clean_dir, filename)
        out_noisy = os.path.join(args.noisy_dir, filename)

        process_one_image(
            p,
            out_clean,
            out_noisy,
            target_size=(args.size, args.size)
        )

        if (i + 1) % 50 == 0 or i == 0:
            print(f"{i+1}/{len(paths)} images traitées...")

    print("Terminé.")

    if args.debug and len(paths) > 0:
        # On prend la première image pour visualiser l’histo
        filename = os.path.splitext(os.path.basename(paths[0]))[0] + ".png"
        path_clean = os.path.join(args.clean_dir, filename)
        path_noisy = os.path.join(args.noisy_dir, filename)
        show_debug_example(path_clean, path_noisy)


if __name__ == "__main__":
    main("propres", "clean_out", "noisy_out", 128)

