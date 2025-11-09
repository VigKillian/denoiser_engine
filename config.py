# config.py
import os
from datetime import datetime

train_dir = "./dataset/train"
val_dir   = "./dataset/val"
model_root = "./model_data"
log_root   = "./logs"

SUP_FLAG = 1  
pic_size = (32, 32) 
epochs = 20
batch_size = 64
learning_rate = 1e-3
keep_prob_v = 0.7
mask_prob_v = 0.0 if SUP_FLAG else 0.3

CUDA_VISIBLE_DEVICES = "0"  
gpu_allow_growth = True
gpu_mem_fraction = 0.8

timestamp = '{:%m-%d_%H-%M/}'.format(datetime.now())

def make_model_path(model_name: str) -> str:
    path = os.path.join(model_root, model_name + "--" + timestamp)
    os.makedirs(path, exist_ok=True)
    return path

def make_log_dirs(run_name: str):
    tr = os.path.join(log_root, "train", f"{run_name}_{timestamp}")
    te = os.path.join(log_root, "test",  f"{run_name}_{timestamp}")
    os.makedirs(tr, exist_ok=True)
    os.makedirs(te, exist_ok=True)
    return tr, te

import glob
import numpy as np
from skimage.io import imread
from skimage.transform import resize

def list_images(d, exts=(".pgm",".png",".jpg",".jpeg",".bmp",".tif",".tiff")):
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(d, f"*{ext}")))
    return sorted(paths)

def load_gray_resized(paths, size_hw):
    arr = []
    for p in paths:
        img = imread(p, as_gray=True)
        if img.dtype not in (np.float32, np.float64):
            img = img.astype(np.float32) / 255.0
        img = resize(img, (size_hw[0], size_hw[1]),
                     preserve_range=True, anti_aliasing=True).astype(np.float32)
        arr.append(img)
    if not arr:
        raise RuntimeError(f"No images in {os.path.dirname(paths[0]) if paths else d}")
    return np.stack(arr, axis=0)

def _stem(path): 
    return os.path.splitext(os.path.basename(path))[0]

def _strip_noise_suffix(stem_str, suffix="_noise"):
    return stem_str[:-len(suffix)] if stem_str.endswith(suffix) else stem_str

def build_pairs(noisy_dir, clean_dir, size_hw):
    noisy_list = list_images(noisy_dir)
    clean_list = list_images(clean_dir)
    noisy_map = {_strip_noise_suffix(_stem(p)): p for p in noisy_list}  # class1_0_noise -> class1_0
    clean_map = {_stem(p): p for p in clean_list}
    keys = sorted(set(noisy_map) & set(clean_map))
    if not keys:
        raise RuntimeError("No filename pairs matched between noisy/*_noise and imgs/*")
    X = load_gray_resized([noisy_map[k] for k in keys], size_hw)
    Y = load_gray_resized([clean_map[k] for k in keys], size_hw)
    return X, Y

def build_unsup(imgs_dir, size_hw):
    X = load_gray_resized(list_images(imgs_dir), size_hw)
    return X, X

def build_datasets():
    size = pic_size
    if SUP_FLAG:
        train_x, train_y = build_pairs(os.path.join(train_dir, "noisy"),
                                       os.path.join(train_dir, "imgs"), size)
        test_x,  test_y  = build_pairs(os.path.join(val_dir, "noisy"),
                                       os.path.join(val_dir, "imgs"),  size)
    else:
        train_x, train_y = build_unsup(os.path.join(train_dir, "imgs"), size)
        test_x,  test_y  = build_unsup(os.path.join(val_dir, "imgs"),  size)
    return train_x, train_y, test_x, test_y

def upsample_sizes(pools=3):
    H, W = pic_size
    return (H//4, W//4), (H//2, W//2), (H, W)