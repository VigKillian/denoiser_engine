from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import config as cfg

_, _, test_x, test_y = cfg.build_datasets()
pic_size = cfg.pic_size

print('enter the time of file: ')
date = input()
print('enter the nb of epochs you want to check: ')
nb_epochs = input()

mat = loadmat("logs/test/bsc-ConvDAE_"+date+"/val_imgs_epoch"+nb_epochs+".mat")
rec_all = mat["reconstructed"]

print("rec_all shape:", rec_all.shape)

take = min(100, len(test_x))
test_idx = np.linspace(0, len(test_x)-1, take).astype('int32')

idx = np.linspace(0, take-1, 10).astype('int32')

in_imgs = test_x[test_idx[idx]] 
gt_imgs = test_y[test_idx[idx]] 
reconstructed = rec_all[idx]  

fig, axes = plt.subplots(nrows=3, ncols=10, sharex=True, sharey=True, figsize=(20, 4))
for images, row in zip([in_imgs, reconstructed, gt_imgs], axes):
    for img, ax in zip(images, row):
        img_disp = np.clip(img, 0.0, 1.0) 
        ax.imshow(img_disp)
        ax.set_axis_off()

fig.tight_layout(pad=0.1)
plt.show()
