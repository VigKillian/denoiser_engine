import numpy as np
import matplotlib.pyplot as plt
import os

print("Enter the file name : [format]mm-dd_hh-mm")
file_time = input()
data = np.loadtxt("histos/"+file_time+".dat")
curve_root    = os.path.join("curve", file_time)
os.makedirs(curve_root, exist_ok=True)
epoch      = data[:, 0]
G_total    = data[:, 1]
D_loss     = data[:, 2]
GAN_G      = data[:, 3]
recon      = data[:, 4]
val_loss   = data[:, 5]

plt.figure()
plt.plot(epoch, D_loss, label="Discriminator loss")
plt.plot(epoch, G_total, label="Generator loss (VAE+GAN)")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("./curve/"+file_time+"/loss_curves.png")

plt.figure()
plt.plot(epoch, recon,    label="Train recon MSE")
plt.plot(epoch, val_loss, label="Val total loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("./curve/"+file_time+"/loss_recon_val.png")