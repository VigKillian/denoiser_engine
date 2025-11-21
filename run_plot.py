import numpy as np
import matplotlib.pyplot as plt

print("Enter the file name : [format]mm-dd_hh-mm")
file_time = input()
data = np.loadtxt("histos/"+file_time+".dat")
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
plt.savefig("loss_curves.png")

plt.figure()
plt.plot(epoch, recon,    label="Train recon MSE")
plt.plot(epoch, val_loss, label="Val total loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_recon_val.png")