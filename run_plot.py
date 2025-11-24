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



# fig, ax1 = plt.subplots()

# color = 'tab:red'
# ax1.set_xlabel('time (s)')
# ax1.set_ylabel('exp', color=color)
# ax1.plot(t, data1, color=color)
# ax1.tick_params(axis='y', labelcolor=color)

# ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

# color = 'tab:blue'
# ax2.set_ylabel('sin', color=color)  # we already handled the x-label with ax1
# ax2.plot(t, data2, color=color)
# ax2.tick_params(axis='y', labelcolor=color)

# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.show()

fig, ax1 = plt.subplots()
color = 'tab:blue'
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Discriminator loss", color=color)
ax1.plot(epoch, D_loss, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()

color = 'tab:orange'
ax2.set_ylabel("Generator loss (VAE+GAN)", color=color)
ax2.plot(epoch, G_total, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.savefig("./curve/"+file_time+"/loss_curves.png")

# plt.figure()
# plt.plot(epoch, D_loss, label="Discriminator loss")
# plt.plot(epoch, G_total, label="Generator loss (VAE+GAN)")
# plt.xlabel("Epochs")
# ax2 = plt.xlabel("Epochs").twinx()
# plt.ylabel("Loss")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("./curve/"+file_time+"/loss_curves.png")

plt.figure()
plt.plot(epoch, recon,    label="Train recon MSE")
plt.plot(epoch, val_loss, label="Val total loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("./curve/"+file_time+"/loss_recon_val.png")