# -*- coding: utf-8 -*-
'''
【basic convolutional denoising autoencoder——bsc-ConvDAE】
--max_pool in decoder, non dropout layer
contains two modes:
unspervised: input data will be randomly masked(through dropout), inputs_ & targets_ are the same (noisy data)
supervised: inputs_ & targets_ are clean data and noisy data separately

Author: zhihong (z_zhi_hong@163.com)
Date: 20190505
Modified: zhihong_20210223

in="dataset/train_color_128/noisy"
out="dataset/val_color_128/noisy"
k=200
mkdir -p "$out"
N=$(find "$in" -maxdepth 1 -type f | wc -l)
stride=$(( (N + k - 1) / k ))

find "$in" -maxdepth 1 -type f -print0 \
| sort -z \
| awk -v RS='\0' -v ORS='\0' -v s="$stride" -v k="$k" 'NR % s == 1 { print; c++; if (c>=k) exit }' \
| xargs -0 -I{} mv "{}" "$out"/

'''

# In[]:
# import modules
import numpy as np
from tf_compat import tf
from datetime import datetime  
from time import time
import matplotlib.pyplot as plt
import os
from util import my_io
import config as cfg

# In[] 
# environment config
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CUDA_VISIBLE_DEVICES
config = tf.ConfigProto()
config.gpu_options.allow_growth = cfg.gpu_allow_growth
config.gpu_options.per_process_gpu_memory_fraction = cfg.gpu_mem_fraction


# In[]:
# graph reset
tf.reset_default_graph() 


# In[]:
# parameters config
# flags
SUP_FLAG = 1 # supervised learning flag

# setting
SUP_FLAG = cfg.SUP_FLAG
epochs = cfg.epochs
batch_size = cfg.batch_size
learning_rate = cfg.learning_rate
pic_size = list(cfg.pic_size)  
keep_prob_v = cfg.keep_prob_v
mask_prob_v = cfg.mask_prob_v

train_log_dir, test_log_dir = cfg.make_log_dirs("bsc-ConvDAE")
model_path = cfg.make_model_path("bsc-ConvDAE(sup)" if SUP_FLAG else "bsc-ConvDAE(unsup)")

from skimage.metrics import peak_signal_noise_ratio as sk_psnr

def psnr_batch(pred, gt, data_range=1.0):
    if pred.ndim == 4 and pred.shape[-1] == 1:
        pred = pred[..., 0]
    if gt.ndim == 4 and gt.shape[-1] == 1:
        gt = gt[..., 0]
    psnrs = [sk_psnr(gt[i], pred[i], data_range=data_range) for i in range(len(pred))]
    return psnrs, float(np.mean(psnrs))

# In[]: 
# functions
# tensorflow log summary
def summaryWriter(train_writer, test_writer, record_point, run_tensor, train_feed_dict, test_feed_dict, iter):
    tr, tr_cost = sess.run([record_point, run_tensor], feed_dict=train_feed_dict)
    te, te_cost = sess.run([record_point, run_tensor], feed_dict=test_feed_dict)        
    train_writer.add_summary(tr, iter)
    test_writer.add_summary(te, iter)         
    print("Epoch:",iter,"Train cost:",tr_cost,"Test cost",te_cost)  

    
# In[]:
# model
# activate function
act_fun = tf.nn.relu    # inner layer act_fun
act_fun_out = tf.nn.relu     # output layer act_fun, for slm
# act_fun_out = tf.nn.tanh     # output layer act_fun, for N-MNIST

with tf.name_scope('inputs'):
    inputs_ = tf.placeholder(tf.float32, (None, *pic_size, 3), name='inputs_')
    targets_ = tf.placeholder(tf.float32, (None, *pic_size, 3), name='targets_')
    keep_prob = tf.placeholder(tf.float32)    #range 0.0-1.0
    mask_prob = tf.placeholder(tf.float32)    #range 0.0-1.0
    
# net structure
# Encoder
with tf.name_scope('encoder'):
    drop = tf.nn.dropout(inputs_, 1-mask_prob)  # unsupervised：randomly masked

    conv1 = tf.layers.conv2d(drop, 64, (3,3), padding='same', activation=act_fun)
    max_p1 = tf.layers.max_pooling2d(conv1, (2,2), (2,2), padding='same')

    conv2 = tf.layers.conv2d(max_p1, 32, (3,3), padding='same', activation=act_fun)
    max_p2 = tf.layers.max_pooling2d(conv2, (2,2), (2,2), padding='same')

    conv3 = tf.layers.conv2d(max_p2, 16, (3,3), padding='same', activation=act_fun)
    max_p3 = tf.layers.max_pooling2d(conv3, (2,2), (2,2), padding='same')
    # max_p3 shape: [batch, H/8, W/8, 16] si pic_size=(64,64)


with tf.name_scope('latent'):
    # encoder 
    flat = tf.layers.flatten(max_p3)    # [batch, H/8 * W/8 * 16]

    latent_dim = 128   #  32/64/256 

    z_mu     = tf.layers.dense(flat, latent_dim, name='z_mu')
    z_logvar = tf.layers.dense(flat, latent_dim, name='z_logvar')

    # reparameterization trick: z = mu + sigma * eps
    eps = tf.random_normal(tf.shape(z_mu))
    z = z_mu + tf.exp(0.5 * z_logvar) * eps

up1, up2, up3 = cfg.upsample_sizes()
H, W = pic_size
enc_H = H // 8
enc_W = W // 8
enc_C = 16     
enc_flat_dim = enc_H * enc_W * enc_C

with tf.name_scope('decoder'):
    fc = tf.layers.dense(z, enc_flat_dim, activation=act_fun, name='z_to_feature')
    dec_in = tf.reshape(fc, (-1, enc_H, enc_W, enc_C))

    res4 = tf.image.resize_nearest_neighbor(dec_in, up1)
    conv4 = tf.layers.conv2d(res4, 16, (3,3), padding='same', activation=act_fun)

    res5 = tf.image.resize_nearest_neighbor(conv4, up2)
    conv5 = tf.layers.conv2d(res5, 32, (3,3), padding='same', activation=act_fun)

    res6 = tf.image.resize_nearest_neighbor(conv5, up3)
    conv6 = tf.layers.conv2d(res6, 64, (3,3), padding='same', activation=act_fun)

# ----------  ----------
with tf.name_scope('outputs'):
    logits_ = tf.layers.conv2d(conv6, 3, (3,3), padding='same', activation=None)
    outputs_ = act_fun_out(logits_, name='outputs_')

# ---------- VAE  loss ----------
with tf.name_scope('loss'):
    
    recon_per_sample = tf.reduce_sum(
        tf.square(targets_ - outputs_),
        axis=[1, 2, 3]
    )
    recon_loss = tf.reduce_mean(recon_per_sample, name='recon_loss')

    kl_per_sample = -0.5 * tf.reduce_sum(
        1.0 + z_logvar - tf.square(z_mu) - tf.exp(z_logvar),
        axis=1
    )
    kl_loss = tf.reduce_mean(kl_per_sample, name='kl_loss')

    beta = 1e-3   #  1e-4, 5e-4, 1e-3
    cost = recon_loss + beta * kl_loss

    tf.summary.scalar('recon_loss', recon_loss)
    tf.summary.scalar('kl_loss', kl_loss)
    tf.summary.scalar('cost', cost)

with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


# In[]:
# load data
train_x, train_y, test_x, test_y = cfg.build_datasets()

print('train_x: ', train_x.shape, '\ttrain_y: ', train_y.shape,
      '\ntest_x: ',  test_x.shape,  '\ttest_y: ',  test_y.shape)

take = min(100, len(test_x))
test_idx = np.linspace(0, len(test_x)-1, take).astype('int32')
test_x1 = test_x[test_idx]
test_y1 = test_y[test_idx]

# data disp
#for k in range(5):
#    plt.subplot(2,5,k+1)
#    plt.imshow(train_x[k])
#    plt.title('train_x_%d'%(k+1))
#    plt.xticks([])
#    plt.yticks([])        
#    plt.subplot(2,5,k+6)
#    plt.imshow(train_y[k])
#    plt.title('train_y_%d'%(k+1))
#    plt.xticks([])
#    plt.yticks([])


# In[]
# initialize
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()

writer_tr = tf.summary.FileWriter(train_log_dir, sess.graph)
writer_te = tf.summary.FileWriter(test_log_dir)
merged = tf.summary.merge_all() 

# In[]:
# train
test_feed_dict={inputs_: test_x1, targets_: test_y1, keep_prob: 1.0, mask_prob:0.0}
time_start = time()

summaryWriter(writer_tr, writer_te, merged, cost, test_feed_dict, test_feed_dict, 0)
for e in range(1, 1+epochs):
    for batch_x, batch_y in my_io.batch_iter(batch_size, train_x, train_y, throw_insufficient=True):
        
        x = batch_x
        y = batch_y
            
        train_feed_dict = {inputs_: x, targets_: y, keep_prob: keep_prob_v, mask_prob: mask_prob_v}
        sess.run(optimizer, feed_dict=train_feed_dict)
        
    if e%2 == 0: 
        time_cost = time()-time_start
        summaryWriter(writer_tr, writer_te, merged, cost, train_feed_dict, test_feed_dict, e)
        
        res_imgs = sess.run(outputs_, feed_dict={inputs_: test_x1, targets_: test_y1,keep_prob:1.0, mask_prob: 0.0})
        # res_imgs = np.squeeze(res_imgs)

        #psnr
        pred_clip = np.clip(res_imgs.astype(np.float32), 0.0, 1.0)
        gt_clip   = np.clip(test_y1.squeeze().astype(np.float32), 0.0, 1.0)
        psnr_list, psnr_mean = psnr_batch(pred_clip, gt_clip, data_range=1.0)
        print(f'[VAL @ epoch {e}] mean PSNR = {psnr_mean:.3f} dB')

        with open(os.path.join(test_log_dir, f'val_psnr_epoch{e}.txt'), 'w') as f:
            for i, v in enumerate(psnr_list):
                f.write(f'{i}\t{v:.3f}\n')
            f.write(f'\nMEAN\t{psnr_mean:.3f}\n')
        #psnr fin

        data_save = {'reconstructed': res_imgs}
        my_io.save_mat(os.path.join(test_log_dir, f'val_imgs_epoch{e}.mat'), {'reconstructed': res_imgs})
        print('Time:', time_cost, '   Reconstruction test data saved to :',test_log_dir + '\n')    
            
    if e%20 == 0 and e!=0:
        saver.save(sess, model_path+'my_model',global_step=e, write_meta_graph=False)
        # saver.save(sess,model_path+'my_model') 
        print('epoch %d model saved to:'%e, model_path+'my_model\n')


summaryWriter(writer_tr, writer_te, merged, cost, train_feed_dict, test_feed_dict, e)
    
saver.save(sess,model_path+'my_model') 
print('epoch: %d model saved to:'%e, model_path+'my_model') 

# In[61]:
# test
start = 0
end = len(test_x)-1
idx = np.linspace(start, end, 10).astype('int32')  # show 10 results at equal intervals

in_imgs = test_x[idx]
gt_imgs = test_y[idx]  

reconstructed = sess.run(outputs_, feed_dict={inputs_: in_imgs, keep_prob: 1.0, mask_prob:0.0})
# reconstructed = np.squeeze(reconstructed)

    
fig, axes = plt.subplots(nrows=3, ncols=10, sharex=True, sharey=True, figsize=(20,4))
for images, row in zip([in_imgs, reconstructed, gt_imgs], axes):
    for img, ax in zip(images, row):
        ax.imshow(img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
fig.tight_layout(pad=0.1)
plt.show()

# In[ ]:
# release
sess.close()







