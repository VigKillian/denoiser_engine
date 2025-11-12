# -*- coding: utf-8 -*-
'''
【predictor for ConvDAE】
usage：
contents to modify：model path, model dir; SAVE_FLAG; 
in_imgs,  gt_imgs; keep_prob, feed_dict; outputs (according to act_out_fun)

notes：
Different models' "outputs_" could be different. Sometimes it won't report errors, 
but the result is not as expected.In this case, return to your model definition and confirm your "outputs_"

Author: zhihong (z_zhi_hong@163.com)
Date: 20190505
Modified: zhihong_20190809

'''
print('>>> predictor start', flush=True)
# In[]:
# import modules
import os
import numpy as np
from tf_compat import tf
from time import time
import matplotlib.pyplot as plt
import matplotlib.image as plt_img
from util import my_io
#from util import my_img_evaluation as my_evl
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import config as cfg
from skimage.metrics import peak_signal_noise_ratio as psnr
import glob, cv2, numpy as np, os

# In[] 
# environment config
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
config.gpu_options.per_process_gpu_memory_fraction = 0.8


# In[]:
# graph reset
tf.reset_default_graph()  


# In[]:
# parameters config
# setting 
SAVE_FLAG = 1   # flag of saving the outputs of the network's prediction
test_batch_size = 100
pic_size = list(cfg.pic_size)
# pic_size = [28,28] # picture size, for N-MNIST

# path
data_path = "./dataset/single_molecule_localization/sml_test.mat" # for sml
# data_path = "./dataset/N_MNIST_pic/N_MNIST_pic_test.mat" # for N-MNIST

root_model_path = "model_data/"  # model's root dir
model_dir = "bsc-ConvDAE(unsup)--02-24_10-33" # model'saving dir
model_path = root_model_path + model_dir + "/"

# model
model_name = 'my_model' #model's name
model_ind = -1  #model's index

pred_res_path = './predict_res/' + data_path.split('/')[-1][0:-4] + "-" + model_dir + "/" # dir of the prediction results
if not os.path.isdir(pred_res_path) and SAVE_FLAG:
   os.makedirs(pred_res_path)


# In[]
# load model
# sess = tf.Session()
sess = InteractiveSession(config=config)
sess.run(tf.global_variables_initializer())

restorer = tf.train.import_meta_graph(model_path+model_name+'.meta')

ckpt = tf.train.get_checkpoint_state(model_path)
if ckpt:
    ckpt_states = ckpt.all_model_checkpoint_paths
    restorer.restore(sess, ckpt_states[model_ind])
    
# load Ops and variables according to old model and your need
graph = tf.get_default_graph()
inputs_ = graph.get_tensor_by_name("inputs/inputs_:0")
mask_prob = graph.get_tensor_by_name("inputs/Placeholder_1:0")
targets_ = graph.get_tensor_by_name("inputs/targets_:0")
keep_prob = graph.get_tensor_by_name("inputs/Placeholder:0")  # for dropout

outputs_ = graph.get_tensor_by_name("outputs/outputs_:0") # for act_fun = relu
# outputs_ = graph.get_tensor_by_name("outputs/conv2d/Relu:0") 
# outputs_ = graph.get_tensor_by_name("outputs/conv2d/Tanh:0") #for act_fun = tanh

cost = graph.get_tensor_by_name("loss/Mean:0")


# In[]:
# load data
import glob, os
from skimage.io import imread
from skimage.transform import resize

from config import list_images, load_gray_resized
test_noisy_dir = cfg.val_dir + "/noisy"
in_imgs = load_gray_resized(list_images(test_noisy_dir), pic_size)
print('pic_test_x: ', in_imgs.shape)


# In[]:
# prediction
ind = 0
mean_cost = 0
time_cost = 0
reconstructed = np.zeros(in_imgs.shape, dtype='float32')
for batch_x, _ in my_io.batch_iter(test_batch_size,in_imgs, in_imgs, shuffle=False):
    x = batch_x.reshape((-1, *pic_size, 1))
    feed_dict = {inputs_: x, keep_prob:1.0, mask_prob:0.0} #for dropout
#    feed_dict = {inputs_: x, targets_: y}  #for non dropout
    
    time1 = time()
    res_imgs = sess.run(outputs_, feed_dict=feed_dict)
    time2 = time()
    time_cost += (time2 - time1)
    res_imgs = np.squeeze(res_imgs)
    reconstructed[ind*test_batch_size:(ind+1)*test_batch_size] = res_imgs
    ind += 1
time_cost = time_cost/len(in_imgs)
print('\nmean time cost(ms):%f\n'%(time_cost*1e3))


# In[]:
# save the prediction results
if SAVE_FLAG:
#    np.save(pred_res_path+'pred_res',reconstructed)   # save pics in the format of .npy
#    print('\nreconstruction data saved to : \n',pred_res_path+'pred_res.npy' ) 

    data_save = {'reconstructed': reconstructed} # save pics in the format of .mat
    my_io.save_mat(pred_res_path +'recon.mat', data_save) 
     
    for i in range(len(reconstructed)): # save pics in the format of .png
        plt_img.imsave(pred_res_path+'png'+str(i)+'.png', reconstructed[i], cmap=plt.cm.gray)
    print('\nreconstruction data saved to : \n',pred_res_path)
    

gt_dir = os.path.join(cfg.val_dir, "imgs")
gt_paths = sorted(glob.glob(os.path.join(gt_dir, "*")))
N = min(len(gt_paths), len(reconstructed))
if N == 0:
    print("[PSNR] not find GT img, skip PSNR calculate. gt_dir=", gt_dir)
else:
    psnrs = []
    names = []
    for i in range(N):
        gt = cv2.imread(gt_paths[i], 0)
        if gt is None:
            continue
        gt = gt.astype(np.float32) / 255.0

        if gt.shape != tuple(pic_size):
            from skimage.transform import resize
            gt = resize(gt, tuple(pic_size), anti_aliasing=True).astype(np.float32)

        rec = np.clip(reconstructed[i].astype(np.float32), 0.0, 1.0)

        val = psnr(gt, rec, data_range=1.0)
        psnrs.append(val)
        names.append(os.path.basename(gt_paths[i]))

    if psnrs:
        mean_psnr = float(np.mean(psnrs))
        print(f"[PSNR] number of exemple N={len(psnrs)} | average PSNR = {mean_psnr:.3f} dB")

        try:
            os.makedirs(pred_res_path, exist_ok=True)
            with open(os.path.join(pred_res_path, "psnr.txt"), "w") as f:
                for n, p in zip(names, psnrs):
                    f.write(f"{n}\t{p:.3f}\n")
                f.write(f"\nMEAN\t{mean_psnr:.3f}\n")
            print(f"[PSNR] result wrote in {os.path.join(pred_res_path,'psnr.txt')}")
        except Exception as e:
            print("[PSNR] fail to write ficher: ", e)
    else:
        print("[PSNR] not find exemple")


# In[]:
# illustrate the results
start = 0
end = len(reconstructed)-1
idx = np.linspace(start, end, 10).astype('int32')  # show 10 results at equal intervals

in_images = in_imgs[idx]
recon_images = reconstructed[idx]

fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(20,4))
for images, row in zip([in_images, recon_images], axes):
    for img, ax in zip(images, row):
        ax.imshow(img.reshape((*pic_size)), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
fig.tight_layout(pad=0.1)
plt.show()

# In[24]:
# release
sess.close()





