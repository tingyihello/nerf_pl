import imageio
import cv2
import os
import numpy as np

# data_dir = "/home/sy/nerf/code/IBRNet_depth/eval/pose_test/eval_pose_toys/chairs_044800"
data_dir = '/data/sy/nerf_pose_test/1/2021-07-31/train/eval/2021-07-31-epoch=14/circles'
imgs_path = [os.path.join(data_dir, f) for f in sorted(os.listdir(data_dir)) if f.startswith('rgb_fine') and not f.startswith('disp')]
imgs = []
print("len:   ", len(imgs_path))
for i in range(len(imgs_path)):
    # imgs.append(imageio.imread(imgs_path[i]).astype(np.uint8))
    
    # img_path = data_dir+'/'+str(i)+'_pred_fine.png'
    # print(img_path)
    imgs.append(imageio.imread(imgs_path[i]).astype(np.uint8))

# path_save=os.path.join(data_dir,'lego_rgb.mp4')
# imageio.mimsave(path_save,imgs,fps=30,quality=8)
imageio.mimsave(os.path.join(data_dir, 'sfm.gif'), imgs, fps=20)