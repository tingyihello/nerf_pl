#### Train on a synthetic scene
```
python train.py \
   --dataset_name blender \
   --root_dir /home/sy/nerf/data/nerf_synthetic/$SCENE_NAME \
   --N_importance 64 --img_wh 400 400 --noise_std 0 \
   --num_epochs 16 --batch_size 1024 \
   --optimizer adam --lr 5e-4 \
   --lr_scheduler steplr --decay_step 2 4 8 --decay_gamma 0.5 \
   --exp_name exp
```
#### Train on llff
```
python train.py \
   --dataset_name llff \
   --root_dir /home/sy/nerf/data/nerf_llff_data/$SCENE_NAME \
   --N_importance 64 --img_wh 504 378 \
   --num_epochs 30 --batch_size 1024 \
   --optimizer adam --lr 5e-4 \
   --lr_scheduler steplr --decay_step 10 20 --decay_gamma 0.5 \
   --exp_name exp
   # --ckpt_path /home/sy/nerf/model/nerf_pl
```

## the default path of saved model
```
./ckpts/exp/
```

### Test on a scene
```
python eval.py \
   --root_dir /home/sy/nerf/data/nerf_synthetic/$SCENE_NAME \
   --dataset_name blender --scene_name lego \
   --img_wh 400 400 --N_importance 64 --ckpt_path $CKPT_PATH
```

# the path of result
./results/{dataset_name}/{scene_name}


#### Train on our dataset, for exapmle *IMG_0064*
```
python train.py \
   --dataset_name llff \
   --root_dir /home/sy/nerf/data/IMG_0064/preprocess/sfm \
   --N_importance 64 --img_wh 960 540 \
   --num_epochs 6 --batch_size 1024 \
   --optimizer adam --lr 5e-4 \
   --lr_scheduler steplr --decay_step 10 20 --decay_gamma 0.5 \
   --exp_name IMG_0064
   # --ckpt_path /home/sy/nerf/model/nerf_pl
```

## the default path of saved model

```
./ckpts/IMG_0064/
```


### Test on our dataset, for example *IMG_0064*
```
python eval.py \
   --root_dir /home/sy/nerf/data/IMG_0064/preprocess/sfm \
   --dataset_name llff --scene_name IMG_0064 \
   --img_wh 960 540 --N_importance 64 --ckpt_path $CKPT_PATH
```

# the saved path of result
```
./results/{dataset_name}/{scene_name}
```

