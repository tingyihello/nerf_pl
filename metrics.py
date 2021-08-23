import torch
from kornia.losses import ssim as dssim
import lpips

def mse(image_pred, image_gt, valid_mask=None, reduction='mean'):
    value = (image_pred-image_gt)**2
    if valid_mask is not None:
        value = value[valid_mask]
    if reduction == 'mean':
        return torch.mean(value)
    return value

def psnr(image_pred, image_gt, valid_mask=None, reduction='mean'):
    return -10*torch.log10(mse(image_pred, image_gt, valid_mask, reduction))

def ssim(image_pred, image_gt, reduction='mean'):
    """
    image_pred and image_gt: (1, 3, H, W)
    """
    dssim_ = dssim(image_pred, image_gt, 3, reduction) # dissimilarity in [0, 1]
    return 1-2*dssim_ # in [-1, 1]

def LPIPS(image_pred, image_gt, alex=None):
    image_pred = image_pred.permute(3, 1, 2)
    image_gt = image_gt.cpu().permute(3, 1, 2)
    
    if alex is not None:
        loss_fn = lpips.LPIPS(net='alex')
    else:
        loss_fn = lpips.LPIPS(net='vgg')

    return loss_fn(image_pred, image_gt)