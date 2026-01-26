import math
import os


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.utils import make_grid
import imageio
import matplotlib.pyplot as plt
try:
    import accimage
except ImportError:
    accimage = None


def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def to_pil_image(pic, mode=None):

    if not (_is_numpy_image(pic) or _is_tensor_image(pic)):
        raise TypeError("pic should be Tensor or ndarray. Got {}.".format(type(pic)))

    npimg = pic
    if isinstance(pic, torch.FloatTensor):
        pic = pic.mul(255).byte()
    if torch.is_tensor(pic):
        npimg = np.transpose(pic.numpy(), (1, 2, 0))

    if not isinstance(npimg, np.ndarray):
        raise TypeError(
            "Input pic must be a torch.Tensor or NumPy ndarray, "
            + "not {}".format(type(npimg))
        )

    if npimg.shape[2] == 1:
        expected_mode = None
        npimg = npimg[:, :, 0]
        if npimg.dtype == np.uint8:
            expected_mode = "L"
        if npimg.dtype == np.int16:
            expected_mode = "I;16"
        if npimg.dtype == np.int32:
            expected_mode = "I"
        elif npimg.dtype == np.float32:
            expected_mode = "F"
        if mode is not None and mode != expected_mode:
            raise ValueError(
                "Incorrect mode ({}) supplied for input type {}. Should be {}".format(
                    mode, np.dtype, expected_mode
                )
            )
        mode = expected_mode

    elif npimg.shape[2] == 4:
        permitted_4_channel_modes = ["RGBA", "CMYK"]
        if mode is not None and mode not in permitted_4_channel_modes:
            raise ValueError(
                "Only modes {} are supported for 4D inputs".format(
                    permitted_4_channel_modes
                )
            )

        if mode is None and npimg.dtype == np.uint8:
            mode = "RGBA"
    else:
        permitted_3_channel_modes = ["RGB", "YCbCr", "HSV"]
        if mode is not None and mode not in permitted_3_channel_modes:
            raise ValueError(
                "Only modes {} are supported for 3D inputs".format(
                    permitted_3_channel_modes
                )
            )
        if mode is None and npimg.dtype == np.uint8:
            mode = "RGB"

    if mode is None:
        raise TypeError("Input type {} is not supported".format(npimg.dtype))

    return Image.fromarray(npimg, mode=mode)


def to_tensor(pic):
    if not (_is_pil_image(pic) or _is_numpy_image(pic)):
        raise TypeError("pic should be PIL Image or ndarray. Got {}".format(type(pic)))

    if isinstance(pic, np.ndarray):
        # handle numpy array
        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        # backward compatibility
        return img.float().div(255)

    if accimage is not None and isinstance(pic, accimage.Image):
        nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
        pic.copyto(nppic)
        return torch.from_numpy(nppic)

    # handle PIL Image
    if pic.mode == "I":
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == "I;16":
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == "YCbCr":
        nchannel = 3
    elif pic.mode == "I;16":
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float().div(255)
    else:
        return img

import fastmri
def get_xs_s_mean(x0,mask_t):
    x0 = x0.to(mask_t.device)
    # x0对应的kspace
    x_k = fastmri.fft2c(x0.permute(0,2,3,1))
    # ksapce下采样
    xt_kmean = x_k * mask_t
    w = int(torch.sqrt(mask_t.sum()/2))
    w = int((320-w)/2)
    if w !=0:
        xt_kmean = xt_kmean[:,w:-w,w:-w,:]
    # 你傅里叶变换
    xt_mean = fastmri.ifft2c(xt_kmean)
    # 交换通道
    xt_mean = xt_mean.permute(0,3,1,2)
    return xt_mean

def get_xs_mean(x0,mask_t):
    x0 = x0.to(mask_t.device)
    # x0对应的kspace
    x_k = fastmri.fft2c(x0.permute(0,2,3,1))
    # ksapce下采样
    xt_kmean = x_k * mask_t
    w = int(torch.sqrt(mask_t.sum()/2))
    w = int((320-w)/2)
    # 你傅里叶变换
    xt_mean = fastmri.ifft2c(xt_kmean)
    # 交换通道
    xt_mean = xt_mean.permute(0,3,1,2)
    return xt_mean

def tensor2mriimg(tensor, out_type=np.uint8, min_max=(0, 1)):
    """
    将张量换为图像数据
    """
    tensor = tensor.squeeze().float().cpu()
    n_dim = tensor.dim()
    #print(n_dim)
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()

        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    # 对于我们的MRI图像，很明显是着一个
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.abs(img_np[0]+img_np[1]*1j)
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            "Only support 4D, 3D and 2D tensor. But received with dimension: {:d}".format(
                n_dim
            )
        )
    
    return img_np





def save_mriimg(img,img_path):
    plt.imsave(img_path,img,cmap="gray")


def save_mri(img,img_path):
    imageio.imsave(img_path,img)



# TODO 关于计算psnr的方式不知道对不对
def cal_psnr_old(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * math.log10(1.0 / math.sqrt(mse))

import lpips
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

mylpips = lpips.LPIPS(net='vgg')
def cal_lpips(a,b):
    
    a=torch.from_numpy(a)
    b = torch.from_numpy(b)

    return mylpips(a,b).item()

def cal_psnr(img1, img2,peak = 'max'):
    '''
    img1 re
    img2 gt
    '''
    img1 = np.abs(img1)
    img2 = np.abs(img2)
    mse = np.mean(np.abs(img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    if peak =='max':
        return 10 * np.log10(np.max(np.abs(img2))**2/mse)
    else :
        return 10*np.log10(1./mse)

def cal_psnr_s(img1, img2,peak = 'max'):

    img1 = np.abs(img1)
    img2 = np.abs(img2)
    mse = np.mean(np.abs(img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    if peak =='max':
        return 10 * np.log10(np.max(np.abs(img2))**2/mse)
    else :
        return 10*np.log10(1./mse)

def cal_mse(img1, img2,peak = 'max'):
    '''
    img1 re
    img2 gt
    '''
    img1 = np.abs(img1)
    img2 = np.abs(img2)
    mse = np.mean(np.abs(img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    if peak =='max':
        return 10 * np.log10(np.max(np.abs(img2))**2/mse)
    else :
        return 10*np.log10(1./mse)


from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def cal_ssim( pred,gt):
    """
    Compute Structural Similarity (SSIM), gt and pred are with shape of (C, H, W).
    """
    #channel_axis=0  data_range=gt.max()
    return structural_similarity(
        gt, pred,
        data_range=gt.max()   
        
    )

def cal_nrmse(img1, img2):
    '''
    img1 re
    img2 gt
    '''
    img1 = np.abs(img1)
    img2 = np.abs(img2)
    mse = np.mean(np.abs(img1 - img2) ** 2)
    denom = np.sqrt(np.mean((img1 * img1), dtype=np.float64))
    # mse = np.mean(np.abs(x - y) ** 2)
    out = np.sqrt(mse) / denom
    return out