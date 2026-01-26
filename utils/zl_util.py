"""Abstract SDE classes, Reverse SDE, and VE/VP SDEs."""
import abc
import torch
import numpy as np
import fastmri
import math
from tqdm import tqdm
import torchvision.utils as tvul
import os
import matplotlib.pyplot as plt
from utils.img_util import *
import matplotlib.pyplot as plt

import torchvision


class ZLDPM:
    def __init__(self,N,max_sigma,noise_schedule='consine',eps=0.01,device=None):
      self.N = N
      self.dt = 1 / N
      self.device = device
      self.max_sigma = max_sigma
      
      # 初始化一些设置，比如说 mask的list 以及每次加的噪声强度 还有 逆向重建过程所用的dt
      self._initialize(self.max_sigma,N,noise_schedule,eps)

    def _initialize(self, max_sigma, N, schedule, eps=0.01):
        
        ###############################
        # mask list的生成
        ###############################

        # 生成mask对应宽度的list
        def gen_sequence(start, end, step):
            ############################################
            #### generate_Arithmetic progression  ####
            ############################################
            sequence = torch.arange(start, end-step, -step )
            return sequence

        # 生成单个 mask
        def get_center_mask(im_shape, pct):
            ############################################
            ####    generate_center_mask    ####
            ############################################

            mask = np.zeros(im_shape)
            half_pct = (pct / 2)
            center = [int(x / 2) for x in im_shape]

            if len(im_shape) == 3:
                mask[center[0] - math.ceil(half_pct):math.ceil(center[0] + half_pct),
                center[1] - math.ceil( half_pct):math.ceil(center[1] + half_pct),
                center[2] - math.ceil(half_pct):math.ceil(center[2] + half_pct)] = 1

            elif len(im_shape) == 2:
                mask[center[0] - math.ceil(half_pct):math.ceil(center[0] +  half_pct),
                center[1] - math.ceil(half_pct):math.ceil(center[1] + half_pct)] = 1
            return mask

        # 生成所用的mask list 注意这里的mask 我已经自己设置成了 (1,320,320,2)
        def gen_mask(image_shape):
            ############################################
            ####    generate_all_center_mask    ####
            ############################################
            ####所有的mask存在一个mask数组里
            mask =  []
            total_scale = gen_sequence(320, 32, 2) #源代码 40
            for i in range(len(total_scale)):
                m = get_center_mask(image_shape, total_scale[i])
                m = torch.from_numpy(m)
                mm = torch.stack((m,m), dim = 0).view(1,2,320,320)
                mm = mm.permute(0,2,3,1)
                mm = mm.float()
                mask.append(mm)
            return mask


        #############################################
        # 生成对应的方差 
        ##############################################

        # 常数 theta
        def constant_theta_schedule(timesteps, v=1.):
            """
            constant schedule
            """
            print('constant schedule')
            timesteps = timesteps + 1 # T from 1 to 100
            return torch.ones(timesteps, dtype=torch.float32)

        # 线性 theta
        def linear_theta_schedule(timesteps):
            """
            linear schedule
            """
            print('linear schedule')
            timesteps = timesteps + 1 # T from 1 to 100
            scale = 1000 / timesteps
            beta_start = scale * 0.0001
            beta_end = scale * 0.02
            return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)

        # cosine theta
        def cosine_theta_schedule(timesteps, s = 0.008):
            """
            cosine schedule
            """
            print('cosine schedule')
            timesteps = timesteps + 2 # for truncating from 1 to -1
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - alphas_cumprod[1:-1]
            return betas

        # 累和
        def get_thetas_cumsum(thetas):
            return torch.cumsum(thetas, dim=0)

        # 得到sigama
        def get_sigmas(thetas):
            return torch.sqrt(max_sigma**2 * 2 * thetas)

        # 得到 sigma的积分
        def get_sigma_bars(thetas_cumsum):
            return torch.sqrt(max_sigma**2 * (1 - torch.exp(-2 * thetas_cumsum * self.dt)))
        
        # 判断需要那种 theta
        if schedule == 'cosine':
            thetas = cosine_theta_schedule(N)
        elif schedule == 'linear':
            thetas = linear_theta_schedule(N)
        elif schedule == 'constant':
            thetas = constant_theta_schedule(N)
        else:
            print('Not implemented such schedule yet!!!')


        sigmas = get_sigmas(thetas)
        thetas_cumsum = get_thetas_cumsum(thetas) - thetas[0] # for that thetas[0] is not 0
        self.dt = -1 / thetas_cumsum[-1] * math.log(eps)
        sigma_bars = get_sigma_bars(thetas_cumsum)
        
        self.thetas = thetas.to(self.device)
        self.sigmas = sigmas.to(self.device)
        self.thetas_cumsum = thetas_cumsum.to(self.device)
        self.sigma_bars = sigma_bars.to(self.device)

        self.model = None
        self.mask = gen_mask((320,320))

    def get_xt_mean(self,x0,mask_t):
        # x0对应的kspace
        x_k = fastmri.fft2c(x0.permute(0,2,3,1))
        # ksapce下采样
        xt_kmean = x_k * mask_t
        # 你傅里叶变换
        xt_mean = fastmri.ifft2c(xt_kmean)
        # 交换通道
        xt_mean = xt_mean.permute(0,3,1,2)
        return xt_mean

    def get_xt_mean_k(self,x0,mask_t):
        # x0对应的kspace
        x_k = fastmri.fft2c(x0.permute(0,2,3,1))
        # ksapce下采样
        xt_kmean = x_k * mask_t
        # 交换通道
        xt_kmean = xt_kmean.permute(0,3,1,2)
        return xt_kmean

    def set_mask(self,t):
        mask_t = self.mask[t]
        mask_t = mask_t.to(self.device)
        return mask_t

    # 得到k空间处理的噪声
    def get_xt_noise(self,noise,mask_t):
        # 噪声kspace
        n_k = fastmri.fft2c(noise.permute(0,2,3,1))
        # 噪声补欠采样
        nt_k = (torch.ones_like(mask_t) -mask_t)*n_k 
        # 你傅里叶变换
        noise_t = fastmri.ifft2c(nt_k)
        # 交换通道
        noise_t = noise_t.permute(0,3,1,2)
        return noise_t

    # 得到图像域的噪声
    def get_xt_noisei(self,noise,mask_t):
        noise_t = torch.randn_like(noise)
        return noise_t

    

    def get_xt(self,x0,t):
        '''考虑到fft带来的一些线性误差,这里为一步直接生成xt的函数'''

        # mask
        mask_t = self.mask[t]
        mask_t = mask_t.to(self.device)

        # At(x0)
        x_k = fastmri.fft2c(x0.permute(0,2,3,1))
        xt_kmean = x_k * mask_t
        # xt_mean = fastmri.ifft2c(xtk)
        # xt_mean = xt_mean.permute(0,3,1,2)


        # (I-At)z
        noise = torch.randn_like(x0)
        n_k = fastmri.fft2c(noise.permute(0,2,3,1))
        nt_k = (torch.ones_like(mask_t) -mask_t)*n_k 
        # noise_t = fastmri.ifft2c(nk_t)
        # noise_t = noise_t.permute(0,3,1,2)


        # sigma_t 
        sigma_t = self.get_sigma(t)

        # 相加并作逆变换
        xt_k = xt_kmean + sigma_t * nt_k
        xt = fastmri.ifft2c(xt_k)
        
        xt = xt.permute(0,3,1,2)

        return xt

    def get_xt1(self,x0,t):
        '''考虑到fft带来的一些线性误差,这里为一步直接生成xt的函数'''

        # mask
        mask_t = self.mask[t]
        mask_t = mask_t.to(self.device)
        # sigma_t 
        sigma_t = self.get_sigma(t)
        # At(x0)


        x_k = fastmri.fft2c(x0.permute(0,2,3,1))
        xt_kmean = x_k * mask_t

        if t ==0:
            xt_kmean_samll = xt_kmean
        else:
            xt_kmean_samll = xt_kmean[:,t:-t,t:-t,:]

        xt_small = fastmri.ifft2c(xt_kmean_samll).permute(0,3,1,2)
        
        noise = sigma_t * torch.randn_like(x0)
        n_k = fastmri.fft2c(noise.permute(0,2,3,1))
        xt_k = xt_kmean# + n_k

        xt_k = xt_k.permute(0,3,1,2)

        xt_mean = fastmri.ifft2c(xt_kmean).permute(0,3,1,2)
        xt = xt_mean #+  noise
        return xt,xt_k,xt_small
    
    # TODO
    def get_sigma(self,t):
        
        sigma_t = self.sigma_bars[t]

        return sigma_t
    
    def set_model(self,model,teacher_model):
        teacher_model.eval()
        self.teacher_model = teacher_model
        self.model = model

    # 网络预测重建
    def pred_fn(self, xt, t):
        
        return self.model(xt, t)
    
    def teacher_pred_fn(self, xt, t):
        
        return self.teacher_model(xt, t)

    def get_score_from_x0(self,x0_pred,xt, t):
        # mask
        mask_t = self.mask[t]
        mask_t = mask_t.to(self.device)
        A_xt = self.get_xt_mean(x0_pred,mask_t)
        sigma_t = self.get_sigma(t)
        score = (A_xt-xt)/(sigma_t ** 2)
        return score
    

    def reverse_base_add_sde(self, xt, N=-1, save_states=False, save_dir='sde_state'):
        N = self.N if N < 0 else N
        b = xt.shape[0]
        # 开始重建
        for t in tqdm(reversed(range(1, N + 1))):
            time = t * torch.ones((b,)).long()
            time = time.to(self.device)
            timestepsnet = time.float()
            prev_t = t-1
            pred_x0 = self.pred_fn(xt,timestepsnet)
            pred_x0 = self.process_x0(pred_x0)

            # 得到t-1时刻的一些数据
            mask_t = self.set_mask(t)

            mask_t_prev = self.set_mask(prev_t)
            xt_prev_mean = self.get_xt_mean(pred_x0,mask_t=mask_t_prev)
            sigma_t_prev = self.get_sigma(prev_t)
            
            xt_mean = self.get_xt_mean(pred_x0,mask_t=mask_t)
            xt_mean1 = self.get_xt_mean(xt,mask_t=mask_t)

            noise = torch.randn_like(xt)

            # xt_prev_noise = self.get_xt_noise(noise,mask_t_prev)

            xt =xt_mean1 - xt_mean + xt_prev_mean + sigma_t_prev * noise

            # 保存中间状态 看一下
            if save_states: 
                if not os.path.exists(save_dir):
                        os.makedirs(save_dir, exist_ok=True)
                interval = self.N // 7 #存 7 张 
                if t % interval == 0:
                    idx = t // interval
                    
                    xt_img = torch.abs(xt[0,0]+xt[0,1]*1j).cpu().data.numpy()
                    plt.imsave( f'{save_dir}/mix_xt_{t}.png',xt_img, cmap="gray")
                    x0_img = torch.abs(pred_x0[0,0]+xt[0,1]*1j).cpu().data.numpy()
                    plt.imsave(f'{save_dir}/mix_x0_{t}.png', x0_img,cmap="gray")

        return xt


    def mix(self, xt, N=-1, save_states=False, save_dir='sde_state'):
        N = self.N if N < 0 else N
        b = xt.shape[0]
        # 开始重建
        for t in tqdm(reversed(range(1, N + 1))):
            time = t * torch.ones((b,)).long()
            time = time.to(self.device)
            timestepsnet = time.float()
            prev_t = t-1
            pred_x0 = self.pred_fn(xt,timestepsnet)
            pred_x0 = self.process_x0(pred_x0)

            # 得到t-1时刻的一些数据
            mask_t = self.set_mask(t)

            mask_t_prev = self.set_mask(prev_t)
            xt_prev_mean = self.get_xt_mean(pred_x0,mask_t=mask_t_prev)
            
            sigma_t_prev = self.get_sigma(prev_t)
            sigma_t = self.get_sigma(t)
            
            xt_mean = self.get_xt_mean(pred_x0,mask_t=mask_t)
            xt_mean1 = self.get_xt_mean(xt,mask_t=mask_t)

            noise= torch.rand_like(xt)
            xt =xt - xt_mean + xt_prev_mean - (sigma_t -  sigma_t_prev) * noise

            # xt_prev_noise = self.get_xt_noise(noise,mask_t_prev)

            #noise = xt-xt_mean
            #xt =xt_prev_mean + sigma_t_prev * noise / sigma_t

            # 保存中间状态 看一下
            if save_states: 
                if not os.path.exists(save_dir):
                        os.makedirs(save_dir, exist_ok=True)
                interval = self.N // 7 #存 7 张 
                if t % interval == 0:
                    idx = t // interval
                    
                    xt_img = torch.abs(xt[0,0]+xt[0,1]*1j).cpu().data.numpy()
                    plt.imsave( f'{save_dir}/mix_xt_{t}.png',xt_img, cmap="gray")
                    x0_img = torch.abs(pred_x0[0,0]+xt[0,1]*1j).cpu().data.numpy()
                    plt.imsave(f'{save_dir}/mix_x0_{t}.png', x0_img,cmap="gray")

        return xt

    # 整个逆向重建过程 base

    def mix_wucha(self, gt,xt, N=-1, save_states=False, save_dir='sde_state'):
        N = self.N if N < 0 else N
        b = xt.shape[0]
        # 开始重建
        for t in tqdm(reversed(range(1, N + 1))):
            time = t * torch.ones((b,)).long()
            time = time.to(self.device)
            timestepsnet = time.float()
            prev_t = t-1
            pred_x0 = self.pred_fn(xt,timestepsnet)
            pred_x0 = self.process_x0(pred_x0)

            # 得到t-1时刻的一些数据
            mask_t = self.set_mask(t)

            mask_t_prev = self.set_mask(prev_t)
            xt_prev_mean = self.get_xt_mean(pred_x0,mask_t=mask_t_prev)
            
            sigma_t_prev = self.get_sigma(prev_t)
            sigma_t = self.get_sigma(t)
            
            xt_mean = self.get_xt_mean(pred_x0,mask_t=mask_t)
            xt_mean1 = self.get_xt_mean(xt,mask_t=mask_t)

            noise = xt-xt_mean

            # xt_prev_noise = self.get_xt_noise(noise,mask_t_prev)

            xt =xt_prev_mean + sigma_t_prev * noise / sigma_t

            # 保存中间状态 看一下
            if save_states: 
                if not os.path.exists(save_dir):
                        os.makedirs(save_dir, exist_ok=True)
                interval = self.N // 7 #存 7 张 
                if t % interval == 0:
                    idx = t // interval
                    
                    xt_img = torch.abs(xt[0,0]+xt[0,1]*1j).cpu().data.numpy()
                    plt.imsave( f'{save_dir}/mix_xt_{t}.png',xt_img, cmap="gray")
                    x0_img = torch.abs(pred_x0[0,0]+xt[0,1]*1j).cpu().data.numpy()
                    plt.imsave(f'{save_dir}/mix_x0_{t}.png', x0_img,cmap="gray")

        return xt

    
    # 整个逆向重建过程 base
    def reverse_sde(self, xt, N=-1, save_states=False, save_dir='sde_state'):
        N = self.N if N < 0 else N
        b = xt.shape[0]
        # 开始重建
        for t in tqdm(reversed(range(1, N + 1))):
            time = t * torch.ones((b,)).long()
            time = time.to(self.device)
            timestepsnet = time.float()
            prev_t = t-1
            pred_x0 = self.pred_fn(xt,timestepsnet)
            pred_x0 = self.process_x0(pred_x0)
            score = self.get_score_from_x0(pred_x0,xt,t)
            # xt - xt-1
            xt = self.reverse_sde_onestep(xt,pred_x0, score, t,prev_t)

            # 保存中间状态 看一下
            if save_states: 
                if not os.path.exists(save_dir):
                        os.makedirs(save_dir, exist_ok=True)
                interval = self.N // 7 #存 7 张 
                if t % interval == 0:
                    idx = t // interval
                    
                    xt_img = torch.abs(xt[0,0]+xt[0,1]*1j).cpu().data.numpy()
                    plt.imsave( f'{save_dir}/sde_xt_{t}.png',xt_img, cmap="gray")
                    x0_img = torch.abs(pred_x0[0,0]+xt[0,1]*1j).cpu().data.numpy()
                    plt.imsave(f'{save_dir}/sde_x0_{t}.png', x0_img,cmap="gray")

        return xt

      # 整个逆向重建过程 plus带有反向投影版本
    # @torch.no_grad()
    def reverse_sde_plus(self, x,y,mask,N=-1, save_states=False, save_dir='sde_state',use_tv=True):
        lr  = 100
        N = self.N if N < 0 else N
        xt = x
        y = y
        b = xt.shape[0]
        # 开始重建
        for t in tqdm(reversed(range(1, N + 1))):
            #t = min(t,140)
            time = t * torch.ones((b,)).long()
            time = time.to(self.device)
            timestepsnet = time.float()
            prev_t = t-1
            pred_x0 = self.pred_fn(xt,timestepsnet)
            pred_x0 = self.process_x0(pred_x0)
            pred_x0 = self.inver_fp(pred_x0,mask,y)


            if use_tv:
                with torch.set_grad_enabled(True):
                    for i in range(5):
                        pred_x0.requires_grad_()
                        loss = tv_loss(pred_x0)
                        loss.backward()
                        with torch.no_grad():
                            pred_x0.add_(pred_x0.grad, alpha=-lr/2)
                        pred_x0.grad.zero_()
                        pred_x0.requires_grad_(False)
                pred_x0 = self.inver_fp(pred_x0,mask,y)



            score = self.get_score_from_x0(pred_x0,xt,t)
            
            # xt - xt-1
            xt = self.reverse_sde_onestep(xt,pred_x0, score, t,prev_t)

            #xt = self.inver_fp(xt,mask,y)
            # 保存中间状态 看一下
            if save_states: 
                interval = self.N // 14 #存 14 张 
                if t % interval == 0:
                    idx = t // interval
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir, exist_ok=True)
                    xt_img = torch.abs(xt[0,0]+xt[0,1]*1j).cpu().data.numpy()
                    plt.imsave( f'{save_dir}/sde_plus_xt_{t}.png',xt_img, cmap="gray")
                    x0_img = torch.abs(pred_x0[0,0]+xt[0,1]*1j).cpu().data.numpy()
                    plt.imsave(f'{save_dir}/sde_plus_x0_{t}.png', x0_img,cmap="gray")

        return xt
    
    def reverse_sde_plus_add(self, x,y,mask,N=-1, save_states=False, save_dir='sde_state',use_tv=True):
        lr  = 100
        N = self.N if N < 0 else N
        xt = x
        y = y
        b = xt.shape[0]
        # 开始重建
        condition = y.float()
        for t in tqdm(reversed(range(1, N + 1))):
            mask_t = t
            t = min(t,140)
            time = t * torch.ones((b,)).long()
            time = time.to(self.device)
            timestepsnet = time.float()
            prev_t = t-1

            mask_time = mask_t * torch.ones((b,)).long()
            mask_time = mask_time.to(self.device)
            mask_timestepsnet = mask_time.float()
            netinput = torch.cat([xt.float(),condition],dim=1)
            pred_x0 = self.pred_fn(netinput,mask_timestepsnet)
            pred_x0 = self.process_x0(pred_x0)
            pred_x0 = self.inver_fp(pred_x0,mask,y)


            if use_tv:
                with torch.set_grad_enabled(True):
                    for i in range(5):
                        pred_x0.requires_grad_()
                        loss = tv_loss(pred_x0)
                        loss.backward()
                        with torch.no_grad():
                            pred_x0.add_(pred_x0.grad, alpha=-lr/2)
                        pred_x0.grad.zero_()
                        pred_x0.requires_grad_(False)
                pred_x0 = self.inver_fp(pred_x0,mask,y)



            score = self.get_score_from_x0(pred_x0,xt,t)
            
            # xt - xt-1
            xt = self.reverse_sde_onestep_add(xt,pred_x0, score, t,prev_t,mask_t)

            #xt = self.inver_fp(xt,mask,y)
            # 保存中间状态 看一下
            if save_states: 
                interval = self.N // 14 #存 14 张 
                if t % interval == 0:
                    idx = t // interval
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir, exist_ok=True)
                    xt_img = torch.abs(xt[0,0]+xt[0,1]*1j).cpu().data.numpy()
                    plt.imsave( f'{save_dir}/sde_plus_xt_{t}.png',xt_img, cmap="gray")
                    x0_img = torch.abs(pred_x0[0,0]+xt[0,1]*1j).cpu().data.numpy()
                    plt.imsave(f'{save_dir}/sde_plus_x0_{t}.png', x0_img,cmap="gray")

        return xt

    def reverse_sde_plus_add_condition(self, x,y,mask,N=-1, save_states=False, save_dir='sde_state',use_tv=True):
        lr  = 100
        N = self.N if N < 0 else N
        xt = x
        y = y
        b = xt.shape[0]
        condition = y.float()
        # 开始重建
        for t in tqdm(reversed(range(1, N + 1))):
            mask_t = t
            t = min(t,140)
            time = t * torch.ones((b,)).long()
            time = time.to(self.device)
            timestepsnet = time.float()
            prev_t = t-1

            mask_time = mask_t * torch.ones((b,)).long()
            mask_time = mask_time.to(self.device)
            mask_timestepsnet = mask_time.float()
            netinput = torch.cat([xt.float(),condition.float()],dim=1)
            pred_x0 = self.pred_fn(netinput,mask_timestepsnet)
            pred_x0 = self.process_x0(pred_x0)
            pred_x0 = self.inver_fp(pred_x0,mask,y)


            if use_tv:
                with torch.set_grad_enabled(True):
                    for i in range(5):
                        pred_x0.requires_grad_()
                        loss = tv_loss(pred_x0)
                        loss.backward()
                        with torch.no_grad():
                            pred_x0.add_(pred_x0.grad, alpha=-lr/2)
                        pred_x0.grad.zero_()
                        pred_x0.requires_grad_(False)
                pred_x0 = self.inver_fp(pred_x0,mask,y)

            mask_t_prev = self.mask[mask_t-1].to(self.device)
            condition = self.get_xt_mean(pred_x0,mask_t_prev)
            score = self.get_score_from_x0(pred_x0,xt,t)
            
            # xt - xt-1
            xt = self.reverse_sde_onestep_add(xt,pred_x0, score, t,prev_t,mask_t)

            #xt = self.inver_fp(xt,mask,y)
            # 保存中间状态 看一下
            if save_states: 
                interval = self.N // 14 #存 14 张 
                if t % interval == 0:
                    idx = t // interval
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir, exist_ok=True)
                    xt_img = torch.abs(xt[0,0]+xt[0,1]*1j).cpu().data.numpy()
                    plt.imsave( f'{save_dir}/sde_plus_xt_{t}.png',xt_img, cmap="gray")
                    x0_img = torch.abs(pred_x0[0,0]+xt[0,1]*1j).cpu().data.numpy()
                    plt.imsave(f'{save_dir}/sde_plus_x0_{t}.png', x0_img,cmap="gray")

        return xt

    def reverse_sde_plus_add_speed(self, x,y,mask,N=-1,skip=1, save_states=False, save_dir='sde_state',use_tv=True):
        lr  = 100
        N = self.N if N < 0 else N
        num_point = N // skip
        use_time = np.linspace(1, N, num_point, endpoint=True)
        use_time = [ round(_) for _ in use_time]
        pre_time = use_time.copy()
        pre_time.insert(0,0)
        pre_time.pop()
        xt = x
        y = y
        b = xt.shape[0]
        # 开始重建
        for t,prev_t in tqdm(zip(reversed(use_time),reversed(pre_time))):
            mask_t = t
            t = min(t,140)
            time = t * torch.ones((b,)).long()
            time = time.to(self.device)
            timestepsnet = time.float()

            mask_time = mask_t * torch.ones((b,)).long()
            mask_time = mask_time.to(self.device)
            mask_timestepsnet = mask_time.float()
            pred_x0 = self.pred_fn(xt,mask_timestepsnet)
            pred_x0 = self.process_x0(pred_x0)
            pred_x0 = self.inver_fp(pred_x0,mask,y)


            if use_tv:
                with torch.set_grad_enabled(True):
                    for i in range(5):
                        pred_x0.requires_grad_()
                        loss = tv_loss(pred_x0)
                        loss.backward()
                        with torch.no_grad():
                            pred_x0.add_(pred_x0.grad, alpha=-lr/2)
                        pred_x0.grad.zero_()
                        pred_x0.requires_grad_(False)
                pred_x0 = self.inver_fp(pred_x0,mask,y)



            score = self.get_score_from_x0(pred_x0,xt,t)
            
            # xt - xt-1
            xt = self.reverse_sde_onestep_add(xt,pred_x0, score, t,prev_t,mask_t)

            #xt = self.inver_fp(xt,mask,y)
            # 保存中间状态 看一下
            if save_states: 
                interval = self.N // 14 #存 14 张 
                if t % interval == 0:
                    idx = t // interval
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir, exist_ok=True)
                    xt_img = torch.abs(xt[0,0]+xt[0,1]*1j).cpu().data.numpy()
                    plt.imsave( f'{save_dir}/sde_plus_xt_{t}.png',xt_img, cmap="gray")
                    x0_img = torch.abs(pred_x0[0,0]+xt[0,1]*1j).cpu().data.numpy()
                    plt.imsave(f'{save_dir}/sde_plus_x0_{t}.png', x0_img,cmap="gray")

        return xt

    def reverse_sde_plus_add_speedadd(self, x,y,mask,N=-1,skip=1, mid_time=80,skipadd=1,save_states=False, save_dir='sde_state',use_tv=True):
        lr  = 100
        N = self.N if N < 0 else N
        num_point1 = mid_time // skip
        num_point2 = (N-mid_time) // skipadd
        use_time_q = np.linspace(1, mid_time, num_point1, endpoint=True).tolist()
        use_time_h = np.linspace( mid_time, N,num_point2, endpoint=True).tolist()
        use_time_q.pop()
        use_time = use_time_q + use_time_h
        use_time = [ round(_) for _ in use_time]
        pre_time = use_time.copy()
        pre_time.insert(0,0)
        pre_time.pop()
        xt = x
        y = y
        b = xt.shape[0]
        # 开始重建
        for t,prev_t in tqdm(zip(reversed(use_time),reversed(pre_time))):
            mask_t = t
            t = min(t,140)
            time = t * torch.ones((b,)).long()
            time = time.to(self.device)
            timestepsnet = time.float()

            mask_time = mask_t * torch.ones((b,)).long()
            mask_time = mask_time.to(self.device)
            mask_timestepsnet = mask_time.float()
            pred_x0 = self.pred_fn(xt,mask_timestepsnet)
            pred_x0 = self.process_x0(pred_x0)
            pred_x0 = self.inver_fp(pred_x0,mask,y)


            if use_tv:
                with torch.set_grad_enabled(True):
                    for i in range(5):
                        pred_x0.requires_grad_()
                        loss = tv_loss(pred_x0)
                        loss.backward()
                        with torch.no_grad():
                            pred_x0.add_(pred_x0.grad, alpha=-lr/2)
                        pred_x0.grad.zero_()
                        pred_x0.requires_grad_(False)
                pred_x0 = self.inver_fp(pred_x0,mask,y)



            score = self.get_score_from_x0(pred_x0,xt,t)
            
            # xt - xt-1
            xt = self.reverse_sde_onestep_add(xt,pred_x0, score, t,prev_t,mask_t)

            #xt = self.inver_fp(xt,mask,y)
            # 保存中间状态 看一下
            if save_states: 
                interval = self.N // 14 #存 14 张 
                if t % interval == 0:
                    idx = t // interval
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir, exist_ok=True)
                    xt_img = torch.abs(xt[0,0]+xt[0,1]*1j).cpu().data.numpy()
                    plt.imsave( f'{save_dir}/sde_plus_xt_{t}.png',xt_img, cmap="gray")
                    x0_img = torch.abs(pred_x0[0,0]+xt[0,1]*1j).cpu().data.numpy()
                    plt.imsave(f'{save_dir}/sde_plus_x0_{t}.png', x0_img,cmap="gray")

        return xt




    # 反投影

    def reverse_sde_plus_plot(self, x,y,gt,mask,N=-1, save_states=False, save_dir='sde_state',use_tv=True,plus=True):
        lr  = 100
        N = self.N if N < 0 else N
        xt = x
        y = y
        b = xt.shape[0]
        t_list =[]
        psnr_list = []
        plus_psnr_list = []
        tv_plus_psnr_list = []
        # 开始重建
        for t in tqdm(reversed(range(1, N + 1))):
            t_list.append(t)
            time = t * torch.ones((b,)).long()
            time = time.to(self.device)
            timestepsnet = time.float()
            prev_t = t-1
            pred_x0 = self.pred_fn(xt,timestepsnet)
            pred_x0 = self.process_x0(pred_x0)
            psnr_list.append(cal_psnr(np.abs((pred_x0[0,0]+pred_x0[0,1]*1j).cpu().data.numpy()),gt))

            if plus:
                pred_x0 = self.inver_fp(pred_x0,mask,y)
                plus_psnr_list.append(cal_psnr(np.abs((pred_x0[0,0]+pred_x0[0,1]*1j).cpu().data.numpy()),gt))

            if use_tv:
                with torch.set_grad_enabled(True):
                    for i in range(5):
                        pred_x0.requires_grad_()
                        loss = tv_loss(pred_x0)
                        loss.backward()
                        with torch.no_grad():
                            pred_x0.add_(pred_x0.grad, alpha=-lr/2)
                        pred_x0.grad.zero_()
                        pred_x0.requires_grad_(False)
                pred_x0 = self.inver_fp(pred_x0,mask,y)
                tv_plus_psnr_list.append(cal_psnr(np.abs((pred_x0[0,0]+pred_x0[0,1]*1j).cpu().data.numpy()),gt))



            score = self.get_score_from_x0(pred_x0,xt,t)
            
            # xt - xt-1
            xt = self.reverse_sde_onestep(xt,pred_x0, score, t,prev_t,save_dir)

            #xt = self.inver_fp(xt,mask,y)
            # 保存中间状态 看一下
            if save_states: 
                interval = self.N // 14 #存 7 张 
                if t % interval == 1:
                    idx = t // interval
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir, exist_ok=True)
                    xt_img = torch.abs(xt[0,0]+xt[0,1]*1j).cpu().data.numpy()
                    
                    x0_img = torch.abs(pred_x0[0,0]+xt[0,1]*1j).cpu().data.numpy()
                    if plus==False:
                        plt.imsave( f'{save_dir}/base_xt_{t}.png',xt_img, cmap="gray")
                        plt.imsave(f'{save_dir}/base_x0_{t}.png', x0_img,cmap="gray")
                    elif use_tv:
                        plt.imsave( f'{save_dir}/tvplus_xt_{t}.png',xt_img, cmap="gray")
                        plt.imsave(f'{save_dir}/tvplus_x0_{t}.png', x0_img,cmap="gray")
                    else :
                        plt.imsave( f'{save_dir}/plus_xt_{t}.png',xt_img, cmap="gray")
                        plt.imsave(f'{save_dir}/plus_x0_{t}.png', x0_img,cmap="gray")
                    


        # hutu
         
        #fig = plt.figure()  
        
        # 绘制累计频率曲线  
        #plt.plot(t_list,psnr_list,'-k',linewidth = 1,label='none')  
        #plt.plot(t_list,plus_psnr_list,'-r',linewidth = 1,label="plus")  
        #plt.plot(t_list,tv_plus_psnr_list,'-b',linewidth = 1,label="plus+tv")  
        
        # 设置题目与坐标轴名称  
        #plt.title('evaluate pred_x0')  
        #plt.ylabel('psnr')  
        #plt.xlabel('Time(sec)') 
        #plt.legend()
        # 设置图例（置于右下方）  
        #plt.legend(loc='lower right')  
        
        # 显示图片  
        # plt.savefig('test.png')
        if plus==False:
            mylist  = psnr_list
        elif use_tv:
            mylist = tv_plus_psnr_list
        else :
            mylist = plus_psnr_list
        return xt,mylist



    def reverse_sde_plus_plot_conitune(self, x,y,gt,mask,N=-1, save_states=False, save_dir='sde_state',use_tv=True,plus=True):
        lr  = 100
        N = self.N if N < 0 else N
        xt = x
        y = y
        b = xt.shape[0]
        t_list =[]
        my_list = []

        psnr_list = []
        plus_psnr_list = []
        tv_plus_psnr_list = []

        mse_list = []
        plus_mse_list = []
        tv_plus_mse_list = []

        av_mse_list = []
        av_plus_mse_list = []
        av_tv_plus_mse_list = []

        gti = np.abs((gt[0,0]+gt[0,1]*1j).cpu().data.numpy())
        gtk = fastmri.fft2c(gt.permute(0,2,3,1))

        # 开始重建
        for t in tqdm(reversed(range(1, N + 1))):
            t_list.append(t)
            time = t * torch.ones((b,)).long()
            time = time.to(self.device)
            timestepsnet = time.float()
            prev_t = t-1
            pred_x0 = self.pred_fn(xt,timestepsnet)
            pred_x0 = self.process_x0(pred_x0)

            mask_c = self.mask[prev_t].cuda()
            pred_x0_k = fastmri.fft2c(pred_x0.permute(0,2,3,1))


            psnr_list.append(cal_psnr(np.abs((pred_x0[0,0]+pred_x0[0,1]*1j).cpu().data.numpy()),gti))
            mse_temp = (mask_c*(gtk-pred_x0_k)*(gtk-pred_x0_k)).cpu().data.numpy()
            mse_list.append(mse_temp.sum())
            av_mse_list.append(mse_temp.sum()/mask_c.sum().cpu().data.numpy())

            if plus:
                pred_x0 = self.inver_fp(pred_x0,mask,y)

                pred_x0_k = fastmri.fft2c(pred_x0.permute(0,2,3,1))
                mse_temp = (mask_c*(gtk-pred_x0_k)*(gtk-pred_x0_k)).cpu().data.numpy()
                plus_mse_list.append(mse_temp.sum())
                av_plus_mse_list.append(mse_temp.sum()/mask_c.sum().cpu().data.numpy())
                plus_psnr_list.append(cal_psnr(np.abs((pred_x0[0,0]+pred_x0[0,1]*1j).cpu().data.numpy()),gti))

            if use_tv:
                with torch.set_grad_enabled(True):
                    for i in range(5):
                        pred_x0.requires_grad_()
                        loss = tv_loss(pred_x0)
                        loss.backward()
                        with torch.no_grad():
                            pred_x0.add_(pred_x0.grad, alpha=-lr/2)
                        pred_x0.grad.zero_()
                        pred_x0.requires_grad_(False)
                pred_x0 = self.inver_fp(pred_x0,mask,y)
                pred_x0_k = fastmri.fft2c(pred_x0.permute(0,2,3,1))
                mse_temp = (mask_c*(gtk-pred_x0_k)*(gtk-pred_x0_k)).cpu().data.numpy()

                tv_plus_mse_list.append(mse_temp.sum())
                av_tv_plus_mse_list.append(mse_temp.sum()/mask_c.sum().cpu().data.numpy())
                tv_plus_psnr_list.append(cal_psnr(np.abs((pred_x0[0,0]+pred_x0[0,1]*1j).cpu().data.numpy()),gti))



            score = self.get_score_from_x0(pred_x0,xt,t)
            
            # xt - xt-1
            xt = self.reverse_sde_onestep(xt,pred_x0, score, t,prev_t)

            #xt = self.inver_fp(xt,mask,y)
            # 保存中间状态 看一下
            if save_states: 
                interval = self.N // 14 #存 7 张 
                if t % interval == 1:
                    idx = t // interval
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir, exist_ok=True)
                    xt_img = torch.abs(xt[0,0]+xt[0,1]*1j).cpu().data.numpy()
                    
                    x0_img = torch.abs(pred_x0[0,0]+xt[0,1]*1j).cpu().data.numpy()
                    if plus==False:
                        plt.imsave( f'{save_dir}/base_xt_{t}.png',xt_img, cmap="gray")
                        plt.imsave(f'{save_dir}/base_x0_{t}.png', x0_img,cmap="gray")
                    elif use_tv:
                        plt.imsave( f'{save_dir}/tvplus_xt_{t}.png',xt_img, cmap="gray")
                        plt.imsave(f'{save_dir}/tvplus_x0_{t}.png', x0_img,cmap="gray")
                    else :
                        plt.imsave( f'{save_dir}/plus_xt_{t}.png',xt_img, cmap="gray")
                        plt.imsave(f'{save_dir}/plus_x0_{t}.png', x0_img,cmap="gray")
                    


        # hutu
         
        #fig = plt.figure()  
        
        # 绘制累计频率曲线  
        #plt.plot(t_list,psnr_list,'-k',linewidth = 1,label='none')  
        #plt.plot(t_list,plus_psnr_list,'-r',linewidth = 1,label="plus")  
        #plt.plot(t_list,tv_plus_psnr_list,'-b',linewidth = 1,label="plus+tv")  
        
        # 设置题目与坐标轴名称  
        #plt.title('evaluate pred_x0')  
        #plt.ylabel('psnr')  
        #plt.xlabel('Time(sec)') 
        #plt.legend()
        # 设置图例（置于右下方）  
        #plt.legend(loc='lower right')  
        
        # 显示图片  
        # plt.savefig('test.png')
        if plus==False:
            my_list.append(psnr_list)
            my_list.append(mse_list)
            my_list.append(av_mse_list)
        elif use_tv:
            my_list.append(tv_plus_psnr_list)
            my_list.append(tv_plus_mse_list)
            my_list.append(av_tv_plus_mse_list)
        else :
            my_list.append(plus_psnr_list)
            my_list.append(plus_mse_list)
            my_list.append(av_plus_mse_list)
        return xt,my_list

    # 反投影
    def inver_fp(self,x0,mask,y):
        x_k = fastmri.fft2c(x0.permute(0,2,3,1))
        x_k_inv_mask = x_k *(torch.ones_like(mask)-mask)

        y_k = fastmri.fft2c(y.permute(0,2,3,1))
        y_k_mask = y_k * mask

        x0 = fastmri.ifft2c(x_k_inv_mask+y_k_mask)

        x0 = x0.permute(0,3,1,2)
        
        return x0


    # 整个逆向重建过程 base
    def reverse_re(self, xt, N=-1, save_states=False, save_dir='sde_state'):
        N = self.N if N < 0 else N
        b = xt.shape[0]
        # 开始重建
        for t in tqdm(reversed(range(1, N + 1))):
            time = t * torch.ones((b,)).long()
            time = time.to(self.device)
            timestepsnet = time.float()

            prev_t = t-1
            pred_x0 = self.pred_fn(xt,timestepsnet)
            pred_x0 = self.process_x0(pred_x0)

            # 得到t-1时刻的一些数据
            mask_t_prev = self.set_mask(prev_t)
            xt_prev_mean = self.get_xt_mean(pred_x0,mask_t=mask_t_prev)
            sigma_t_prev = self.get_sigma(prev_t)
            
            noise = torch.randn_like(xt)
            # xt_prev_noise = self.get_xt_noise(noise,mask_t_prev)

            xt = xt_prev_mean + sigma_t_prev * noise

            # 保存中间状态 看一下
            if save_states: 
                interval = self.N // 7 #存 7 张 
                if t % interval == 0:
                    idx = t // interval
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir, exist_ok=True)
                    xt_img = torch.abs(xt[0,0]+xt[0,1]*1j).cpu().data.numpy()
                    plt.imsave( f'{save_dir}/base_xt_{t}.png',xt_img, cmap="gray")
                    x0_img = torch.abs(pred_x0[0,0]+xt[0,1]*1j).cpu().data.numpy()
                    plt.imsave(f'{save_dir}/base_x0_{t}.png', x0_img,cmap="gray")
        return xt

      # 整个逆向重建过程 plus带有反向投影版本
    def reverse_re_plus(self, x,y,mask,N=-1, save_states=False, save_dir='sde_state'):
        N = self.N if N < 0 else N
        xt = x
        y = y
        b = xt.shape[0]
        for t in tqdm(reversed(range(1, N + 1))):
            time = t * torch.ones((b,)).long()
            time = time.to(self.device)
            timestepsnet = time.float()
            prev_t = t-1
            pred_x0 = self.pred_fn(xt,timestepsnet)
            pred_x0 = self.process_x0(pred_x0)
            # 修改x0
            pred_x0 = self.inver_fp(pred_x0,mask,y)

            # 得到t-1时刻的一些数据
            mask_t_prev = self.set_mask(prev_t)
            xt_prev_mean = self.get_xt_mean(pred_x0,mask_t=mask_t_prev)
            sigma_t_prev = self.get_sigma(prev_t)
            
            noise = torch.randn_like(xt)
            # xt_prev_noise = self.get_xt_noise(noise,mask_t_prev)

            xt = xt_prev_mean + sigma_t_prev * noise

            # 保存中间状态 看一下
            if save_states: 
                interval = self.N // 7 #存 7 张 
                if t % interval == 0:
                    idx = t // interval
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir, exist_ok=True)
                    xt_img = torch.abs(xt[0,0]+xt[0,1]*1j).cpu().data.numpy()
                    plt.imsave( f'{save_dir}/base_plus_xt_{t}.png',xt_img, cmap="gray")
                    x0_img = torch.abs(pred_x0[0,0]+xt[0,1]*1j).cpu().data.numpy()
                    plt.imsave(f'{save_dir}/base_plus_x0_{t}.png', x0_img,cmap="gray")
        return xt

    def reverse_sde_pc(self, xt, pc_step, pc_snr,N=-1,save_states=False, save_dir='sde_state'):
        N = self.N if N < 0 else N
        b = xt.shape[0]
        # 开始重建
        for t in tqdm(reversed(range(1, N + 1))):
            time = t * torch.ones((b,)).long()
            time = time.to(self.device)
            timestepsnet = time.float()

            prev_t = t-1

            prev_time = prev_t * torch.ones((b,)).long()
            timepre_net = prev_time.to(self.device)
            timepre_net = timepre_net.float()

            ## P采样
            pred_x0 = self.pred_fn(xt,timestepsnet)
            pred_x0 = self.process_x0(pred_x0)
            
            score = self.get_score_from_x0(pred_x0,xt,t)
            # xt - xt-1
            xt = self.reverse_sde_onestep(xt,pred_x0, score, t,prev_t)

            alpha = torch.ones_like(timestepsnet)
            ## C修正
            if prev_t>0:
                for i in range(pc_step):
                    pred_x0 = self.pred_fn(xt,timepre_net)
                    pred_x0 = self.process_x0(pred_x0)
                    grad = self.get_score_from_x0(pred_x0,xt,prev_t)
                    noise = torch.randn_like(xt)
                    grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
                    noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()

                    step_size = (pc_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
                    x_mean = xt + step_size[:, None, None, None] * grad
                    xt = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise

            # 保存中间状态 看一下
            if save_states: 
                interval = self.N // 7 #存 7 张 
                if t % interval == 0:
                    idx = t // interval
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir, exist_ok=True)
                    xt_img = torch.abs(xt[0,0]+xt[0,1]*1j).cpu().data.numpy()
                    plt.imsave( f'{save_dir}/pc_xt_{t}.png',xt_img, cmap="gray")
                    x0_img = torch.abs(pred_x0[0,0]+xt[0,1]*1j).cpu().data.numpy()
                    plt.imsave(f'{save_dir}/pc_x0_{t}.png', x0_img,cmap="gray")

        return xt

      # 整个逆向重建过程 plus带有反向投影版本
    def reverse_sde_pc_plus(self, x,pc_step,pc_snr,y,mask, N=-1, save_states=False, save_dir='sde_state'):
        N = self.N if N < 0 else N
        xt = x
        y = y
        b = xt.shape[0]
        # 开始重建
        for t in tqdm(reversed(range(1, N + 1))):
            time = t * torch.ones((b,)).long()
            time = time.to(self.device)
            timestepsnet = time.float()

            prev_t = t-1

            prev_time = prev_t * torch.ones((b,)).long()
            timepre_net = prev_time.to(self.device)
            timepre_net = timepre_net.float()

            ## P采样
            pred_x0 = self.pred_fn(xt,timestepsnet)
            pred_x0 = self.process_x0(pred_x0)
            pred_x0 = self.inver_fp(pred_x0,mask,y)

            score = self.get_score_from_x0(pred_x0,xt,t)
            # xt - xt-1
            xt = self.reverse_sde_onestep(xt,pred_x0, score, t,prev_t)

            alpha = torch.ones_like(timestepsnet)
            ## C修正
            if prev_t>0:
                for i in range(pc_step):
                    pred_x0 = self.pred_fn(xt,timepre_net)
                    pred_x0 = self.process_x0(pred_x0)
                    pred_x0 = self.inver_fp(pred_x0,mask,y)
                    grad = self.get_score_from_x0(pred_x0,xt,prev_t)
                    noise = torch.randn_like(xt)
                    grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
                    noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()

                    step_size = (pc_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
                    x_mean = xt + step_size[:, None, None, None] * grad
                    xt = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise

            # 保存中间状态 看一下
            if save_states: 
                interval = self.N // 7 #存 7 张 
                if t % interval == 0:
                    idx = t // interval
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir, exist_ok=True)
                    xt_img = torch.abs(xt[0,0]+xt[0,1]*1j).cpu().data.numpy()
                    plt.imsave( f'{save_dir}/pc_plus_xt_{t}.png',xt_img, cmap="gray")
                    x0_img = torch.abs(pred_x0[0,0]+xt[0,1]*1j).cpu().data.numpy()
                    plt.imsave(f'{save_dir}/pc_plus_x0_{t}.png', x0_img,cmap="gray")

        return xt

    
    # 整个逆向重建过程 base
    def reverse_sde1(self, xt, N=-1, save_states=False, save_dir='sde_state'):
        N = self.N if N < 0 else N
        x = xt.clone()
        b = xt.shape[0]
        # 开始重建
        for t in tqdm(reversed(range(1, N + 1))):
            time = t * torch.ones((b,)).long()
            time = time.to(self.device)
            timestepsnet = time.float()
            prev_t = t-1


            pred_x0 = self.pred_fn(xt,timestepsnet)
            pred_x0 = self.process_x0(pred_x0)

            score = self.get_score_from_x0(pred_x0,xt,t)
            # xt - xt-1
            xt = self.reverse_sde_onestep(xt,pred_x0, score, t,prev_t)

            # 保存中间状态 看一下
            if save_states: 
                interval = self.N // 7 #存 7 张 
                if t % interval == 0:
                    idx = t // interval
                    os.makedirs(save_dir, exist_ok=True)
                    tvul.imsave(xt.data, f'{save_dir}/state_{idx}.png', normalize=False)

        return xt

      # 整个逆向重建过程 plus带有反向投影版本
    def reverse_sde_plus1(self, x,y,mask,N=-1, save_states=False, save_dir='sde_state'):
        N = self.N if N < 0 else N
        xt = x
        y = y
        b = xt.shape[0]
        # 开始重建
        for t in tqdm(reversed(range(1, N + 1))):
            time = t * torch.ones((b,)).long()
            time = time.to(self.device)
            timestepsnet = time.float()
            prev_t = t-1
            pred_x0 = self.pred_fn(xt,timestepsnet)
            pred_x0 = self.process_x0(pred_x0)

            pred_x0 = self.inver_fp(pred_x0,mask,y)

            score = self.get_score_from_x0(pred_x0,xt,t)
            # xt - xt-1
            xt = self.reverse_sde_onestep(xt,pred_x0, score, t,prev_t)

            # 保存中间状态 看一下
            if save_states: 
                interval = self.N // 7 #存 7 张 
                if t % interval == 0:
                    idx = t // interval
                    os.makedirs(save_dir, exist_ok=True)
                    tvul.imsave(xt.data, f'{save_dir}/state_{idx}.png', normalize=False)

        return xt


    # 反投影

    # 单步重建
    def reverse_sde_onestep(self,xt,pred_x0,score,t,t_prev,save_dir=None):
        mask_t = self.mask[t]
        mask_t = mask_t.to(self.device)
        mask_t_prev = self.mask[t_prev]
        mask_t_prev = mask_t_prev.to(self.device)

        sigma_t = self.get_sigma(t)
        sigma_t_prev = self.get_sigma(t_prev)

        # prev_s = max(t-5,0)
        # prev_b = max(t-10,0) 
        # mask_prev_s = self.mask[prev_s].to(self.device)
        # mask_prev_b = self.mask[prev_b].to(self.device)

        # sa = self.get_xt_mean(pred_x0,mask_prev_s)
        # ba = self.get_xt_mean(pred_x0,mask_prev_b)
        # b = self.get_xt_mean(pred_x0,mask_t)
        # sa_img = torch.abs(sa[0,0]+sa[0,1]*1j)
        # ba_img = torch.abs(ba[0,0]+ba[0,1]*1j)
        # b_img = torch.abs(b[0,0]+b[0,1]*1j)
        # plt.imsave(f'{save_dir}/s_increment_{t}.png',(sa_img-b_img).cpu().data.numpy(),cmap='gray')
        # torchvision.utils.save_image((sa_img-b_img),f'{save_dir}/s_torch_increment_{t}.png')

        # plt.imsave(f'{save_dir}/b_increment_{t}.png',(ba_img-b_img).cpu().data.numpy(),cmap='gray')
        # torchvision.utils.save_image((ba_img-b_img),f'{save_dir}/b_torch_increment_{t}.png')

        increment = self.add_increment(pred_x0,mask_t,mask_t_prev,t,save_dir)

        
        



        
        denoise = self.sde_denoise(score,mask_t,mask_t_prev,sigma_t,sigma_t_prev)

        noise = torch.randn_like(xt)
        addnoise = self.sde_add_noise(noise,mask_t_prev,sigma_t,sigma_t_prev)
        #addnoise = self.sde_add_noise1(noise,mask_t,sigma_t)

        #in_img = torch.abs(increment[0,0]+increment[0,1]*1j).cpu().data.numpy()
        #plt.imsave( f'{save_dir}/increment_{t}.png',in_img, cmap="gray")

        # denoise_img = torch.abs(denoise[0,0]+denoise[0,1]*1j).cpu().data.numpy()
        # plt.imsave( f'{save_dir}/denoise_{t}.png',denoise_img, cmap="gray")
        # torchvision.utils.save_image(torch.abs(denoise[0,0]+denoise[0,1]*1j),f'{save_dir}/torch_denoise_{t}.png')

        # addnoise_img = torch.abs(addnoise[0,0]+addnoise[0,1]*1j).cpu().data.numpy()
        # plt.imsave( f'{save_dir}/addnoise_{t}.png',addnoise_img, cmap="gray")
        # torchvision.utils.save_image(torch.abs(addnoise[0,0]+addnoise[0,1]*1j),f'{save_dir}/torch_addnoise_{t}.png')

      
        x_prev  =xt + increment - denoise + addnoise

        return x_prev 
    
    def reverse_sde_onestep_add(self,xt,pred_x0,score,t,t_prev,maskt,save_dir=None):
        mask_t = self.mask[maskt]
        mask_t = mask_t.to(self.device)
        mask_t_prev = self.mask[maskt-1]
        mask_t_prev = mask_t_prev.to(self.device)

        sigma_t = self.get_sigma(t)
        sigma_t_prev = self.get_sigma(t_prev)

        

        increment = self.add_increment(pred_x0,mask_t,mask_t_prev,t,save_dir)


        denoise = self.sde_denoise(score,mask_t,mask_t_prev,sigma_t,sigma_t_prev)

        noise = torch.randn_like(xt)
        addnoise = self.sde_add_noise(noise,mask_t_prev,sigma_t,sigma_t_prev)
        
      
        x_prev  =xt + increment - denoise + addnoise

        return x_prev 
    
    # 重建过程的增量
    def add_increment(self,x0,mask_t,mask_t_prev,t,save_dir = None):
        
        increment = self.get_xt_mean(x0,mask_t_prev) - self.get_xt_mean(x0,mask_t)

        return increment

    # 重建过程中的分数去噪
    def sde_denoise(self,score,mask_t,mask_t_prev,sigma_t,sigma_t_prev):

        #s_t  = self.get_xt_noise(score,mask_t)
        #s_t_prev = self.get_xt_noise(score,mask_t_prev)
        
        #denoise = sigma_t_prev * sigma_t_prev * s_t_prev - sigma_t * sigma_t * s_t
        
        denoise1 = (sigma_t_prev **2 - sigma_t**2) * score

        return denoise1

    # 重建过程中的郎之万加噪 TODO 这个地方可能需要在研究一下
    def sde_add_noise(self,noise,mask_t_prev,sigma_t,sigma_t_prev):
        # noise = self.get_xt_noise(noise,mask_t_prev)
        scale = (sigma_t**2 - sigma_t_prev**2).sqrt()
        snoise = scale * noise
        return snoise

    

    # 前向过程
    def forward(self, x0, N=-1, save_dir='forward_state'):
        N = self.N if N < 0 else N
        x = x0.clone()
        for t in tqdm(range(0, N + 1)):
            xt,xt_k ,xt_samll = self.get_xt1(x, t)
            os.makedirs(save_dir, exist_ok=True)
            xt_k1 = torch.abs(xt_k[:,0,:,:]+xt_k[:,1,:,:]*1j)
            tvul.save_image(xt_k1.data, f'{save_dir}/k_{t}.png', normalize=False)
            #import imageio
            xt = xt.cpu().data.numpy()
            xt_k = xt_k.cpu().data.numpy()
            xt_samll = xt_samll.cpu().data.numpy()

            out = np.abs(xt[0,0]+xt[0,1]*1j)
            out_k = np.abs(xt_k[0,0]+xt_k[0,1]*1j)
            out_small = np.abs(xt_samll[0,0]+xt_samll[0,1]*1j)

            plt.imsave(f'{save_dir}/x_{t}.png',out, cmap="gray")
            plt.imsave(f'{save_dir}/x_small_{t}.png',out_small, cmap="gray")
            #plt.imsave(f'{save_dir}/xk_{t}.png',out_k, cmap="gray")
            #imageio.imsave(f'{save_dir}/x_{t}.png',out/out.max())
            #imageio.imsave(f'{save_dir}/k_{t}.png',out_k)
            
        return x

    # 得到时间t处的退化图像，方便在测试时候使用
    def get_deg_t(self,x0,t=None):
        x0 = x0.to(self.device)
        b = x0.shape[0]
        # 这里默认在训练时候的验证只验证8倍超分的情况
        if t is None :
            t =(self.N * torch.ones((b,1,1,1))).long()
        else:
            t = (t * torch.ones((b,1,1,1))).long()
        mask_t = self.mask[t]
        mask_t = mask_t.to(self.device)
        # sigma_t 
        sigma_t = self.get_sigma(min(t,140))

        # At(x0)
        x_k = fastmri.fft2c(x0.permute(0,2,3,1))

        ## 环城膝关节后需要fftshit一下
        # x_k= fastmri.fftshift(x_k,dim=[-3, -2])


        xt_kmean = x_k * mask_t


        xt_mean = fastmri.ifft2c(xt_kmean)


        xt_noise = torch.randn_like(xt_mean)
        #xt_noise = self.get_xt_noise(noise,mask_t)

        xt = xt_mean + sigma_t * xt_noise
        
        xt = xt.permute(0,3,1,2)

        return xt,mask_t,xt_mean.permute(0,3,1,2)
    

    # 随机生成退图像以及时间t
    def generate_random_degenerate(self,x0):
        '''
        训练阶段
        该函数功能为,随机生成一个时间t,然后得到对应的时间t以及退化后的图像xt
        '''
        x0 = x0.to(self.device)

        batch_size = x0.shape[0]

        # 随机选取时间 训练时要用
        timesteps = torch.randint(1,self.N +1,(batch_size,)).long()

        ######################################################
        # (这里可以看code说明为什么有两段式生成或者一段式生成)
        ######################################################


        ######################################################
        # 一段式生成，或者说直接生成（默认）
        ######################################################

        # xt = self.get_xt(x0,timesteps)


        ######################################################
        # 两段式生成
        ######################################################

        mask_t = self.set_mask(timesteps)
        sigma_t = self.get_sigma(timesteps)

        xt_mean = self.get_xt_mean(x0,mask_t)
 
        xt_noise = torch.randn_like(x0)
        #xt_noise = self.get_xt_noise(noise,mask_t)

        xt = xt_mean + sigma_t * xt_noise

        timesteps = timesteps
        return timesteps,xt,xt_mean
    # 随机生成退图像以及时间t 并固定八倍炒粉
    def generate_random_degenerate_gu(self,x0):
        '''
        训练阶段
        该函数功能为,随机生成一个时间t,然后得到对应的时间t以及退化后的图像xt
        '''
        x0 = x0.to(self.device)

        batch_size = x0.shape[0]

        # 随机选取时间 训练时要用
        timesteps = torch.randint(1,self.N +1,(batch_size,)).long()

        ######################################################
        # (这里可以看code说明为什么有两段式生成或者一段式生成)
        ######################################################


        ######################################################
        # 一段式生成，或者说直接生成（默认）
        ######################################################

        # xt = self.get_xt(x0,timesteps)


        ######################################################
        # 两段式生成
        ######################################################
        mask_t_gu = self.set_mask(140)
        xt_mean_gu = self.get_xt_mean(x0,mask_t_gu)
        mask_t = self.set_mask(timesteps)
        sigma_t = self.get_sigma(timesteps)

        xt_mean = self.get_xt_mean(x0,mask_t)
 
        xt_noise = torch.randn_like(x0)
        #xt_noise = self.get_xt_noise(noise,mask_t)

        xt = xt_mean + sigma_t * xt_noise

        timesteps = timesteps
        return timesteps,xt,xt_mean_gu
    def process_x0(self,x0):
        return x0.clamp(-1, 1)



def tv_loss(x):
    batch_size, c, h, w = x.size()
    tv_h = torch.abs(x[:,:,1:,:] - x[:,:,:-1,:]).sum()
    tv_w = torch.abs(x[:,:,:,1:] - x[:,:,:,:-1]).sum()
    return (tv_h + tv_w) / (batch_size * c * h * w)

