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


class SRDPM:
    def __init__(self,N,max_sigma,noise_schedule='consine',eps=0.01,device=None):
      self.N = N
      self.dt = 1 / N
      self.device = device
      self.max_sigma = max_sigma
      

      self._initialize(self.max_sigma,N,noise_schedule,eps)

    def _initialize(self, max_sigma, N, schedule, eps=0.01):
        
        ###############################
        # mask list
        ###############################


        def gen_sequence(start, end, step):
            ############################################
            #### generate_Arithmetic progression  ####
            ############################################
            sequence = torch.arange(start, end-step, -step )
            return sequence


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


        def gen_mask(image_shape):
            ############################################
            ####    generate_all_center_mask    ####
            ############################################
            mask =  []
            total_scale = gen_sequence(320, 40, 2)

            for i in range(len(total_scale)):
                m = get_center_mask(image_shape, total_scale[i])
                m = torch.from_numpy(m)
                mm = torch.stack((m,m), dim = 0).view(1,2,320,320)
                mm = mm.permute(0,2,3,1)
                mm = mm.float()
                mask.append(mm)
            return mask

        def gaussian_kernel(size, sigma):
            x = torch.arange(start = -size//2+1, end  = size//2+1, step = 1, dtype = torch.float32)
            y = x.view(-1, 1)
            x0, y0 = 0,0
            kernel = torch.exp(-0.5 * ((x-x0)**2+(y-y0)**2/sigma **2))
            kernel /= kernel.sum()
            return kernel





        def constant_theta_schedule(timesteps, v=1.):
            """
            constant schedule
            """
            print('constant schedule')
            timesteps = timesteps + 1 # T from 1 to 100
            return torch.ones(timesteps, dtype=torch.float32)


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


        def get_thetas_cumsum(thetas):
            return torch.cumsum(thetas, dim=0)


        def get_sigmas(thetas):
            return torch.sqrt(max_sigma**2 * 2 * thetas)


        def get_sigma_bars(thetas_cumsum):
            return torch.sqrt(max_sigma**2 * (1 - torch.exp(-2 * thetas_cumsum * self.dt)))
        

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

        x_k = fastmri.fft2c(x0.permute(0,2,3,1))

        xt_kmean = x_k * mask_t

        xt_mean = fastmri.ifft2c(xt_kmean)

        xt_mean = xt_mean.permute(0,3,1,2)
        return xt_mean

    def get_xs_s_mean(self,x0,mask_t):

        x_k = fastmri.fft2c(x0.permute(0,2,3,1))

        xt_kmean = x_k * mask_t
        w = int(torch.sqrt(mask_t.sum()/2))
        w = int((320-w)/2)
        xt_kmean = xt_kmean[:,w:-w,w:-w,:]

        xt_mean = fastmri.ifft2c(xt_kmean)

        xt_mean = xt_mean.permute(0,3,1,2)
        return xt_mean

    def get_xt_mean_k(self,x0,mask_t):

        x_k = fastmri.fft2c(x0.permute(0,2,3,1))

        xt_kmean = x_k * mask_t

        xt_kmean = xt_kmean.permute(0,3,1,2)
        return xt_kmean

    def set_mask(self,t):
        mask_t = self.mask[t]
        mask_t = mask_t.to(self.device)
        return mask_t


    def get_xt_noise(self,noise,mask_t):

        n_k = fastmri.fft2c(noise.permute(0,2,3,1))

        nt_k = (torch.ones_like(mask_t) -mask_t)*n_k 

        noise_t = fastmri.ifft2c(nt_k)

        noise_t = noise_t.permute(0,3,1,2)
        return noise_t


    def get_xt_noisei(self,noise,mask_t):
        noise_t = torch.randn_like(noise)
        return noise_t

    

    def get_xt(self,x0,t):


        # mask
        mask_t = self.mask[t]
        mask_t = mask_t.to(self.device)

        # At(x0)
        x_k = fastmri.fft2c(x0.permute(0,2,3,1))
        xt_kmean = x_k * mask_t


        # (I-At)z
        noise = torch.randn_like(x0)
        n_k = fastmri.fft2c(noise.permute(0,2,3,1))
        nt_k = (torch.ones_like(mask_t) -mask_t)*n_k 



        # sigma_t 
        sigma_t = self.get_sigma(t)


        xt_k = xt_kmean + sigma_t * nt_k
        xt = fastmri.ifft2c(xt_k)
        
        xt = xt.permute(0,3,1,2)

        return xt

    def get_xt1(self,x0,t):


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
    
    def set_model(self,model):
        self.model = model


    def pred_fn(self, xt, t,res):

        return self.model(xt, t,res)

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

        for t in tqdm(reversed(range(1, N + 1))):
            time = t * torch.ones((b,)).long()
            time = time.to(self.device)
            timestepsnet = time.float()
            prev_t = t-1
            pred_x0 = self.pred_fn(xt,timestepsnet)
            pred_x0 = self.process_x0(pred_x0)

            mask_t = self.set_mask(t)

            mask_t_prev = self.set_mask(prev_t)
            xt_prev_mean = self.get_xt_mean(pred_x0,mask_t=mask_t_prev)
            sigma_t_prev = self.get_sigma(prev_t)
            
            xt_mean = self.get_xt_mean(pred_x0,mask_t=mask_t)
            xt_mean1 = self.get_xt_mean(xt,mask_t=mask_t)

            noise = torch.randn_like(xt)



            xt =xt_mean1 - xt_mean + xt_prev_mean + sigma_t_prev * noise


            if save_states: 
                if not os.path.exists(save_dir):
                        os.makedirs(save_dir, exist_ok=True)
                interval = self.N // 7
                if t % interval == 0:
                    idx = t // interval
                    
                    xt_img = torch.abs(xt[0,0]+xt[0,1]*1j).cpu().data.numpy()
                    plt.imsave( f'{save_dir}/mix_xt_{t}.png',xt_img, cmap="gray")
                    x0_img = torch.abs(pred_x0[0,0]+xt[0,1]*1j).cpu().data.numpy()
                    plt.imsave(f'{save_dir}/mix_x0_{t}.png', x0_img,cmap="gray")

        return xt



    def reverse_sde(self, xt, N=-1, save_states=False, save_dir='sde_state'):
        N = self.N if N < 0 else N
        b = xt.shape[0]

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


            if save_states: 
                if not os.path.exists(save_dir):
                        os.makedirs(save_dir, exist_ok=True)
                interval = self.N // 7 #
                if t % interval == 0:
                    idx = t // interval
                    
                    xt_img = torch.abs(xt[0,0]+xt[0,1]*1j).cpu().data.numpy()
                    plt.imsave( f'{save_dir}/sde_xt_{t}.png',xt_img, cmap="gray")
                    x0_img = torch.abs(pred_x0[0,0]+xt[0,1]*1j).cpu().data.numpy()
                    plt.imsave(f'{save_dir}/sde_x0_{t}.png', x0_img,cmap="gray")

        return xt


    # @torch.no_grad()
    def reverse_sde_plus(self, x,y,mask,N=-1, save_states=False, save_dir='sde_state',use_tv=True):
        lr  = 100
        N = self.N if N < 0 else N
        xt = x
        y = y
        b = xt.shape[0]

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
                    for i in range(1):
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


            if save_states: 
                interval = self.N // 14
                if t % interval == 0:
                    idx = t // interval
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir, exist_ok=True)
                    xt_img = torch.abs(xt[0,0]+xt[0,1]*1j).cpu().data.numpy()
                    plt.imsave( f'{save_dir}/sde_plus_xt_{t}.png',xt_img, cmap="gray")
                    x0_img = torch.abs(pred_x0[0,0]+xt[0,1]*1j).cpu().data.numpy()
                    plt.imsave(f'{save_dir}/sde_plus_x0_{t}.png', x0_img,cmap="gray")

        return xt
    
    def reverse_sde_plus_add(self, x,y,condition,mask,N=-1, save_states=False, save_dir='sde_state',use_inver_fp=False,use_tv=False):
        lr  = 100
        N = self.N if N < 0 else N
        xt = x
        y = y
        b = xt.shape[0]

        condition = condition.float()
        for t in tqdm(reversed(range(1, N + 1))):
            mask_t = t
            t = min(t, 140)
            time = t * torch.ones((b,)).long()
            time = time.to(self.device)
            timestepsnet = time.float()
            prev_t = t - 1
            res_in = (320 - 2 * t, 320 - 2 * t)
            res_out = (320, 320)
            res = self.get_res_s(res_in,res_out)
            res = res.to(self.device)
            mask_time = mask_t * torch.ones((b,)).long()
            mask_time = mask_time.to(self.device)
            mask_timestepsnet = mask_time.float()
            netinput = torch.cat([xt.float()],dim=1)
            pred_x0 = self.pred_fn(netinput,mask_timestepsnet,res)
            pred_x0 = self.process_x0(pred_x0)
            if use_inver_fp:
                pred_x0 = self.inver_fp(pred_x0,mask,y)


            if use_tv:
                with torch.set_grad_enabled(True):
                    for i in range(1):
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


            if save_states: 
                interval = self.N // 14
                if t % interval == 0:
                    idx = t // interval
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir, exist_ok=True)
                    xt_img = torch.abs(xt[0,0]+xt[0,1]*1j).cpu().data.numpy()
                    plt.imsave( f'{save_dir}/sde_plus_xt_{t}.png',xt_img, cmap="gray")
                    x0_img = torch.abs(pred_x0[0,0]+xt[0,1]*1j).cpu().data.numpy()
                    plt.imsave(f'{save_dir}/sde_plus_x0_{t}.png', x0_img,cmap="gray")

        return xt


    def reverse_sde_plus_add_t_s(self, x,y,condition,mask,N=-1, save_states=False, save_dir='sde_state',use_inver_fp=False,use_tv=True,time_t=None,time_s=None):
        lr  = 100
        N = time_t
        xt = x
        y = y
        b = xt.shape[0]

        condition = condition.float()
        for t in tqdm(reversed(range(time_s+1, N + 1))):
            if t > 200:
                time_s = int((320 - 80) / 2)
                mask_t = t
                t = min(t, 140)
                time = t * torch.ones((b,)).long()
                time = time.to(self.device)
                timestepsnet = time.float()
                prev_t = t - 1
                res_in = (320 - 2 * t, 320 - 2 * t)
                res_out = (320 - 2 * time_s, 320 - 2 * time_s)
                res = self.get_res_s(res_in, res_out)
                res = res.to(self.device)
                mask_time = mask_t * torch.ones((b,)).long()
                mask_time = mask_time.to(self.device)
                mask_timestepsnet = mask_time.float()
                netinput = xt.float()  # torch.cat([xt.float(),condition],dim=1)
                pred_x0 = self.pred_fn(netinput, mask_timestepsnet, res)
                pred_x0 = self.process_x0(pred_x0)
                if use_inver_fp:
                    pred_x0 = self.inver_fp(pred_x0, mask, y)

                if use_tv:
                    with torch.set_grad_enabled(True):
                        for i in range(1):
                            pred_x0.requires_grad_()
                            loss = tv_loss(pred_x0)
                            loss.backward()
                            with torch.no_grad():
                                pred_x0.add_(pred_x0.grad, alpha=-lr / 2)
                            pred_x0.grad.zero_()
                            pred_x0.requires_grad_(False)
                    pred_x0 = self.inver_fp(pred_x0, mask, y)
            else:
                mask_t = t
                t = min(t, 140)
                time = t * torch.ones((b,)).long()
                time = time.to(self.device)
                timestepsnet = time.float()
                prev_t = t - 1
                res_in = (320 - 2 * t, 320 - 2 * t)
                res_out = (320 - 2 * time_s, 320 - 2 * time_s)
                res = self.get_res_s(res_in, res_out)
                res = res.to(self.device)
                mask_time = mask_t * torch.ones((b,)).long()
                mask_time = mask_time.to(self.device)
                mask_timestepsnet = mask_time.float()
                netinput = xt.float()  # torch.cat([xt.float(),condition],dim=1)
                pred_x0 = self.pred_fn(netinput, mask_timestepsnet, res)
                pred_x0 = self.process_x0(pred_x0)
                if use_inver_fp:
                    pred_x0 = self.inver_fp(pred_x0, mask, y)

                if use_tv:
                    with torch.set_grad_enabled(True):
                        for i in range(1):
                            pred_x0.requires_grad_()
                            loss = tv_loss(pred_x0)
                            loss.backward()
                            with torch.no_grad():
                                pred_x0.add_(pred_x0.grad, alpha=-lr / 2)
                            pred_x0.grad.zero_()
                            pred_x0.requires_grad_(False)
                    pred_x0 = self.inver_fp(pred_x0, mask, y)


            score = self.get_score_from_x0(pred_x0,xt,t)
            
            # xt - xt-1
            xt = self.reverse_sde_onestep_add(xt,pred_x0, score, t,prev_t,mask_t)


            if save_states: 
                interval = self.N // 14
                if t % interval == 0:
                    idx = t // interval
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir, exist_ok=True)
                    xt_img = torch.abs(xt[0,0]+xt[0,1]*1j).cpu().data.numpy()
                    plt.imsave( f'{save_dir}/sde_plus_xt_{t}.png',xt_img, cmap="gray")
                    x0_img = torch.abs(pred_x0[0,0]+xt[0,1]*1j).cpu().data.numpy()
                    plt.imsave(f'{save_dir}/sde_plus_x0_{t}.png', x0_img,cmap="gray")

        return xt,pred_x0




    def inver_fp(self,x0,mask,y):
        x_k = fastmri.fft2c(x0.permute(0,2,3,1))
        x_k_inv_mask = x_k *(torch.ones_like(mask)-mask)

        y_k = fastmri.fft2c(y.permute(0,2,3,1))
        y_k_mask = y_k * mask

        x0 = fastmri.ifft2c(x_k_inv_mask+y_k_mask)

        x0 = x0.permute(0,3,1,2)
        
        return x0


    def reverse_re(self, xt, N=-1, save_states=False, save_dir='sde_state'):
        N = self.N if N < 0 else N
        b = xt.shape[0]

        for t in tqdm(reversed(range(1, N + 1))):
            time = t * torch.ones((b,)).long()
            time = time.to(self.device)
            timestepsnet = time.float()

            prev_t = t-1
            pred_x0 = self.pred_fn(xt,timestepsnet)
            pred_x0 = self.process_x0(pred_x0)

            mask_t_prev = self.set_mask(prev_t)
            xt_prev_mean = self.get_xt_mean(pred_x0,mask_t=mask_t_prev)
            sigma_t_prev = self.get_sigma(prev_t)
            
            noise = torch.randn_like(xt)


            xt = xt_prev_mean + sigma_t_prev * noise


            if save_states: 
                interval = self.N // 7
                if t % interval == 0:
                    idx = t // interval
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir, exist_ok=True)
                    xt_img = torch.abs(xt[0,0]+xt[0,1]*1j).cpu().data.numpy()
                    plt.imsave( f'{save_dir}/base_xt_{t}.png',xt_img, cmap="gray")
                    x0_img = torch.abs(pred_x0[0,0]+xt[0,1]*1j).cpu().data.numpy()
                    plt.imsave(f'{save_dir}/base_x0_{t}.png', x0_img,cmap="gray")
        return xt


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

            pred_x0 = self.inver_fp(pred_x0,mask,y)


            mask_t_prev = self.set_mask(prev_t)
            xt_prev_mean = self.get_xt_mean(pred_x0,mask_t=mask_t_prev)
            sigma_t_prev = self.get_sigma(prev_t)
            
            noise = torch.randn_like(xt)


            xt = xt_prev_mean + sigma_t_prev * noise

            if save_states: 
                interval = self.N // 7
                if t % interval == 0:
                    idx = t // interval
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir, exist_ok=True)
                    xt_img = torch.abs(xt[0,0]+xt[0,1]*1j).cpu().data.numpy()
                    plt.imsave( f'{save_dir}/base_plus_xt_{t}.png',xt_img, cmap="gray")
                    x0_img = torch.abs(pred_x0[0,0]+xt[0,1]*1j).cpu().data.numpy()
                    plt.imsave(f'{save_dir}/base_plus_x0_{t}.png', x0_img,cmap="gray")
        return xt




    def reverse_sde_onestep(self,xt,pred_x0,score,t,t_prev,save_dir=None):
        mask_t = self.mask[t]
        mask_t = mask_t.to(self.device)
        mask_t_prev = self.mask[t_prev]
        mask_t_prev = mask_t_prev.to(self.device)

        sigma_t = self.get_sigma(t)
        sigma_t_prev = self.get_sigma(t_prev)

        increment = self.add_increment(pred_x0,mask_t,mask_t_prev,t,save_dir)
        denoise = self.sde_denoise(score,mask_t,mask_t_prev,sigma_t,sigma_t_prev)

        noise = torch.randn_like(xt)
        addnoise = self.sde_add_noise(noise,mask_t_prev,sigma_t,sigma_t_prev)

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
    

    def add_increment(self,x0,mask_t,mask_t_prev,t,save_dir = None):
        
        increment = self.get_xt_mean(x0,mask_t_prev) - self.get_xt_mean(x0,mask_t)

        return increment


    def sde_denoise(self,score,mask_t,mask_t_prev,sigma_t,sigma_t_prev):
        
        denoise1 = (sigma_t_prev **2 - sigma_t**2) * score

        return denoise1


    def sde_add_noise(self,noise,mask_t_prev,sigma_t,sigma_t_prev):

        scale = (sigma_t**2 - sigma_t_prev**2).sqrt()
        snoise = scale * noise
        return snoise

    


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

            
        return x


    def get_deg_t(self,x0,t=None):
        x0 = x0.to(self.device)
        b = x0.shape[0]
        if t is None :
            t =(self.N * torch.ones((b,1,1,1))).long()
        else:
            t = (t * torch.ones((b,1,1,1))).long()
        t_condition = (self.N * torch.ones((b,1,1,1))).long()
        mask_condition = self.mask[t_condition]
        mask_condition = mask_condition.to(self.device)
        mask_t = self.mask[t]
        mask_t = mask_t.to(self.device)
        # sigma_t 
        sigma_t = self.get_sigma(min(t,140))

        # At(x0)
        x_k = fastmri.fft2c(x0.permute(0,2,3,1))


        xt_kmean = x_k * mask_t
        xt_condition_kmean = x_k * mask_condition

        xt_mean = fastmri.ifft2c(xt_kmean)
        xt_condition = fastmri.ifft2c(xt_condition_kmean)

        xt_noise = torch.randn_like(xt_mean)

        xt = xt_mean + sigma_t * xt_noise
        
        xt = xt.permute(0,3,1,2)

        return xt, mask_t, xt_mean.permute(0,3,1,2),xt_condition.permute(0,3,1,2)
    
    def get_deg_t_s(self,x0, t=None, s=None):
        x0 = x0.to(self.device)
        b = x0.shape[0]

        if t is None :
            t =(self.N * torch.ones((b,1,1,1))).long()
        else:
            t = (t * torch.ones((b,1,1,1))).long()
        t_condition = t
        mask_condition = self.mask[t_condition]
        mask_condition = mask_condition.to(self.device)
        mask_t = self.mask[t]
        mask_t = mask_t.to(self.device)
        # sigma_t 
        sigma_t = self.get_sigma(min(t,140))


        # At(x0)
        x_k = fastmri.fft2c(x0.permute(0,2,3,1))


        xt_kmean = x_k * mask_t

        xt_condition_kmean = x_k * mask_condition

        xt_mean = fastmri.ifft2c(xt_kmean)
        xt_condition = fastmri.ifft2c(xt_condition_kmean)

        xt_noise = torch.randn_like(xt_mean)

        xt = xt_mean + sigma_t * xt_noise

        
        xt = xt.permute(0,3,1,2)

        time_s =   (s * torch.ones((b,1,1,1))).long()
        mask_s = self.set_mask(time_s)
        xs = self.get_xt_mean(x0,mask_s)

        return xt, mask_t, xt_mean.permute(0,3,1,2),xt_condition.permute(0,3,1,2), xs, mask_s

    def get_deg_t_s_two(self, x0, sr, t=None, s=None):
        sr = sr.to(self.device)
        x0 = x0.to(self.device)
        b = sr.shape[0]

        if t is None:
            t = (self.N * torch.ones((b, 1, 1, 1))).long()
        else:
            t = (t * torch.ones((b, 1, 1, 1))).long()
        t_condition = t
        mask_condition = self.mask[t_condition]
        mask_condition = mask_condition.to(self.device)
        mask_t = self.mask[t]
        mask_t = mask_t.to(self.device)
        # sigma_t
        sigma_t = self.get_sigma(min(t, 140))

        # At(x0)
        x_k = fastmri.fft2c(sr.permute(0, 2, 3, 1))

        xt_kmean = x_k * mask_t

        xt_condition_kmean = x_k * mask_condition

        xt_mean = fastmri.ifft2c(xt_kmean)
        xt_condition = fastmri.ifft2c(xt_condition_kmean)
        xt_noise = torch.randn_like(xt_mean)
        xt = xt_mean + sigma_t * xt_noise
        xt = xt.permute(0, 3, 1, 2)

        time_s = (s * torch.ones((b, 1, 1, 1))).long()
        mask_s = self.set_mask(time_s)
        xs = self.get_xt_mean(x0, mask_s)
        return xt, mask_t, xt_mean.permute(0, 3, 1, 2), xt_condition.permute(0, 3, 1, 2), xs, mask_s


    def generate_random_degenerate_gu(self, x0, is_con=False):
        '''
        Training stage
        Randomly Generate Time t, Get Corresponding t and Degraded Image xt
        '''
        x0 = x0.to(self.device)

        batch_size = x0.shape[0]


        timesteps = torch.randint(1,self.N +1,(batch_size,)).long()

        res_in = (320-timesteps *2 ,320-timesteps *2)


        mask_t_condition = self.set_mask(140)
        xt_mean_condition = self.get_xt_mean(x0,mask_t_condition)

        mask_t = self.set_mask(timesteps)
        sigma_t = self.get_sigma(timesteps)

        xt_mean = self.get_xt_mean(x0,mask_t)
 
        xt_noise = torch.randn_like(x0)

        xt = xt_mean + sigma_t * xt_noise
        timesteps = timesteps
        
        if is_con and timesteps[0]>1:
            time_s =   torch.randint(1,timesteps[0].data,(batch_size,)).long()
            mask_s = self.set_mask(time_s)
            xs = self.get_xt_mean(x0,mask_s)
            res_s = (320- time_s*2, 320- time_s *2)
            res = self.get_res_s(res_in,res_s)
        else:
            xs = x0
            time_s = torch.randint(1,2,(batch_size,)).long()
            res_s = (320, 320)
            res = self.get_res_s(res_in,res_s)
            mask_s = torch.ones_like(mask_t)

        return timesteps,xt,xt_mean_condition,time_s,xs,res,mask_s


    def get_res_s(
        self, input_size,target_size
    ):

        add_time_ids = list(input_size  + target_size)
        add_time_ids = torch.tensor([add_time_ids])
        return add_time_ids

    def process_x0(self,x0):
        return x0.clamp(-1, 1)


def tv_loss(x):
    batch_size, c, h, w = x.size()
    tv_h = torch.abs(x[:,:,1:,:] - x[:,:,:-1,:]).sum()
    tv_w = torch.abs(x[:,:,:,1:] - x[:,:,:,:-1]).sum()
    return (tv_h + tv_w) / (batch_size * c * h * w)

