"""Abstract SDE classes, Reverse SDE, and VE/VP SDEs."""
import abc
import torch
import numpy as np
import fastmri
import math
from tqdm import tqdm
import torchvision.utils as tvul
import os

class SDE(abc.ABC):
  """SDE abstract class. Functions are designed for a mini-batch of inputs."""

  def __init__(self, N,device=None):
    """Construct an SDE.

    Args:
      N: number of discretization time steps.
    """
    super().__init__()
    self.N = N
    self.dt = 1 / N
    self.device = device


  @abc.abstractmethod
  def T(self):
    """End time of the SDE."""
    pass

  @abc.abstractmethod
  def sde(self, x, t):
    pass

  @abc.abstractmethod
  def marginal_prob(self, x, t):
    """Parameters to determine the marginal distribution of the SDE, $p_t(x)$."""
    pass

  @abc.abstractmethod
  def prior_sampling(self, rng, shape):
    """Generate one sample from the prior distribution, $p_T(x)$."""
    pass

  @abc.abstractmethod
  def prior_logp(self, z):
    """Compute log-density of the prior distribution.

    Useful for computing the log-likelihood via probability flow ODE.

    Args:
      z: latent code
    Returns:
      log probability density
    """
    pass

  def discretize(self, x, t):
    """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

    Useful for reverse diffusion sampling and probabiliy flow sampling.
    Defaults to Euler-Maruyama discretization.

    Args:
      x: a JAX tensor.
      t: a JAX float representing the time step (from 0 to `self.T`)

    Returns:
      f, G
    """
    dt = 1 / self.N
    drift, diffusion = self.sde(x, t)
    f = drift * dt
    G = diffusion * torch.sqrt(dt)
    return f, G

  def reverse(self, score_fn, probability_flow=False):
    """Create the reverse-time SDE/ODE.

    Args:
      score_fn: A time-dependent score-based model that takes x and t and returns the score.
      probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
    """
    N = self.N
    T = self.T
    sde_fn = self.sde
    discretize_fn = self.discretize

    # Build the class for reverse-time SDE.
    class RSDE(self.__class__):
      def __init__(self):
        self.N = N
        self.probability_flow = probability_flow

      @property
      def T(self):
        return T

      def sde(self, x, t):
        """Create the drift and diffusion functions for the reverse SDE/ODE."""
        drift, diffusion = sde_fn(x, t)
        score = score_fn(x, t)
        drift = drift - diffusion[:, None, None, None] ** 2 * score * (0.5 if self.probability_flow else 1.)
        # Set the diffusion function to zero for ODEs.
        diffusion = torch.zeros_like(diffusion) if self.probability_flow else diffusion
        return drift, diffusion

      def discretize(self, x, t):
        """Create discretized iteration rules for the reverse diffusion sampler."""
        f, G = discretize_fn(x, t)
        score = score_fn(x, t)
        rev_f = f - G[:, None, None, None] ** 2 * score * (0.5 if self.probability_flow else 1.)
        rev_G = torch.zeros_like(G) if self.probability_flow else G
        return rev_f, rev_G

    return RSDE()


class SRSDE:
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
            total_scale = gen_sequence(320, 40, 2)
            for i in range(len(total_scale)):
                m = get_center_mask(image_shape, total_scale[i])
                m = torch.from_numpy(m)
                mm = torch.stack((m,m), dim = 0).view(1,2,320,320)
                mm = mm.permute(0,2,3,1)
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

    def set_mask(self,t):
        mask_t = self.mask[t]
        mask_t = mask_t.to(self.device)
        return mask_t

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
        xt_k = xt_k.permute(0,3,1,2)
        return xt,xt_k
    
    # TODO
    def get_sigma(self,t):
        
        sigma_t = self.sigma_bars[t]

        return sigma_t
    
    def set_model(self,model):
        self.model = model

    # 网络预测重建
    def pred_fn(self, xt, t):
        
        return self.model(xt, t)

    def get_score_from_x0(self,x0_pred,xt, t):
        # mask
        mask_t = self.mask[t]
        mask_t = mask_t.to(self.device)
        A_xt = self.get_xt_mean(x0_pred,mask_t)
        sigma_t = self.get_sigma(t)
        score = (A_xt-xt)/(sigma_t ** 2)
        return score
    
    # 整个逆向重建过程 base
    def reverse_sde(self, xt, N=-1, save_states=False, save_dir='sde_state'):
        N = self.N if N < 0 else N
        x = xt.clone()
        # 开始重建
        for t in tqdm(reversed(range(1, N + 1))):
            prev_t = t-1
            pred_x0 = self.pred_fn(xt,t)
            score = self.get_score_from_x0(pred_x0,xt,t)
            # xt - xt-1
            x = self.reverse_sde_onestep(x, score, t)

            # 保存中间状态 看一下
            if save_states: 
                interval = self.N // 7 #存 7 张 
                if t % interval == 0:
                    idx = t // interval
                    os.makedirs(save_dir, exist_ok=True)
                    tvul.save_image(x.data, f'{save_dir}/state_{idx}.png', normalize=False)

        return x

      # 整个逆向重建过程 plus带有反向投影版本
    def reverse_sde_plus(self, xt,mask,N=-1, save_states=False, save_dir='sde_state'):
        N = self.N if N < 0 else N
        x = xt.clone()
        y = xt.clone()
        # 开始重建
        for t in tqdm(reversed(range(1, N + 1))):
            prev_t = t-1
            pred_x0 = self.pred_fn(x,t)

            pred_x0 = self.inver_fp(pred_x0,mask,y)

            score = self.get_score_from_x0(pred_x0,x,t)
            
            # xt - xt-1
            x = self.reverse_sde_onestep(x, score, t)

            # 保存中间状态 看一下
            if save_states: 
                interval = self.N // 7 #存 7 张 
                if t % interval == 0:
                    idx = t // interval
                    os.makedirs(save_dir, exist_ok=True)
                    tvul.save_image(x.data, f'{save_dir}/state_{idx}.png', normalize=False)

        return x

    # 反投影
    def inver_fp(self,x0,mask,y):
        x_k = fastmri.fft2c(x0.permute(0,2,3,1))
        x_k_inv_mask = x_k *(torch.ones_like(mask)-mask)

        y_k = fastmri.fft2c(y.permut(0,2,3,1))
        y_k_mask = y_k * mask

        x0 = fastmri.ifft2c(x_k_inv_mask+y_k_mask)

        x0 = x0.permute(0,3,1,2)
        
        return x0




    # 单步重建
    def reverse_sde_onestep(self,xt,pred_x0,t,t_prev):
        mask_t = self.mask[t]
        mask_t = mask_t.to(self.device)
        mask_t_prev = self.mask[t_prev]
        mask_t_prev = mask_t_prev.to(self.device)

        sigma_t = self.get_sigma(t)
        sigma_t_prev = self.get_sigma(t_prev)

        increment = self.add_increment(pred_x0,mask_t,mask_t_prev)

        score = self.get_score_from_x0(pred_x0,xt,t)
        denoise = self.sde_denoise(score,mask_t,mask_t_prev,sigma_t,sigma_t_prev)

        noise = torch.randn_like(xt)
        addnoise = self.sde_add_noise(noise,mask_t,mask_t_prev,sigma_t,sigma_t_prev)
        #addnoise = self.sde_add_noise1(noise,mask_t,sigma_t)
      
        x_prev  =xt + increment - denoise + addnoise

        return x_prev 
    
    # 重建过程的增量
    def add_increment(self,x0,mask_t,mask_t_prev):
        
        increment = self.get_xt_mean(x0,mask_t_prev) - self.get_xt_mean(x0,mask_t)

        return increment

    # 重建过程中的分数去噪
    def sde_denoise(self,score,mask_t,mask_t_prev,sigma_t,sigma_t_prev):

        s_t  = self.get_xt_noise(score,mask_t)
        s_t_prev = self.get_xt_noise(score,mask_t_prev)
        
        denoise = sigma_t_prev * s_t_prev - sigma_t * s_t

        return denoise

    # 重建过程中的郎之万加噪 TODO 这个地方可能需要在研究一下
    def sde_add_noise(noise,mask_t,mask_t_prev,sigma_t,sigma_t_prev):

        return 0

    # 近似每步加的噪声
    def sde_add_noise1(self,noise,mask_t,sigma_t):
        noise_q = self.get_xt_noise(noise,mask_t)
        return sigma_t * noise_q * math.sqrt(self.dt).to(self.device)
    

    # 前向过程
    def forward(self, x0, N=-1, save_dir='forward_state'):
        N = self.N if N < 0 else N
        x = x0.clone()
        for t in tqdm(range(0, N + 1)):
            xt,xt_k = self.get_xt1(x, t)
            os.makedirs(save_dir, exist_ok=True)
            xt_k1 = torch.abs(xt_k[:,0,:,:]+xt_k[:,1,:,:]*1j)
            tvul.save_image(xt_k1.data, f'{save_dir}/state_{t}.png', normalize=False)
            import imageio
            xt = xt.cpu().data.numpy()
            xt_k = xt_k.cpu().data.numpy()

            out = np.abs(xt[0,0]+xt[0,1]*1j)
            out_k = np.abs(xt_k[0,0]+xt_k[0,1]*1j)
            
            imageio.imsave(f'{save_dir}/x_{t}.png',out/out.max())
            imageio.imsave(f'{save_dir}/k_{t}.png',out_k)
            
            
        return x

    # 得到时间t处的退化图像，方便在测试时候使用
    # 得到时间t处的退化图像，方便在测试时候使用
    def get_deg_t(self,x0,t=None):
        x0 = x0.to(self.device)
        b = x0.shape[0]
        # 这里默认在训练时候的验证只验证8倍超分的情况
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

        ## 环城膝关节后需要fftshit一下
        # x_k= fastmri.fftshift(x_k,dim=[-3, -2])


        xt_kmean = x_k * mask_t
        xt_condition_kmean = x_k * mask_condition

        xt_mean = fastmri.ifft2c(xt_kmean)
        xt_condition = fastmri.ifft2c(xt_condition_kmean)

        xt_noise = torch.randn_like(xt_mean)
        #xt_noise = self.get_xt_noise(noise,mask_t)

        xt = xt_mean + sigma_t * xt_noise
        
        xt = xt.permute(0,3,1,2)

        return xt, mask_t, xt_mean.permute(0,3,1,2),xt_condition.permute(0,3,1,2)

    def get_deg_t_s(self,x0,t=None,s=None):
        x0 = x0.to(self.device)
        b = x0.shape[0]
        # 这里默认在训练时候的验证只验证8倍超分的情况
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

        ## 环城膝关节后需要fftshit一下
        # x_k= fastmri.fftshift(x_k,dim=[-3, -2])


        xt_kmean = x_k * mask_t
        xt_condition_kmean = x_k * mask_condition

        xt_mean = fastmri.ifft2c(xt_kmean)
        xt_condition = fastmri.ifft2c(xt_condition_kmean)

        xt_noise = torch.randn_like(xt_mean)
        #xt_noise = self.get_xt_noise(noise,mask_t)

        xt = xt_mean + sigma_t * xt_noise
        
        xt = xt.permute(0,3,1,2)

        return xt, mask_t, xt_mean.permute(0,3,1,2),xt_condition.permute(0,3,1,2)
    
    # 随机生成退图像以及时间t
    def generate_random_degenerate(self,x0):
        '''
        训练阶段
        该函数功能为,随机生成一个时间t,然后得到对应的时间t以及退化后的图像xt
        '''
        x0 = x0.to(self.device)

        batch_size = x0.shape[0]

        # 随机选取时间 训练时要用
        timesteps = torch.randint(1,self.N +1,(batch_size,1,1,1)).long()

        ######################################################
        # (这里可以看code说明为什么有两段式生成或者一段式生成)
        ######################################################


        ######################################################
        # 一段式生成，或者说直接生成（默认）
        ######################################################

        xt = self.get_xt(x0,timesteps)


        ######################################################
        # 两段式生成
        ######################################################

        # mask_t = self.set_mask(timesteps)
        # sigma_t = self.get_sigma(timesteps)

        # xt_mean = self.get_xt_mean(x0,mask_t)
 
        # noise = torch.randn_like(x0)
        # xt_noise = self.get_xt_noise(noise,mask_t)

        # xt = xt_mean + sigma_t * xt_noise


        return timesteps,xt

