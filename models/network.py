import math
import torch
from inspect import isfunction
from functools import partial
import nibabel as nib
import numpy as np
from tqdm import tqdm
from core.base_network import BaseNetwork
class Network(BaseNetwork):
    def __init__(self, unet, beta_schedule, module_name='sr3', **kwargs):
        super(Network, self).__init__(**kwargs)
        if module_name == 'sr3':
            from .sr3_modules.unet import UNet
        elif module_name == 'sr3_3d':
            from .sr3_modules_3d.unet import UNet
        elif module_name == 'guided_diffusion':
            from .guided_diffusion_modules.unet import UNet
        elif module_name == 'guided_diffusion_3d':
            from .guided_diffusion_modules_3d.unet import UNet
        
        self.denoise_fn = UNet(**unet)
        self.beta_schedule = beta_schedule
        # Detect if using 3D module based on module_name
        self.is_3d = module_name.endswith('_3d')

    def set_loss(self, loss_fn):
        self.loss_fn = loss_fn

    def set_new_noise_schedule(self, device=None, phase='train'):
        # Auto-detect device from model parameters if not provided
        if device is None:
            if next(self.parameters()).is_cuda:
                device = next(self.parameters()).device
            elif torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        betas = make_beta_schedule(**self.beta_schedule[phase])
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        
        gammas = np.cumprod(alphas, axis=0)
        gammas_prev = np.append(1., gammas[:-1])

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('gammas', to_torch(gammas))
        self.register_buffer('sqrt_recip_gammas', to_torch(np.sqrt(1. / gammas)))
        self.register_buffer('sqrt_recipm1_gammas', to_torch(np.sqrt(1. / gammas - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - gammas_prev) / (1. - gammas)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(gammas_prev) / (1. - gammas)))
        self.register_buffer('posterior_mean_coef2', to_torch((1. - gammas_prev) * np.sqrt(alphas) / (1. - gammas)))

    def predict_start_from_noise(self, y_t, t, noise):
        return (
            extract(self.sqrt_recip_gammas, t, y_t.shape) * y_t -
            extract(self.sqrt_recipm1_gammas, t, y_t.shape) * noise
        )

    def q_posterior(self, y_0_hat, y_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, y_t.shape) * y_0_hat +
            extract(self.posterior_mean_coef2, t, y_t.shape) * y_t
        )
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, y_t.shape)
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, y_t, t, clip_denoised: bool, y_cond=None):
        # Support both 2D and 3D: determine x_shape based on input tensor dimensions
        if self.is_3d:
            x_shape = (1, 1, 1, 1, 1)  # 3D: B, C, D, H, W
        else:
            x_shape = (1, 1, 1, 1)  # 2D: B, C, H, W
        noise_level = extract(self.gammas, t, x_shape=x_shape).to(y_t.device)
        
        # Flatten noise_level to (b,) for denoise_fn
        
        if 0:
            noise_level_flat = noise_level.view(noise_level.shape[0], -1)[:, 0]
            y_0_hat = self.predict_start_from_noise(
                    y_t, t=t, noise=self.denoise_fn(torch.cat([y_cond, y_t], dim=1), noise_level_flat))
        if 1:
            y_0_hat = self.predict_start_from_noise(
                    y_t, t=t, noise=self.denoise_fn(torch.cat([y_cond, y_t], dim=1), t))

        if clip_denoised:
            y_0_hat.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            y_0_hat=y_0_hat, y_t=y_t, t=t)
        return model_mean, posterior_log_variance

    def q_sample(self, y_0, sample_gammas, noise=None):
        noise = default(noise, lambda: torch.randn_like(y_0))
        return (
            sample_gammas.sqrt() * y_0 +
            (1 - sample_gammas).sqrt() * noise
        )

    @torch.no_grad()
    def p_sample(self, y_t, t, clip_denoised=True, y_cond=None):
        model_mean, model_log_variance = self.p_mean_variance(
            y_t=y_t, t=t, clip_denoised=clip_denoised, y_cond=y_cond)
        noise = torch.randn_like(y_t) if any(t>0) else torch.zeros_like(y_t)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def restoration(self, y_cond, y_t=None, y_0=None, mask=None, sample_num=8):
        b, *_ = y_cond.shape

        assert self.num_timesteps > sample_num, 'num_timesteps must greater than sample_num'
        sample_inter = (self.num_timesteps//sample_num)
        
        y_t = default(y_t, lambda: torch.randn_like(y_cond))
        ret_arr = y_t
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            t = torch.full((b,), i, device=y_cond.device, dtype=torch.long)
            #if i == 500:
            #  affine = np.eye(4)  # 最簡單的 affine（不含真實空間資訊）
            #  nii = nib.Nifti1Image(y_cond[0, 0, ::].permute(1, 0, 2).detach().cpu().numpy(), affine)
            #  nib.save(nii, "tmp/y_cond_500.nii.gz")

            #  affine = np.eye(4)  # 最簡單的 affine（不含真實空間資訊）
            #  nii = nib.Nifti1Image(y_t[0, 0, ::].permute(1, 0, 2).detach().cpu().numpy(), affine)
            #  nib.save(nii, "tmp/y_t_500.nii.gz")
            #  print(y_cond.max(), y_cond.min())
            #  print(y_t.max(), y_t.min())
              
            #if i == 2:
            #  affine = np.eye(4)  # 最簡單的 affine（不含真實空間資訊）
            #  nii = nib.Nifti1Image(y_cond[0, 0, ::].permute(1, 0, 2).detach().cpu().numpy(), affine)
            #  nib.save(nii, "tmp/y_cond_10.nii.gz")

            #  affine = np.eye(4)  # 最簡單的 affine（不含真實空間資訊）
            #  nii = nib.Nifti1Image(y_t[0, 0, ::].permute(1, 0, 2).detach().cpu().numpy(), affine)
            #  nib.save(nii, "tmp/y_t_10.nii.gz")
            #  print(y_cond.max(), y_cond.min())
            #  print(y_t.max(), y_t.min())
            y_t = self.p_sample(y_t, t, y_cond=y_cond)
            # !!!!!!!!!最原先做法!!!!!!!!!!!
            if 1:
                if mask is not None:
                    y_t = y_0*(1.-mask) + mask*y_t
            # 實驗後記得改回來
            if 0:
                if mask is not None:
                    gamma_t = extract(self.gammas, t, y_0.shape).to(y_0.device)
                    noise = torch.randn_like(y_0)
                    y_known_t = gamma_t.sqrt() * y_0 + (1 - gamma_t).sqrt() * noise
                    y_t = y_known_t * (1. - mask) + y_t * mask
                     
            if i % sample_inter == 0:
                ret_arr = torch.cat([ret_arr, y_t], dim=0)
        return y_t, ret_arr

    def forward(self, y_0, y_cond=None, mask=None, noise=None):
        # sampling from p(gammas)
        b, *_ = y_0.shape
        # Support both 2D and 3D: determine x_shape based on input tensor dimensions
        if self.is_3d:
            x_shape = (1, 1, 1, 1, 1)  # 3D: B, C, D, H, W
        else:
            x_shape = (1, 1, 1, 1)  # 2D: B, C, H, W
        t = torch.randint(1, self.num_timesteps, (b,), device=y_0.device).long()
        if 0:
            gamma_t1 = extract(self.gammas, t-1, x_shape=x_shape)
            sqrt_gamma_t2 = extract(self.gammas, t, x_shape=x_shape)
            # sample_gammas will have shape matching x_shape (e.g., (b, 1, 1, 1, 1) for 3D)
            sample_gammas = (sqrt_gamma_t2-gamma_t1) * torch.rand((b, 1), device=y_0.device) + gamma_t1
            # Flatten to (b, 1) for easier handling
            sample_gammas_flat = sample_gammas.view(b, -1)[:, 0:1]  # Ensure shape is (b, 1)
        if 1:
            sample_gammas = extract(self.gammas, t, x_shape=x_shape)
            sample_gammas_flat = sample_gammas.view(b, 1)

        noise = default(noise, lambda: torch.randn_like(y_0))
        # Support both 2D (4D tensor: B,C,H,W) and 3D (5D tensor: B,C,D,H,W)
        if self.is_3d:
            # 3D: expand to (B, 1, 1, 1, 1) for broadcasting with (B, C, D, H, W)
            sample_gammas_expanded = sample_gammas_flat.view(b, 1, 1, 1, 1)
        else:
            # 2D: expand to (B, 1, 1, 1) for broadcasting with (B, C, H, W)
            sample_gammas_expanded = sample_gammas_flat.view(b, 1, 1, 1)
        y_noisy = self.q_sample(
            y_0=y_0, sample_gammas=sample_gammas_expanded, noise=noise)
        

        if mask is not None:
            # For inpainting: mask should have same spatial dimensions as y_0
            # 2D: mask shape should be (B, 1, H, W) or (B, C, H, W)
            # 3D: mask shape should be (B, 1, D, H, W) or (B, C, D, H, W)
            # Use flattened sample_gammas for denoise_fn (it expects 1D)
            if 0:
                noise_hat = self.denoise_fn(torch.cat([y_cond, y_noisy*mask+(1.-mask)*y_0], dim=1), sample_gammas_flat.squeeze(-1))
            if 1:
                noise_hat = self.denoise_fn(torch.cat([y_cond, y_noisy*mask+(1.-mask)*y_0], dim=1), t)
                loss = self.loss_fn(mask*noise, mask*noise_hat)
            if 0:
                noise_hat = self.denoise_fn(torch.cat([y_cond, y_noisy], dim=1), t)
                loss = self.loss_fn(noise, noise_hat)
        else:
            if 0:
                noise_hat = self.denoise_fn(torch.cat([y_cond, y_noisy], dim=1), sample_gammas_flat.squeeze(-1))
            if 1:
                noise_hat = self.denoise_fn(torch.cat([y_cond, y_noisy], dim=1), t)
            loss = self.loss_fn(noise, noise_hat)
        return loss


# gaussian diffusion trainer class
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def extract(a, t, x_shape=(1,1,1,1)):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

# beta_schedule function
def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas

def make_beta_schedule(schedule, n_timestep, linear_start=1e-6, linear_end=1e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


