import math
from functools import partial
from inspect import isfunction

import numpy as np
import torch
from tqdm import tqdm

from core.base_network import BaseNetwork


class Network(BaseNetwork):
    def __init__(self, unet, beta_schedule, module_name='spade_diffusion_3d', **kwargs):
        super(Network, self).__init__(**kwargs)
        if module_name == 'guided_diffusion_3d':
            from .guided_diffusion_modules_3d.unet import UNet
        elif module_name == 'spade_diffusion_3d':
            from .guided_spade_diffusion_modules_3d.unet import UNet
        else:
            raise ValueError(
                f'Unknown module_name: {module_name}. '
                'Supported: guided_diffusion_3d, spade_diffusion_3d'
            )

        self.denoise_fn = UNet(**unet)
        self.beta_schedule = beta_schedule
        self.is_3d = True
        self.use_spade = 'spade' in module_name

    def set_loss(self, loss_fn):
        self.loss_fn = loss_fn

    def set_new_noise_schedule(self, device=None, phase='train'):
        if device is None:
            if next(self.parameters()).is_cuda:
                device = next(self.parameters()).device
            elif torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')

        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        betas = make_beta_schedule(**self.beta_schedule[phase])
        betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        alphas = 1. - betas

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        gammas = np.cumprod(alphas, axis=0)
        gammas_prev = np.append(1., gammas[:-1])

        self.register_buffer('gammas', to_torch(gammas))
        self.register_buffer('sqrt_recip_gammas', to_torch(np.sqrt(1. / gammas)))
        self.register_buffer('sqrt_recipm1_gammas', to_torch(np.sqrt(1. / gammas - 1)))

        posterior_variance = betas * (1. - gammas_prev) / (1. - gammas)
        self.register_buffer(
            'posterior_log_variance_clipped',
            to_torch(np.log(np.maximum(posterior_variance, 1e-20))),
        )
        self.register_buffer(
            'posterior_mean_coef1',
            to_torch(betas * np.sqrt(gammas_prev) / (1. - gammas)),
        )
        self.register_buffer(
            'posterior_mean_coef2',
            to_torch((1. - gammas_prev) * np.sqrt(alphas) / (1. - gammas)),
        )

    def _denoise(self, y_cond, y_t, t, seg=None, vessel_seg=None):
        if vessel_seg is not None:
            model_input = torch.cat([y_cond, y_t, vessel_seg], dim=1)
        else:
            model_input = torch.cat([y_cond, y_t], dim=1)

        if self.use_spade and seg is not None:
            return self.denoise_fn(model_input, t, seg)
        return self.denoise_fn(model_input, t)

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
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, y_t.shape
        )
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, y_t, t, clip_denoised: bool, y_cond=None, seg=None, vessel_seg=None):
        noise_hat = self._denoise(y_cond, y_t, t, seg=seg, vessel_seg=vessel_seg)
        y_0_hat = self.predict_start_from_noise(y_t, t=t, noise=noise_hat)

        if clip_denoised:
            y_0_hat.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            y_0_hat=y_0_hat, y_t=y_t, t=t
        )
        return model_mean, posterior_log_variance, y_0_hat

    def q_sample(self, y_0, sample_gammas, noise=None):
        noise = default(noise, lambda: torch.randn_like(y_0))
        return sample_gammas.sqrt() * y_0 + (1 - sample_gammas).sqrt() * noise

    @torch.no_grad()
    def p_sample(self, y_t, t, clip_denoised=True, y_cond=None, seg=None, vessel_seg=None):
        model_mean, model_log_variance, y_0_hat = self.p_mean_variance(
            y_t=y_t,
            t=t,
            clip_denoised=clip_denoised,
            y_cond=y_cond,
            seg=seg,
            vessel_seg=vessel_seg,
        )
        noise = torch.randn_like(y_t) if any(t > 0) else torch.zeros_like(y_t)
        return model_mean + noise * (0.5 * model_log_variance).exp(), y_0_hat

    @torch.no_grad()
    def restoration(self, y_cond, y_t=None, y_0=None, mask=None, seg=None, vessel_seg=None, sample_num=8):
        b, *_ = y_cond.shape
        assert self.num_timesteps > sample_num, 'num_timesteps must greater than sample_num'
        sample_inter = self.num_timesteps // sample_num

        y_t = default(y_t, lambda: torch.randn_like(y_cond))
        ret_arr = y_t

        for i in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc='sampling loop time step',
            total=self.num_timesteps,
        ):
            t = torch.full((b,), i, device=y_cond.device, dtype=torch.long)

            if mask is not None:
                y_t = y_0 * (1. - mask) + mask * y_t

            y_t, _ = self.p_sample(
                y_t, t, y_cond=y_cond, seg=seg, vessel_seg=vessel_seg
            )

            if mask is not None:
                y_t = y_0 * (1. - mask) + mask * y_t

            if i % sample_inter == 0:
                ret_arr = torch.cat([ret_arr, y_t], dim=0)

        return y_t, ret_arr

    def forward(self, y_0, y_cond=None, mask=None, seg=None, vessel_seg=None, noise=None):
        b, *_ = y_0.shape
        t = torch.randint(1, self.num_timesteps, (b,), device=y_0.device).long()

        sample_gammas = extract(self.gammas, t, x_shape=(1, 1, 1, 1, 1))
        sample_gammas_expanded = sample_gammas.view(b, 1, 1, 1, 1)

        noise = default(noise, lambda: torch.randn_like(y_0))
        y_noisy = self.q_sample(y_0=y_0, sample_gammas=sample_gammas_expanded, noise=noise)

        if mask is not None:
            y_input = y_noisy * mask + (1. - mask) * y_0
            noise_hat = self._denoise(y_cond, y_input, t, seg=seg, vessel_seg=vessel_seg)
            loss = self.loss_fn(mask * noise, mask * noise_hat)
        else:
            noise_hat = self._denoise(y_cond, y_noisy, t, seg=seg, vessel_seg=vessel_seg)
            loss = self.loss_fn(noise, noise_hat)
        return loss


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def extract(a, t, x_shape=(1, 1, 1, 1)):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-6, linear_end=1e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end, n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end, n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':
        betas = 1. / np.linspace(n_timestep, 1, n_timestep, dtype=np.float64)
    elif schedule == 'cosine':
        timesteps = torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas
