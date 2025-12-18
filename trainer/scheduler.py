import math
import numpy as np
import torch
import torch.distributed as dist
from diffusers import CogVideoXDDIMScheduler as CogVideoXDDIMSchedulerBase
from diffusers import CogVideoXDPMScheduler as CogVideoXDPMSchedulerBase
from diffusers import DDIMScheduler as DDIMSchedulerBase


class ZoeScheduler:

    def get_snr(self, timesteps):
        alphas_cumprod = self.alphas_cumprod[timesteps]
        sigma_power = 1 - alphas_cumprod
        return alphas_cumprod / sigma_power

    def get_min_snr_weight(self, timesteps, gamma: float = 5.0):
        """the original min snr weighting in paper
        https://arxiv.org/abs/2303.09556."""
        snr = self.get_snr(timesteps)
        min_snr_gamma = torch.min(snr, torch.ones_like(snr) * gamma)
        if self.config.prediction_type == 'epsilon':
            return min_snr_gamma / (snr + 1e-8)
        elif self.config.prediction_type == 'sample':
            return min_snr_gamma
        elif self.config.prediction_type == 'v_prediction':
            return min_snr_gamma / (snr + 1)

    def get_min_snr_weight_modified(self,
                                    timesteps,
                                    gamma: float = 5.0,
                                    min_weight: float = 0.01):
        """modified loss weighting, so that the weighting at zero snr is not
        zero the original min snr weighting in paper
        https://arxiv.org/abs/2303.09556."""
        snr = self.get_snr(timesteps)
        min_snr_gamma = torch.min(snr, torch.ones_like(snr) * gamma)
        if self.config.prediction_type == 'epsilon':
            return torch.max(min_snr_gamma / (snr + 1e-8), min_weight)
        elif self.config.prediction_type == 'sample':
            return torch.max(min_snr_gamma, min_weight)
        elif self.config.prediction_type == 'v_prediction':
            return torch.max(min_snr_gamma / (snr + 1), min_weight)

    def get_min_snr_weight_cogvideo(self, timesteps):
        """modified loss weighting, so that the weighting at zero snr is not
        zero the original min snr weighting in paper
        https://arxiv.org/abs/2303.09556."""
        snr = self.get_snr(timesteps)
        if self.config.prediction_type == 'epsilon':
            return (snr + 1) / (snr + 1e-8)
        elif self.config.prediction_type == 'sample':
            return snr + 1
        elif self.config.prediction_type == 'v_prediction':
            return torch.ones_like(snr)


class CogVideoXDDIMScheduler(CogVideoXDDIMSchedulerBase, ZoeScheduler):
    pass


class DDIMScheduler(DDIMSchedulerBase, ZoeScheduler):
    pass


class CogVideoXDPMScheduler(CogVideoXDPMSchedulerBase, ZoeScheduler):
    pass


class TimestepSamplerMP:

    def __init__(self,
                 dp_group: dist.ProcessGroup,
                 num_train_timesteps: int = 1000) -> None:
        self.dp_size = dp_group.size()
        self.dp_rank = dp_group.rank()
        self.train_timesteps = np.arange(num_train_timesteps)
        self.t_step_split = np.array_split(self.train_timesteps, self.dp_size)
        self.iter = 0

    def sample_t(self,
                 batch_size: int,
                 training_iter: int = None) -> np.ndarray:
        if training_iter is None:
            bin_idx = (self.iter + self.dp_rank) % self.dp_size
            self.iter += 1
        else:
            bin_idx = (training_iter + self.dp_rank) % self.dp_size
        bin_steps = self.t_step_split[bin_idx]
        return np.random.choice(bin_steps, size=batch_size, replace=False)


def get_scheduler(beta_type: str = None,
                  use_dpm_solver: bool = False) -> DDIMScheduler:
    ddim_kwargs = dict(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule='scaled_linear',
        clip_sample=False,
        set_alpha_to_one=True,
        steps_offset=1,
        rescale_betas_zero_snr=True,
        timestep_spacing='trailing',
        prediction_type='v_prediction',
    )
    if 'cogvideo' not in beta_type:
        return DDIMScheduler(**ddim_kwargs)
    else:
        snr_shift_scale = 3.0 if '2b' in beta_type else 1.0
        ddim_kwargs['steps_offset'] = 0
        ddim_kwargs['snr_shift_scale'] = snr_shift_scale
        if use_dpm_solver:
            return CogVideoXDPMScheduler(**ddim_kwargs)
        else:
            return CogVideoXDDIMScheduler(**ddim_kwargs)


def get_smooth_dynamic_cfg_scale_list(cfg_scale, num_inference_steps):

    cfg_list = [
        1 + cfg_scale * ((1 - math.cos(math.pi * (
            (num_inference_steps - idx) / num_inference_steps)**5.0)) / 2)
        for idx in range(0, int(num_inference_steps))
    ]

    return cfg_list


def get_cogvideo_dynamic_cfg_scale_list(cfg_scale, num_inference_steps,
                                        timesteps):

    cfg_list = [
        1 + cfg_scale * ((1 - math.cos(math.pi * (
            (num_inference_steps - t.item()) / num_inference_steps)**5.0)) / 2)
        for t in timesteps
    ]

    return cfg_list


# https://arxiv.org/pdf/2404.07724 Applying Guidance in a Limited Interval Improves Sample and Distribution Quality in Diffusion Models
def get_cfg_scale_list(cfg_scale,
                       enable_cfg_interval,
                       timesteps: torch.Tensor,
                       t_scale: int = 1000):
    use_cfg = (timesteps / t_scale) < enable_cfg_interval
    cfg_list = torch.ones_like(timesteps)
    cfg_list[use_cfg] = cfg_scale
    return cfg_list


class FlowMatchScheduler():

    def __init__(self, num_inference_steps=100, num_train_timesteps=1000, shift=5.0, sigma_max=1.0, sigma_min=0.0, inverse_timesteps=False, extra_one_step=True, reverse_sigmas=False):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas
        self.num_inference_steps = num_inference_steps
        self.set_timesteps(num_inference_steps)

    def set_timesteps(self, num_inference_steps=100, denoising_strength=1.0, training=False):
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
        if self.extra_one_step:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps + 1)[:-1]
        else:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps)
        if self.inverse_timesteps:
            self.sigmas = torch.flip(self.sigmas, dims=[0])
        self.sigmas = self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas)
        if self.reverse_sigmas:
            self.sigmas = 1 - self.sigmas
        self.timesteps = self.sigmas * self.num_train_timesteps
        if training:
            x = self.timesteps
            y = torch.exp(-2 * ((x - num_inference_steps / 2) / num_inference_steps) ** 2)
            y_shifted = y - y.min()
            bsmntw_weighing = y_shifted * (num_inference_steps / y_shifted.sum())
            self.linear_timesteps_weights = bsmntw_weighing


    def step(self, model_output, timestep, sample, to_final=False):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        if to_final or timestep_id + 1 >= len(self.timesteps):
            sigma_ = 1 if (self.inverse_timesteps or self.reverse_sigmas) else 0
        else:
            sigma_ = self.sigmas[timestep_id + 1]
        prev_sample = sample + model_output * (sigma_ - sigma)
        return prev_sample
    

    def return_to_timestep(self, timestep, sample, sample_stablized):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        model_output = (sample - sample_stablized) / sigma
        return model_output
    
    
    def add_noise(self, original_samples, noise, timestep):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        # timestep: [batch_size], self.timesteps: [num_inference_steps]
        timestep_id = torch.argmin((self.timesteps - timestep
                                   .unsqueeze(-1).repeat(1, self.timesteps.shape[0])
                                   .to(self.timesteps.device)).abs(), dim=1)
        sigma = self.sigmas[timestep_id].to(noise.device)    # [batch_size]
        # unsqueeze to match the shape of original_samples
        shape_diff = len(original_samples.shape) - len(sigma.shape)
        for _ in range(shape_diff):
            sigma = sigma.unsqueeze(-1)
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample
    

    def training_target(self, sample, noise, timestep):
        target = noise - sample
        return target
    

    def training_weight(self, timestep):

        timestep_id = torch.argmin((self.timesteps - timestep
                                   .unsqueeze(-1).repeat(1, self.timesteps.shape[0])
                                   .to(self.timesteps.device)).abs(), dim=1)
        # timestep_id = torch.argmin((self.timesteps - timestep.to(self.timesteps.device)).abs())
        weights = self.linear_timesteps_weights[timestep_id]
        return weights.to(timestep.device)
