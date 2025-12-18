import numpy as np
import torch
import torch.nn.functional as F

# this file deals with timestep sampling and training target of rectified flow TODO discuss with yanglei to check the correctness
# please refer to wikipedia https://en.wikipedia.org/wiki/Logit-normal_distribution
# please refer to paper eq 19, 23
# NOTE, the timesteps in Chenyang's code may be not correct, since they are not continuous


class SD3RectFlow:

    def __init__(
        self,
        logit_mean: float = 0.0,
        logid_std: float = 1.0,
        base_height: int = 256,
        base_width: int = 256,
        base_frames: int = 1,
        base_scale: float = 1,
    ) -> None:
        self.t_scale = 1000
        self.logit_mean = logit_mean
        self.logid_std = logid_std
        self.base_height = base_height
        self.base_width = base_width
        self.base_frames = base_frames
        self.base_scale = base_scale

    def sample_t_and_sigma(
        self,
        batch_size: int,
        frames: int,
        height: int,
        width: int,
        sample_type: str = 'logitnorm',
        device=None,
    ):
        if sample_type == 'logitnorm':
            t = self.logitnorm_sample_t(
                batch_size, self.logit_mean, self.logid_std, device=device)
        elif sample_type == 'uniform':
            t = torch.rand((batch_size, ), device=device)
        sigma = self.sigma_shift(t, frames, height, width)
        # NOTE in the training process, the t is also shifted according to resolution
        return sigma

    @staticmethod
    def logitnorm_sample_t(
        batch_size: int,
        logit_mean: float = 0.0,
        logid_std: float = 1.0,
        device=None,
    ):
        """
        NOTE the returned t is in range [0, 1], before sending into transformer, you need to scale it by num_train_steps = 1000
        """
        t = torch.normal(
            mean=logit_mean,
            std=logid_std,
            size=(batch_size, ),
            device=device or 'cpu')
        t = F.sigmoid(t)
        return t

    def sigma_shift(
        self,
        t,
        frames,
        height,
        width,
    ):
        """according to eq 23, and  the shift scale for 1024 is 4 here, not
        consistent with paper, but It's ok.

        Change of base_scale will make shift_scale of 256 != 1
        """
        shift_scale = self.get_shift_scale(frames, height, width)
        return self.sigma_shift_given_shift_scale(t, shift_scale)

    @staticmethod
    def sigma_shift_given_shift_scale(t, shift_scale):
        sigma = t * shift_scale / (1 + (shift_scale - 1) * t)
        return sigma

    @staticmethod
    def step(
        xt,
        v_pred,
        sigma,
        sigma_next,
        cache_dtype=torch.float32,
        return_denoised: bool = False,
    ):
        dtype = xt.dtype
        xt = xt.to(cache_dtype)
        sigma = sigma.to(cache_dtype)
        v_pred = v_pred.to(cache_dtype)
        sigma_next = sigma_next.to(cache_dtype)
        """
        # xt = x0 + sigma * (epsilon - x0) this line is pseudo code
        x0 = xt - v_pred * sigma
        epsilon = (xt - x0) / sigma + x0
        xt_next = x0 + sigma_next * (epsilon - x0)
        #  the above three lines are equivalent to the following line
        """
        xt_next = (xt + v_pred * (sigma_next - sigma)).to(dtype)
        if not return_denoised:
            return xt_next
        else:
            x0 = (xt - v_pred * sigma).to(dtype)
            return xt_next, x0

    def retrieve_inference_timesteps_and_sigma_given_shift_scale(
            self, num_inference_steps: int, shift_scale: int = 3, device=None):
        if device is None:
            device = 'cpu'
        t_min, t_max = 1 / self.t_scale, 1
        sigma_min, sigma_max = self.sigma_shift_given_shift_scale(
            np.array([t_min, t_max]),
            shift_scale=shift_scale)  # adapt sigma min and max
        t = np.linspace(sigma_max, sigma_min, num_inference_steps)  # sample t
        sigmas = self.sigma_shift_given_shift_scale(
            t, shift_scale)  # then shift t
        sigmas = torch.from_numpy(sigmas).to(
            dtype=torch.float32, device=device)
        sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])
        return sigmas

    def get_shift_scale(
        self,
        frames: int,
        height: int,
        width: int,
    ):
        shift_scale = (
            (height * width * frames) /
            (self.base_height * self.base_width * self.base_frames))**0.5
        return shift_scale

    def retrieve_inference_timesteps_and_sigma(
        self,
        num_inference_steps: int,
        frames: int,
        height: int,
        width: int,
        device=None,
    ):
        if device is None:
            device = 'cpu'
        shift_scale = self.get_shift_scale(frames, height, width)
        return self.retrieve_inference_timesteps_and_sigma_given_shift_scale(
            num_inference_steps, shift_scale, device=device)

    def loss_weight(self, sigmas, weight_type: str = 'SNR'):
        if weight_type == 'SNR':
            w = (1 - sigmas)**2
            raise NotImplementedError(f'Not fully implement SNR: w={w}')
        elif weight_type == 'min-SNR':
            w = min((1 - sigmas)**2, 0.5)
            raise NotImplementedError(f'Not fully implement min-SNR: w={w}')
        elif weight_type == 'ones':
            return torch.ones_like(sigmas)
