import math
import numpy as np
import random
import torch
from collections.abc import Mapping, Sequence
from typing import List, Optional, Tuple, Union


def format_numel_str(numel: int) -> str:
    B = 1024**3
    M = 1024**2
    K = 1024
    if numel >= B:
        return f"{numel / B:.2f} B"
    elif numel >= M:
        return f"{numel / M:.2f} M"
    elif numel >= K:
        return f"{numel / K:.2f} K"
    else:
        return f"{numel}"

def sample_from_range(value: Union[float, int, Sequence, Mapping],
                      device: Optional[torch.device] = None,
                      generator: Optional[torch.Generator] = None):
    """Sample a float from a provided range description."""
    if value is None:
        return value

    if isinstance(value, (float, int)):
        return float(value)

    low, high = None, None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 2:
            raise ValueError(f"Range specification must have length 2, got {value}.")
        low, high = value
    elif isinstance(value, Mapping):
        if 'range' in value:
            return sample_from_range(value['range'], device=device, generator=generator)
        if 'min' in value and 'max' in value:
            low, high = value['min'], value['max']
        elif 'low' in value and 'high' in value:
            low, high = value['low'], value['high']
        else:
            raise ValueError(
                "Range dictionary must include either 'min'/'max', 'low'/'high', or 'range'."
            )
    else:
        raise TypeError(f"Unsupported range specification type: {type(value)}")

    low = float(low)
    high = float(high)
    if low > high:
        low, high = high, low
    if math.isclose(low, high, rel_tol=1e-8, abs_tol=1e-8):
        return low

    rand_val = torch.rand((), device=device, generator=generator).item()
    return low + (high - low) * rand_val


def create_ref_motion(ref_motion,
                         ref_motion_mask,
                         corrupt_rate=0.1,
                         noise_scale=0.02,
                         replace_noise_rate=0.05,
                         dropout_rate=0.0,
                         temporal_dropout_rate=0.02,
                         is_test=False,
                         jitter_strength=0.3):
    """
    Apply a mild corruption procedure that mimics the noise patterns observed in
    vision-based motion capture (small jitter, occasional temporal flicker).

    Args:
        ref_motion (Tensor): [B, T, C] reference motion (local joints + optional global orient).
        ref_motion_mask (Tensor): [B, T] validity mask.
        corrupt_rate (float): probability of applying Gaussian noise to a frame.
        noise_scale (float): standard deviation of the Gaussian noise.
        replace_noise_rate (float): probability of replacing a frame with a blend
            of neighbouring frames (simulates short-term jitter).
        dropout_rate (float): probability of dropping the entire reference for a sample.
            Kept for API compatibility; defaults to 0.
        temporal_dropout_rate (float): probability of masking small temporal spans.
        is_test (bool): bypass corruption if True.
        jitter_strength (float): controls the interpolation weight range when blending neighbours.
    """
    if is_test:
        return ref_motion, ref_motion_mask

    device = ref_motion.device
    corrupt_rate = sample_from_range(corrupt_rate, device=device)
    noise_scale = sample_from_range(noise_scale, device=device)
    replace_noise_rate = sample_from_range(replace_noise_rate, device=device)

    B, T, C = ref_motion.shape
    ref_motion_aug = ref_motion.clone()
    mask = ref_motion_mask.clone()
    valid_mask = mask.bool()

    if dropout_rate > 0:
        drop_flags = torch.rand(B, device=device) < dropout_rate
        if drop_flags.any():
            ref_motion_aug[drop_flags] = 0
            mask[drop_flags] = 0
            valid_mask = mask.bool()

    if noise_scale > 0 and corrupt_rate > 0:
        frame_noise = torch.randn_like(ref_motion_aug) * noise_scale
        apply_noise = (torch.rand(B, T, device=device) < corrupt_rate) & valid_mask
        ref_motion_aug = ref_motion_aug + frame_noise * apply_noise.unsqueeze(-1)

    if replace_noise_rate > 0:
        jitter_flags = (torch.rand(B, T, device=device) < replace_noise_rate) & valid_mask
        if jitter_flags.any():
            base = ref_motion.clone()
            prev = torch.roll(base, shifts=1, dims=1)
            next = torch.roll(base, shifts=-1, dims=1)
            prev_valid = torch.roll(valid_mask, shifts=1, dims=1)
            next_valid = torch.roll(valid_mask, shifts=-1, dims=1)
            prev = torch.where(prev_valid.unsqueeze(-1), prev, base)
            next = torch.where(next_valid.unsqueeze(-1), next, base)
            alpha = 0.5 + jitter_strength * (torch.rand(B, T, 1, device=device) - 0.5)
            alpha = alpha.clamp(0.0, 1.0)
            blended = alpha * prev + (1 - alpha) * next
            ref_motion_aug = torch.where(jitter_flags.unsqueeze(-1), blended, ref_motion_aug)

    if temporal_dropout_rate > 0:
        drop_flags = (torch.rand(B, T, device=device) < temporal_dropout_rate) & valid_mask
        if drop_flags.any():
            ref_motion_aug = ref_motion_aug * (~drop_flags).unsqueeze(-1)
            mask = mask * (~drop_flags).to(mask.dtype)

    return ref_motion_aug, mask


def gaussian_kernel(kernel_size: int, sigma: float):
    x = torch.linspace(-(kernel_size - 1) / 2, (kernel_size - 1) / 2,
                       kernel_size)
    gauss = torch.exp(-x**2 / (2 * sigma**2))
    return gauss / gauss.sum()


def smooth_motion_rep(motion_rep, kernel_size: int, sigma: float):
    """Temporal Gaussian smoothing for motion representation."""
    assert kernel_size % 2 == 1, 'kernel_size must be odd'
    data_dim = motion_rep.shape[-1]
    padding = (kernel_size - 1) // 2
    kernel = gaussian_kernel(kernel_size,
                             sigma).to(motion_rep.device)[None, None,
                                                          :].repeat(
                                                              data_dim, 1, 1)
    motion_rep_smoothed = torch.nn.functional.conv1d(
        motion_rep.transpose(0, 1).unsqueeze(0),
        kernel,
        padding=padding,
        groups=data_dim)
    motion_rep_smoothed = motion_rep_smoothed.squeeze(0).transpose(0, 1)
    motion_rep_smoothed[:padding] = motion_rep[:padding]
    motion_rep_smoothed[-padding:] = motion_rep[-padding:]
    return motion_rep_smoothed


def maybe_corrupt_ref_motion(ref_motion: torch.Tensor,
                             ref_motion_mask: torch.Tensor,
                             cfg: dict | None,
                             is_test: bool = False):
    """Apply mild corruption to ref_motion using the shared utils routine."""
    if not cfg or not cfg.get('enable', False):
        return ref_motion, ref_motion_mask

    device = ref_motion.device

    def _get_value(key: str, default: float):
        range_key = f'{key}_range'
        if cfg.get(range_key, None) is not None:
            return cfg.get(range_key)
        return cfg.get(key, default)

    corruption_kwargs = dict(
        corrupt_rate=sample_from_range(_get_value('corrupt_rate', 0.1),
                                       device=device),
        noise_scale=sample_from_range(_get_value('noise_scale', 0.02),
                                      device=device),
        replace_noise_rate=sample_from_range(
            _get_value('replace_noise_rate', 0.05), device=device),
        dropout_rate=cfg.get('dropout_rate', 0.0),
        temporal_dropout_rate=cfg.get('temporal_dropout_rate', 0.0),
        jitter_strength=cfg.get('jitter_strength', 0.3),
    )
    corrupted, corrupted_mask = create_ref_motion(
        ref_motion,
        ref_motion_mask,
        is_test=is_test,
        **corruption_kwargs,
    )
    return corrupted, corrupted_mask
