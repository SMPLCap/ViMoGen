import inspect
import os
import safetensors
import torch
from typing import List, Optional, Tuple, Union


def randn_tensor(
    logger,
    shape: Union[Tuple, List],
    generator: Optional[Union[List['torch.Generator'],
                              'torch.Generator']] = None,
    device: Optional['torch.device'] = None,
    dtype: Optional['torch.dtype'] = None,
    layout: Optional['torch.layout'] = None,
):
    """A helper function to create random tensors on the desired `device` with
    the desired `dtype`.

    When passing a list of generators, you can seed each batch size
    individually. If CPU generators are passed, the tensor is always created on
    the CPU.
    """
    # device on which tensor is created defaults to device
    rand_device = device
    batch_size = shape[0]

    layout = layout or torch.strided
    device = device or torch.device('cpu')

    if generator is not None:
        gen_device_type = generator.device.type if not isinstance(
            generator, list) else generator[0].device.type
        if gen_device_type != device.type and gen_device_type == 'cpu':
            rand_device = 'cpu'
            if device != 'mps':
                logger.info(
                    f"The passed generator was created on 'cpu' even though a tensor on {device} was expected."
                    f" Tensors will be created on 'cpu' and then moved to {device}. Note that one can probably"
                    f' slightly speed up this function by passing a generator that was created on the {device} device.'
                )
        elif gen_device_type != device.type and gen_device_type == 'cuda':
            raise ValueError(
                f'Cannot generate a {device} tensor from a generator of type {gen_device_type}.'
            )

    # make sure generator list of length 1 is treated like a non-list
    if isinstance(generator, list) and len(generator) == 1:
        generator = generator[0]

    if isinstance(generator, list):
        shape = (1, ) + shape[1:]
        latents = [
            torch.randn(
                shape,
                generator=generator[i],
                device=rand_device,
                dtype=dtype,
                layout=layout) for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0).to(device)
    else:
        latents = torch.randn(
            shape,
            generator=generator,
            device=rand_device,
            dtype=dtype,
            layout=layout).to(device)

    return latents


def prepare_extra_step_kwargs(scheduler, generator, eta=0.0):
    # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
    # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
    # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
    # and should be between [0, 1]
    accepts_eta = 'eta' in set(
        inspect.signature(scheduler.step).parameters.keys())
    extra_step_kwargs = {}
    if accepts_eta:
        extra_step_kwargs['eta'] = eta
    # check if the scheduler accepts generator
    accepts_generator = 'generator' in set(
        inspect.signature(scheduler.step).parameters.keys())
    if accepts_generator:
        extra_step_kwargs['generator'] = generator
    return extra_step_kwargs


def count_trainable_parameters(named_parameters):
    total_trainable = 0
    total_untrainable = 0
    for name, param in named_parameters:
        if param.requires_grad:
            total_trainable += param.numel()
        else:
            total_untrainable += param.numel()
    return total_trainable, total_untrainable


def load_safetensors(in_path: str):
    if os.path.isdir(in_path):
        return load_safetensors_from_dir(in_path)
    elif os.path.isfile(in_path):
        return load_safetensors_from_path(in_path)
    else:
        raise ValueError(f'{in_path} does not exist')


def load_safetensors_from_path(in_path: str):
    tensors = {}
    with safetensors.safe_open(in_path, framework='pt', device='cpu') as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def load_safetensors_from_dir(in_dir: str):
    tensors = {}
    safetensors = os.listdir(in_dir)
    safetensors = [f for f in safetensors if f.endswith('.safetensors')]
    for f in safetensors:
        tensors.update(load_safetensors_from_path(os.path.join(in_dir, f)))
    return tensors


def guess_cogvideo_config_from_state_dict(state_dict: dict):
    # TODO ipa config is not included yet
    model_config = {
        "activation_fn": "gelu-approximate",
        "attention_bias": True,
        "attention_head_dim": 64,
        "dropout": 0.0,
        "flip_sin_to_cos": True,
        "freq_shift": 0,
        "in_channels": 16,
        "max_text_seq_length": 226,
        "norm_elementwise_affine": True,
        "norm_eps": 1e-05,
        "num_attention_heads": 30,
        "num_layers": 30,
        "out_channels": 16,
        "patch_size": 2,
        "sample_frames": 49,
        "sample_height": 60,
        "sample_width": 90,
        "spatial_interpolation_scale": 1.875,
        "temporal_compression_ratio": 4,
        "temporal_interpolation_scale": 1.0,
        "text_embed_dim": 4096,
        "time_embed_dim": 512,
        "timestep_activation_fn": "silu"
    }

    def count_num_layers():
        layers = [
            k for k in state_dict.keys()
            if 'transformer_blocks' in k and 'attn1.norm_q.weight' in k
        ]
        return len(layers)

    num_layers = count_num_layers()
    inner_dim, in_channels, patch_size1, patch_size2 = state_dict[
        'patch_embed.proj.weight'].shape
    model_config['in_channels'] = in_channels
    model_config['num_layers'] = num_layers
    out_channels = state_dict['proj_out.bias'].shape[0] // (
        patch_size1 * patch_size2)
    i2v_mode = in_channels // out_channels
    if inner_dim == 1920:
        base_num_layers = 30
        is_5b = False
    elif inner_dim == 3072:
        base_num_layers = 42
        model_config['num_attention_heads'] = 48
        model_config['use_rotary_positional_embeddings'] = True
        is_5b = True
    else:
        raise ValueError('unknown inner dim')

    if in_channels == 32:
        use_learned_positional_embeddings = ('patch_embed.pos_embedding'
                                             in state_dict)
        model_config[
            'use_learned_positional_embeddings'] = use_learned_positional_embeddings

    if 'ctrl_patch_embed.proj.weight' in state_dict:
        is_controlnet = True
        controlnet_proj = [
            k for k in state_dict.keys()
            if 'controlnet_proj' in k and 'weight' in k
        ][0]
        multi_proj_per_block = (len(controlnet_proj.split('.')) == 4)
        model_config['multi_proj_per_block'] = multi_proj_per_block
        model_config['ctrl_in_channels'] = state_dict[
            'ctrl_patch_embed.proj.weight'].shape[1]
        model_config['base_num_layers'] = base_num_layers
    else:
        is_controlnet = False
        assert num_layers == base_num_layers

    return model_config, is_controlnet, i2v_mode, is_5b

def hash_state_dict_keys(state_dict, with_shape=True):
    keys_str = convert_state_dict_keys_to_single_str(state_dict, with_shape=with_shape)
    keys_str = keys_str.encode(encoding="UTF-8")
    return hashlib.md5(keys_str).hexdigest()

def convert_state_dict_keys_to_single_str(state_dict, with_shape=True):
    keys = []
    for key, value in state_dict.items():
        if isinstance(key, str):
            if isinstance(value, torch.Tensor):
                if with_shape:
                    shape = "_".join(map(str, list(value.shape)))
                    keys.append(key + ":" + shape)
                keys.append(key)
            elif isinstance(value, dict):
                keys.append(key + "|" + convert_state_dict_keys_to_single_str(value, with_shape=with_shape))
    keys.sort()
    keys_str = ",".join(keys)
    return keys_str