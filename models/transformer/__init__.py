import os
import torch

from .utils import load_safetensors
from copy import deepcopy
from .wan.modules import WanModelT2M, WanModelTM2M

def get_transformer3d(
    model_name: str,
    load_pretrain: bool,
    patch_size=2,
    in_channel=16,
    base_repo=None,
    strict: bool = True,
    model_kwargs=None,
):
    if model_kwargs is not None:
        dense_interval = model_kwargs.get('dense_interval', 1)
        rope_mode = model_kwargs.get('rope_mode', 'naive')
        force_no_sincos_embed = model_kwargs.get('force_no_sincos_embed', True)
        load_path = model_kwargs.get('load_path', None)

    if 'wanvideotm2m' in model_name.lower():
        import json
        with open(f'{base_repo}/config.json') as f:
            config = json.load(f)
        # remove 'class_name' and '_diffusers_version' keys
        config.pop('_class_name', None)
        config.pop('_diffusers_version', None)
        config['in_dim'] = in_channel
        config['out_dim'] = in_channel
        model = WanModelTM2M(**config, **model_kwargs)
        if load_pretrain or load_path is not None:
            print(f'Loading pretrain from {base_repo}')
            state_dict_base = load_safetensors(base_repo)
            if load_path is not None:
                print(f'Loading pretrain from {load_path}')
                state_dict = torch.load(load_path, map_location='cpu')
            else:
                state_dict = deepcopy(state_dict_base)
            import torch.distributed as dist
            if dist.is_initialized():
                if dist.get_rank() == 0:
                    model.load_state_dict(state_dict, strict=strict)
                dist.barrier()
            else:
                model.load_state_dict(state_dict, strict=strict)
        return model
    
    elif 'wanvideot2m' in model_name.lower():
        import json
        with open(f'{base_repo}/config.json') as f:
            config = json.load(f)
        # remove 'class_name' and '_diffusers_version' keys
        config.pop('_class_name', None)
        config.pop('_diffusers_version', None)
        config['in_dim'] = in_channel
        config['out_dim'] = in_channel
        model = WanModelT2M(**config, **model_kwargs)
        if load_pretrain:
            if load_path is not None:
                print(f'Loading pretrain from {load_path}')
                state_dict = torch.load(load_path, map_location='cpu')
            else:
                print(f'Loading pretrain from {base_repo}')
                state_dict = load_safetensors(base_repo)
            import torch.distributed as dist
            if dist.is_initialized():
                if dist.get_rank() == 0:
                    model.load_state_dict(state_dict, strict=strict)
                dist.barrier()
            else:
                model.load_state_dict(state_dict, strict=strict)
        return model
    
    else:
        raise NotImplementedError(f"Model {model_name} not implemented.")
