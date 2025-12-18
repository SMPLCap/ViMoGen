import numpy as np
import os
import os.path as osp
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as DCP
from collections import OrderedDict
from easydict import EasyDict
from natsort import natsorted
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from time import time
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    get_state_dict,
    set_optimizer_state_dict,
    set_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful

from parallel.logging import TbTracker, get_logger, logging
from motion_rep.motion_checker import motion_vis


class TrainerBase:
    """base trainer with data parallel and tensor parallel functionalities."""

    def __init__(self, save_dir, log_level: str, rank: int = 0, mode='train') -> None:
        self.setup_save(save_dir, log_level, rank, mode=mode)
        self.rank = rank
        timer_names = [
            'data', 'vae', 'text_encoder', 'forward', 'backward', 'log',
            'ckpt', 'ema', 'image_encoder', 'controlnet'
        ]
        self.timer = EasyDict({k: Timer(k, self.logger) for k in timer_names})
        pass

    def save_config(self, config: DictConfig):
        out_path = os.path.join(self.save_dir, 'config.yaml')
        if self.rank == 0:
            OmegaConf.save(config, out_path)

    def setup_save(self, save_dir: str, log_level: str, rank: int = 0, mode='train'):
        log_level_str = log_level.upper()
        self.save_dir = save_dir
        self.ckpt_dir = osp.join(save_dir, 'checkpointing')
        os.makedirs(self.ckpt_dir, exist_ok=True)
        # self.model_weight_dir = osp.join(save_dir, 'weights')
        # os.makedirs(self.model_weight_dir, exist_ok=True)
        if mode == 'train':
            self.vis_dir = osp.join(save_dir, 'visualization')
        else:
            self.vis_dir = osp.join(save_dir, 'test_visualization')
        os.makedirs(self.vis_dir, exist_ok=True)
        self.logfile = osp.join(save_dir, 'train.log')
        if rank == 0:
            handler = logging.FileHandler(self.logfile, mode='a')
            handler.setLevel(log_level_str)
        if dist.is_initialized():
            dist.barrier()
        if rank != 0:
            handler = logging.FileHandler(self.logfile, mode='a')
            handler.setLevel(log_level_str)
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        handler.setFormatter(formatter)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level_str)
        console_handler.setFormatter(formatter)
        logger = get_logger(__name__, log_level=log_level_str)
        logger.logger.addHandler(handler)
        logger.logger.addHandler(console_handler)
        self.logger = logger
        try:
            self.tb_tracker = TbTracker(save_dir)
        except ValueError:
            pass

    # @staticmethod
    # def load_ckpt(model, optimizer=None, ckpt_dir:str=None):
    #     state_dict = AppState(model, optimizer).state_dict()
    #     DCP.load(state_dict, checkpoint_id=ckpt_dir)

    def get_resume_path_and_step(
        self,
        auto_resume,
        resume_path,
    ):
        if auto_resume or resume_path:
            if resume_path is None:
                existing_checkpoints = natsorted(os.listdir(self.ckpt_dir))
                if len(existing_checkpoints) == 0:
                    return None, 0
                resume_path = osp.join(self.ckpt_dir, existing_checkpoints[-1])
                resume_step = int(existing_checkpoints[-1])
                self.logger.info(
                    f'auto resume from {resume_path}', main_process_only=True)
            else:
                self.logger.info(
                    f'resume from {resume_path}', main_process_only=True)
                resume_step = int(os.path.basename(resume_path))
            return resume_path, resume_step
        else:
            return None, 0

    @staticmethod
    def load_model(rank, model, path, lora=False, strict=True, logger=None):
        if os.path.isdir(path):
            # dcp
            state_dict = get_model_state_dict(
                model,
                options=StateDictOptions(
                    full_state_dict=False,
                    cpu_offload=True,
                ),
            )
            DCP.load(state_dict=state_dict, checkpoint_id=path)
            if rank == 0 and logger is not None:
                logger.info(f' Loaded Model from {path}: ')
        elif rank == 0:
            state_dict = torch.load(path, map_location='cpu')
            if lora:
                model_dict = model.state_dict()
                model_dict.update(state_dict)
                if logger is not None:
                    logger.info("loaded parameters from LoRA.")
            missing_keys, unexpected_keys = model.load_state_dict(
                state_dict, strict=strict)
            if logger is not None:
                logger.info(f' Loaded Model from {path}: '
                            f' Missing keys: {missing_keys} '
                            f' Unexpected keys: {unexpected_keys} ')
            
        if dist.is_initialized():
            dist.barrier()

    @staticmethod
    def load_optimizer(model, optimizer, path):
        if os.path.isdir(path):
            # dcp
            state_dict = get_optimizer_state_dict(
                model,
                optimizer,
                options=StateDictOptions(
                    full_state_dict=False,
                    cpu_offload=True,
                ),
            )
            DCP.load(state_dict=state_dict, checkpoint_id=path)
        else:
            state_dict = torch.load(path, map_location='cpu')
            set_optimizer_state_dict(
                model,
                optimizer,
                optim_state_dict=state_dict,
                options=StateDictOptions(full_state_dict=False, strict=True),
            )
        if dist.is_initialized():
            dist.barrier()

    @staticmethod
    def save_model(rank, model, out_path: str, dcp=False, lora=False):

        # if lora:
        #     state_dict = {
        #         name: param
        #         for name, param in model.named_parameters()
        #         if "lora" in name
        #     }
        # else:
        state_dict = get_model_state_dict(
        model,
        options=StateDictOptions(
            full_state_dict=not dcp,
            cpu_offload=True,
        ),
        )
        if lora:
            state_dict = {key: value for key, value in state_dict.items() if 'lora' in key.lower()}

        if dcp:
            DCP.save(state_dict, checkpoint_id=out_path)
        elif rank == 0:
            torch.save(state_dict, out_path)
        del state_dict
        if dist.is_initialized():
            dist.barrier()
        return

    @staticmethod
    def save_optimizer(model, opt, out_path: str, dcp=False):
        state_dict = get_optimizer_state_dict(
            model,
            opt,
            options=StateDictOptions(
                full_state_dict=False,
                cpu_offload=True,
            ),
        )
        if dcp:
            DCP.save(state_dict, checkpoint_id=out_path)
        else:
            torch.save(state_dict, out_path)
        del state_dict
        if dist.is_initialized():
            dist.barrier()
        return

    @staticmethod
    def load_ckpt_from_dir(rank,
                           load_save_dict: dict = None,
                           model_for_opt: tuple = None,
                           optimizer=None,
                           ckpt_dir: str = None,
                           load_optimizer_only=False,
                           lora=False):
        if not load_optimizer_only:
            if load_save_dict is not None:
                for model_name, model in load_save_dict.items():
                    file_path = os.path.join(ckpt_dir, f'{model_name}.pt')
                    folder_path = os.path.join(ckpt_dir, model_name)
                    load_path = file_path if os.path.isfile(
                        file_path) else folder_path
                    TrainerBase.load_model(
                        rank, model, load_path, lora, strict=False, logger=None)
        if optimizer is not None:
            for model_name in model_for_opt:
                file_path = os.path.join(ckpt_dir,
                                         f'{model_name}-opt-{rank:02d}.pt')
                folder_path = os.path.join(ckpt_dir, f'{model_name}-opt')
                load_path = file_path if os.path.isfile(
                    file_path) else folder_path
                TrainerBase.load_optimizer(load_save_dict[model_name],
                                           optimizer, load_path)
        return None

    def load_ckpt(self,
                  rank,
                  load_save_dict: dict = None,
                  model_for_opt: tuple = None,
                  optimizer=None,
                  global_step: int = 0,
                  load_optimizer_only=False,
                  lora=False):
        ckpt_dir = os.path.join(self.ckpt_dir, f'{global_step:08d}')
        return self.load_ckpt_from_dir(rank, load_save_dict, model_for_opt,
                                       optimizer, ckpt_dir,
                                       load_optimizer_only,
                                       lora)

    @staticmethod
    def save_ckpt_to_dir(rank,
                         load_save_dict=None,
                         model_for_opt: tuple = None,
                         optimizer=None,
                         ckpt_dir: str = None,
                         dcp=False,
                         lora=False):
        os.makedirs(ckpt_dir, exist_ok=True)
        if load_save_dict is not None:
            for model_name, model in load_save_dict.items():
                if not dcp:
                    out_path = os.path.join(ckpt_dir, f'{model_name}.pt')
                else:
                    out_path = os.path.join(ckpt_dir, model_name)
                TrainerBase.save_model(rank, model, out_path, dcp, lora)
        if optimizer is not None:
            for model_name in model_for_opt:
                out_path = os.path.join(ckpt_dir, f'{model_name}-opt')
            TrainerBase.save_optimizer(
                load_save_dict[model_name], optimizer, out_path, dcp=True)
        return None

    def save_ckpt(self,
                  rank,
                  load_save_dict=None,
                  model_for_opt: tuple = None,
                  optimizer=None,
                  global_step=None,
                  dcp=False,
                  lora=False,
                  remove_old_ckpts=True):
        if remove_old_ckpts:
            os.system(f'rm -rf {self.ckpt_dir}/*')
        ckpt_dir = os.path.join(self.ckpt_dir, f'{global_step:08d}')
        return self.save_ckpt_to_dir(rank, load_save_dict, model_for_opt,
                                     optimizer, ckpt_dir, dcp, lora)

    def save_motion_dict(self, motion_dict, mean, std, device='cuda:0', vis=True, result_folder=None):
        if result_folder is None:
            result_folder = self.vis_dir
        for file_name, motion in motion_dict.items():
            motion = motion * std + mean
            motion_save_path = os.path.join(result_folder, file_name)
            motion_save_folder = os.path.dirname(motion_save_path)
            os.makedirs(motion_save_folder, exist_ok=True)
            torch.save(motion[0].cpu(), motion_save_path)
            motion_dim = motion.shape[-1]
            if vis:
                if motion_dim == 276:
                    motion_vis(motion_save_path, motion_save_folder, H=384, W=384, batch_size=24, fps=20, device=device, recover_from_velocity=True)
                else:
                    self.logger.info(f'unsupported motion dimension {motion_dim}')
    
    def save_txt_dict(self, txt_dict, result_folder=None):
        if result_folder is None:
            result_folder = self.vis_dir
        for file_name, txt in txt_dict.items():
            txt_save_path = os.path.join(result_folder, file_name)
            os.makedirs(os.path.dirname(txt_save_path), exist_ok=True)
            with open(txt_save_path, 'w') as f:
                f.write(txt)
            
class AppState(Stateful):
    """This is a useful wrapper for checkpointing the Application State. Since
    this object is compliant with the Stateful protocol, DCP will automatically
    call state_dict/load_stat_dict as needed in the dcp.save/load APIs.

    Note: We take advantage of this wrapper to handle calling distributed state dict methods on the model
    and optimizer.
    """

    def __init__(self, model, optimizer=None):
        self.model = model
        self.optimizer = optimizer

    def state_dict(self):
        # this line automatically manages FSDP FQN's, as well as sets the default state dict type to FSDP.SHARDED_STATE_DICT
        model_state_dict, optimizer_state_dict = get_state_dict(
            self.model, self.optimizer)
        state_dict = {
            'model': model_state_dict,
            'optim': optimizer_state_dict,
        }
        if self.ema is not None:
            ema_state_dict, _ = get_state_dict(self.ema, self.optimizer)
            state_dict['ema'] = ema_state_dict
        return state_dict

    def load_state_dict(self, state_dict):
        # sets our state dicts on the model and optimizer, now that we've loaded
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict['model'],
            optim_state_dict=state_dict['optim'],
        )
        if self.ema is not None:
            set_state_dict(
                self.ema,
                model_state_dict=state_dict['ema'],
            )


class Timer:

    def __init__(
        self,
        name,
        logger=None,
    ):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.logger = logger

    @property
    def elapsed_time(self):
        return self.end_time - self.start_time

    def __enter__(self):
        torch.cuda.synchronize()
        self.start_time = time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.synchronize()
        self.end_time = time()
        msg = f'Elapsed time for {self.name}: {self.elapsed_time:.3f} s'
        if self.logger is not None:
            self.logger.info(msg, main_process_only=True)
        else:
            print(msg)


# https://github.com/hpcaitech/Open-Sora/blob/main/opensora/utils/train_utils.py#L44
@torch.no_grad()
def update_ema(
    ema_model: torch.nn.Module,
    model: torch.nn.Module,
    decay: float = 0.9999,
    sharded: bool = True,
) -> None:
    """Step the EMA model towards the current model."""
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        if name == 'pos_embed':
            continue
        if not param.requires_grad:
            continue
        param_data = param.data
        # assert param_data.dtype == torch.float32
        # TODO get float32 version of parameters from optimizer
        ema_params[name].mul_(decay).add_(
            param_data.to(torch.float32), alpha=1 - decay)


def linear_lr_warmpup(warmup_steps):
    assert warmup_steps > 0

    def lr_lambda(current_step):
        if current_step > warmup_steps:
            return 1.0
        else:
            return current_step / warmup_steps

    return lr_lambda
