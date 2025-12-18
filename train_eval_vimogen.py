# yapf: disable
import argparse
import contextlib
import numpy as np
import os
import torch
import torch.distributed as dist
import torch.nn as nn
from copy import deepcopy
from functools import partial
from omegaconf import OmegaConf
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from tqdm import tqdm

from datasets.dataloader import get_dataloader
from models.transformer import get_transformer3d
from models.transformer.utils import (
    count_trainable_parameters,
    randn_tensor,
)
from parallel.parallel import fsdp_transformer_ulysses
from parallel.utils import get_device_mesh
from trainer import (
    TrainerBase,
    linear_lr_warmpup,
    update_ema,
)
from trainer.scheduler import TimestepSamplerMP, FlowMatchScheduler
from utils import maybe_corrupt_ref_motion, smooth_motion_rep

# yapf: enable

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def count_model_parameters(model: nn.Module):
    params = sum([np.prod(p.size()) for p in model.parameters()])
    return params


def sample_data(loader, sampler, start_epoch, start_iter):
    epoch = start_epoch
    while True:
        sampler.set_epoch(epoch)
        begin_iter = start_iter if epoch == start_epoch else 0
        epoch += 1
        for _, batch in enumerate(loader, start=begin_iter):
            yield batch


def main(args):
    is_training = args.mode == 'train'

    train_target = args.experiment.get('train_target', ['transformer'])
    train_transformer = 'transformer' in train_target
    dist.init_process_group('nccl')
    global_rank = dist.get_rank()
    is_main_process = global_rank == 0
    device = torch.device(global_rank % torch.cuda.device_count())
    torch.cuda.set_device(device)

    device_mesh_dp_hybrid = get_device_mesh(use_hybrid=True, tp_size=None)# ?
    device_mesh_dp_tp = get_device_mesh(
        use_hybrid=False, tp_size=args.parallel.tp_size)
    global_device_mesh = get_device_mesh(use_hybrid=False, tp_size=None)
    dp_mesh = device_mesh_dp_tp['dp']
    world_size = dist.get_world_size()

    dp_rank = dp_mesh.get_local_rank()
    dropout_generator = torch.Generator(device)
    dropout_generator.manual_seed(dp_rank + int(args.experiment.global_seed))
    global_dp_rank = global_device_mesh.get_local_rank()
    loglevel = args.experiment.get('loglevel', 'INFO').upper()
    trainer = TrainerBase(
        args.experiment.result_dir, log_level=loglevel, rank=global_rank, mode=args.mode)
    if global_rank == 0:
        trainer.save_config(args)
    dist.barrier()
    logger, tb_tracker, timer = trainer.logger, trainer.tb_tracker, trainer.timer
    
    result_folder = os.path.join(trainer.vis_dir, args.mbench_name)
    os.makedirs(result_folder, exist_ok=True)
    logger.info(f'result_folder: {result_folder}')
    logger.info(f'text_key: {args.dataset.text_key}')
    
    dtype_mapping = dict(
        bf16=torch.bfloat16, fp32=torch.float32, fp16=torch.float16)
    dtype = dtype_mapping[args.precision.mixed_precision]
    grad_dtype = dtype_mapping[args.precision.grad_precision]

    logger.info(
        f'dtype {dtype}, grad_dtype {grad_dtype}'
    )
    dist.barrier()

    ref_corruption_cfg = args.get('ref_motion_corruption', {})
    train_ref_corruption_cfg = None
    if ref_corruption_cfg.get('enable', False):
        train_ref_corruption_cfg = ref_corruption_cfg

    base_repo_path = args.model_path[args.experiment.model_name]

    resume_path, resume_step = trainer.get_resume_path_and_step(
        auto_resume=args.experiment.auto_resume,
        resume_path=args.experiment.resume_path)

    patch_size = 2
    in_channel = args.model.get('in_channels', 16)
    model = get_transformer3d(
        model_name=args.experiment.model_name,
        load_pretrain=args.experiment.load_pretrain,
        patch_size=patch_size,
        in_channel=in_channel,
        base_repo=base_repo_path,
        strict=False,
        model_kwargs=args.get(
            'model',
            dict(
                force_no_sincos_embed=True, rope_mode='naive',
                load_path=None)))
    model = model.to(device=device, dtype=dtype)

    if train_transformer:
        ema = deepcopy(model)
    
    logger.debug(
        f'rank {global_rank:02d} original transformer parameters: {count_model_parameters(model)}',
        main_process_only=False,
    )
    load_save_dict = {}
    model_for_opt = []
    if train_transformer:
        load_save_dict['model'] = model
        load_save_dict['ema'] = ema
        model_for_opt.append('model')

    if resume_path is not None:
        trainer.load_ckpt(
            global_dp_rank,
            load_save_dict,
            model_for_opt=None,
            optimizer=None,
            global_step=resume_step)  # load optimizer after sharding
        logger.info(
            f'resume from {resume_path}, resume_step {resume_step}',
            main_process_only=True,
        )
    if train_transformer:
        dp_strategy = 'op_grad'
        transformer_device_mesh = global_device_mesh
    else:
        dp_strategy = 'hybrid'
        transformer_device_mesh = device_mesh_dp_hybrid

    transformer_fsdp_func = partial(
        fsdp_transformer_ulysses,
        device_mesh=device_mesh_dp_tp,
        global_device_mesh=transformer_device_mesh,
        dtype=dtype,
        grad_dtype=grad_dtype,
        strategy=dp_strategy,
    )
    model = transformer_fsdp_func(model=model)
    if train_transformer:
        ema = transformer_fsdp_func(model=ema)
        ema.requires_grad_(False)
        ema.eval()
        load_save_dict['ema'] = ema
        load_save_dict['model'] = model

    logger.debug(
        f'rank {global_rank:02d} transformer parameters after sharding: {count_model_parameters(model)}',
        main_process_only=False,
    )

    def get_opt_params():
        
        trainable_modules = args.experiment.get('trainable_modules', None)

        # First, set all parameters to not require gradients
        for param in model.parameters():
            param.requires_grad = False
        
        # If specific modules are provided, enable gradients for those
        if trainable_modules:
            # Enable gradients for specified modules
            for name, module in model.named_parameters():
                if any(m in name for m in trainable_modules):
                    logger.info(f'Enabling gradients for {name}')
                    module.requires_grad = True
        else:
            # Enable gradients for all parameters
            for param in model.parameters():
                param.requires_grad = True
        
        # Filter and return only trainable parameters
        params_to_optimize = [p for p in model.parameters() if p.requires_grad]
        logger.info(f'Trainable parameters: {sum([p.numel() for p in params_to_optimize])}')
        return params_to_optimize
    
    if is_training:
        opt = torch.optim.AdamW(
            get_opt_params(),
            lr=args.solver.lr,
            betas=tuple(args.solver.betas),
            weight_decay=args.solver.weight_decay,
        )
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=opt,
            lr_lambda=linear_lr_warmpup(args.solver.warmup_steps))
        scaler = ShardedGradScaler()

        if resume_path is not None:
            trainer.load_ckpt(
                global_dp_rank,
                load_save_dict=load_save_dict,
                model_for_opt=model_for_opt,
                optimizer=None,
                global_step=resume_step,
                load_optimizer_only=True,
            )
            logger.info(
                f'Optimizer Loaded resume from {resume_path}, resume_step {resume_step}',
                main_process_only=True,
            )
    else:
        opt, lr_scheduler = None, None    
    
    
    def to_train_mode():
        model.train(
        )  # NOTE even when train_target == ['controlnet'] the transformer should be on train mode for training

    def to_eval_mode():
        model.eval()
    
    wan_scheduler = FlowMatchScheduler()

    # NOTE to keep the same data within SP
    seed = (args.experiment.global_seed * world_size + dp_rank)

    logger.info(f'seed is {seed}')
    torch.manual_seed(seed)

    bucket_config_type = args.experiment.get('bucket_config_type', None)
    if bucket_config_type is not None:
        data_seed = args.experiment.global_seed
    else:
        data_seed = seed

    if is_training:
        dataloader, sampler = get_dataloader(
            local_batch=args.dataloader.local_batch,
            dp_mesh=dp_mesh,
            dataset_args=args.dataset,
            seed=data_seed,
            num_workers=args.dataloader.num_workers,
            bucket_config_type=bucket_config_type,
            dataset_name=args.experiment.dataset_name, 
            is_test=False)
    test_dataloader, test_sampler = get_dataloader(
        local_batch=args.dataloader.test_local_batch,
        dp_mesh=dp_mesh,
        dataset_args=args.dataset,
        seed=data_seed,
        num_workers=args.dataloader.num_workers,
        bucket_config_type=bucket_config_type,
        dataset_name='MBenchWiRefMotion', 
        is_test=True)

    if args.dataloader.global_batch is not None:
        accumulate_times = args.dataloader.global_batch // (
            args.dataloader.local_batch * dp_mesh.size())
    else:
        accumulate_times = 1
    torch.cuda.empty_cache()

    @torch.no_grad()
    def generate_pipe(
        model,
        prompt_emb,
        prompt_emb_null,
        latents,
        latents_mask,
        ref_latents,
        ref_latents_mask,
        num_inference_steps: int = 50,
        cfg_scale: float = 5.0,
        use_ema: bool = False,
        device: torch.device = torch.device('cuda'),
        dtype: torch.dtype = torch.bfloat16,
        scheduler: FlowMatchScheduler = None,
        seed: int = None,
        logger=None,
        condition_on_text: bool = False,
        attend_to_text_mask: torch.Tensor | None = None,
    ):
        """Generate predictions during validation with Advanced Classifier-Free Guidance."""
        to_eval_mode()
        generator = torch.Generator(device).manual_seed(seed if seed is not None else torch.randint(0, 1000000, (1,)).item())
        
        # Use EMA model if specified
        inf_model = ema if use_ema else model
        
        # Prepare noise and initial latents
        noise = randn_tensor(
            logger,
            latents.shape,
            generator=generator,
            device=device,
            dtype=dtype
        )
        
        # Set up scheduler for inference
        scheduler.set_timesteps(num_inference_steps, training=False, denoising_strength=0.7)
        timesteps = scheduler.timesteps.to(device)
        xt = noise  # Start with pure noise for generation
        
        # Pad prompt_emb_null to the same length as prompt_emb  # [B, L, C]
        # prompt_emb_null: [B, L1, C], prompt_emb: [B, L2, C]
        if prompt_emb_null.size(1) < prompt_emb.size(1):
            prompt_emb_zeros = torch.zeros(prompt_emb.size(0), prompt_emb.size(1) - prompt_emb_null.size(1), prompt_emb.size(2), device=prompt_emb.device, dtype=prompt_emb.dtype)
            prompt_emb_null = torch.cat([prompt_emb_null, prompt_emb_zeros], dim=1)

        # Denoising loop with Advanced CFG
        latents_mask_input = torch.cat([latents_mask] * 2, dim=0)            
        ref_latents_null = torch.zeros_like(ref_latents)
        ref_latents_input = torch.cat([ref_latents, ref_latents_null], dim=0)
        ref_latents_mask_input = torch.cat([ref_latents_mask] * 2, dim=0)
        attend_to_text_mask_input = None
        if attend_to_text_mask is not None:
            attend_to_text_mask_input = torch.cat([attend_to_text_mask] * 2, dim=0)

        # Contexts
        context_input = torch.cat([
            prompt_emb,           # Conditional
            prompt_emb_null,      # Unconditional
        ], dim=0)
        for t in tqdm(timesteps, desc="Validation Generation", disable=logger is None):
            with torch.amp.autocast(dtype=torch.bfloat16, device_type=device.type):
                # Prepare inputs for three branches:
                # 1. Conditional with ref_latents
                # 2. Unconditional with ref_latents
                batch_size = xt.size(0)
                latent_model_input = torch.cat([xt] * 2, dim=0)
                
                # Masks
                timestep_input = t.unsqueeze(0)
                
                # Compute noise predictions
                noise_pred = inf_model(
                    x=latent_model_input,
                    timestep=timestep_input,
                    context=context_input,
                    x_mask=latents_mask_input,
                    ref_motion=ref_latents_input,
                    ref_motion_mask=ref_latents_mask_input,
                    use_gradient_checkpointing=False,
                    attend_to_text_mask=attend_to_text_mask_input,
                )
                
                # Split predictions into three branches
                noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                
                # Compute CFG
                if condition_on_text:
                    noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
                else:
                    noise_pred = noise_pred_cond
    
                # Scheduler step
                xt = scheduler.step(noise_pred, t, xt)

        latents_pred = xt

        # Compute inference loss
        loss = torch.nn.functional.mse_loss(latents_pred.float(), latents.float(), reduction='none').mean(dim=-1)
        loss = loss * latents_mask
        loss = loss.sum(-1) / latents_mask.sum(-1)  # [B]
        loss = loss.mean()
        logger.info(f'Validation loss: {loss.item()}', main_process_only=True)

        # smooth the motion,   # [B, T, C]
        batch_size = xt.shape[0]
        for i in range(batch_size):
            xt[i] = smooth_motion_rep(xt[i], kernel_size=5, sigma=1.0)
        
        xt = xt.to(dtype=dtype)
        to_train_mode()
        return xt

    avg_loss_dict = {'loss': 0, 'loss_text': 0, 'loss_ref_motion': 0}

    if is_training:
        dataloader = sample_data(
            dataloader,
            sampler,
            start_epoch=resume_step // len(dataloader),
            start_iter=resume_step % len(dataloader),
        )

    test_dataloader_len = len(test_dataloader)
    test_dataloader = sample_data(
        test_dataloader,
        test_sampler,
        start_epoch=resume_step // len(test_dataloader),
        start_iter=resume_step % len(test_dataloader),
    )

    if not is_training:
        args.experiment.max_steps = resume_step + 1000
        
    pbar = tqdm(
        range(resume_step, args.experiment.max_steps),
        disable=not is_main_process,
        initial=resume_step,
    )

    to_train_mode()
    total_trainable, total_untrainable = count_trainable_parameters(
                model.named_parameters())
    logger.info(
        f'Total trainable parameters {total_trainable} \n Total untrainable parameters {total_untrainable}',
        main_process_only=True,
    )

    for global_step in pbar:
        step_plus = global_step + 1
        if is_training:
            with timer.data:
                batch = next(dataloader)
                latents = batch.pop('motion').to(device=device, dtype=dtype)  # [B, T, C]
                latents_mask = batch.pop('motion_mask').to(device=device, dtype=dtype)    # [B, T]
                prompt_emb = batch.pop('prompt_emb').to(device=device, dtype=dtype)   # [B, L, C]
                motion_mean = batch.pop('motion_mean').to(device=device)
                motion_std = batch.pop('motion_std').to(device=device)   # []       
                ref_latents = batch.pop('ref_motion').to(device=device, dtype=dtype)  # [B, T, C]
                ref_latents_mask = batch.pop('ref_motion_mask').to(device=device, dtype=dtype)  # [B, T]
                motion_dim_mask = batch.pop('motion_dim_mask').to(device=device)    # [B, C]
                attend_to_text_mask = batch.pop('attend_to_text_mask').to(device=device)  # [B]

            if train_ref_corruption_cfg is not None:
                attend_to_ref = ~attend_to_text_mask.bool()
                if attend_to_ref.any():
                    corrupted_latents, corrupted_mask = maybe_corrupt_ref_motion(
                        ref_latents, ref_latents_mask, train_ref_corruption_cfg, is_test=False)
                    corrupted_latents = corrupted_latents.to(device=device, dtype=dtype)
                    corrupted_mask = corrupted_mask.to(device=device, dtype=dtype)
                    ref_latents[attend_to_ref] = corrupted_latents[attend_to_ref]
                    ref_latents_mask[attend_to_ref] = corrupted_mask[attend_to_ref]

            wan_scheduler.set_timesteps(1000, training=True)
            noise = torch.randn_like(latents)
            timestep_ids = torch.randint(0, wan_scheduler.num_train_timesteps, (latents.shape[0],))
            timesteps = wan_scheduler.timesteps[timestep_ids].to(device=device, dtype=dtype)
            noisy_latents = wan_scheduler.add_noise(latents, noise, timesteps).to(dtype)
            training_target = wan_scheduler.training_target(latents, noise, timesteps)

            with timer.forward:
                with torch.amp.autocast(dtype=torch.bfloat16, device_type=device.type):
                    noise_pred = model(x=noisy_latents, timestep=timesteps, context=prompt_emb, x_mask=latents_mask, ref_motion=ref_latents, 
                    ref_motion_mask=ref_latents_mask, use_gradient_checkpointing=True)

                    loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float(), reduction='none') # [B, T, C]

                    # only compute loss for the unmasked channels and unmasked latents
                    motion_dim_mask = motion_dim_mask.unsqueeze(1).repeat(1, latents.shape[1], 1)  # [B, T, C]
                    latents_mask_expand = latents_mask.unsqueeze(-1).expand(-1, -1, latents.shape[-1])  # [B, T, C]
                    scheduler_weight = wan_scheduler.training_weight(timesteps) # [B]
                    scheduler_weight_expand = scheduler_weight.unsqueeze(-1).unsqueeze(-1).expand(-1, latents.shape[1], latents.shape[-1])  # [B, T, C] 
                    loss_mask = motion_dim_mask * latents_mask_expand   # [B, T, C]
                    channel_weights = torch.ones(loss.shape[-1], device=latents.device, dtype=latents.dtype).view(1, 1, -1)
                    channel_weights[:, :, 258:] = 3.0  # upweight the global motion channels
                    loss = loss * loss_mask * scheduler_weight_expand * channel_weights   # [B, T, C]
                    
                    # get loss_text and loss_ref_motion based on the attend_to_text_mask
                    loss_text = loss[attend_to_text_mask==1]
                    loss_ref_motion = loss[attend_to_text_mask==0]
                    loss_mask_text = loss_mask[attend_to_text_mask==1]
                    loss_mask_ref_motion = loss_mask[attend_to_text_mask==0]

                    # compute the mean loss for the non-zero values
                    loss_text_mean = loss_text.reshape(-1).sum() / loss_mask_text.reshape(-1).sum()
                    loss_ref_motion_mean = loss_ref_motion.reshape(-1).sum() / loss_mask_ref_motion.reshape(-1).sum()
                    loss_mean = loss.reshape(-1).sum() / loss_mask.reshape(-1).sum()
                    avg_loss_dict['loss'] += loss_mean.item()
                    if loss_mask_text.reshape(-1).sum() != 0:
                        avg_loss_dict['loss_text'] += loss_text_mean.item()
                    if loss_mask_ref_motion.reshape(-1).sum() != 0:
                        avg_loss_dict['loss_ref_motion'] += loss_ref_motion_mean.item()

                    loss = loss_mean
                    
            no_sync = step_plus % accumulate_times != 0 and dp_strategy == 'op_grad'
            with model.no_sync() if no_sync else contextlib.nullcontext():
                with timer.backward:
                    if dtype == torch.float16:
                        scaler.scale(loss).backward()
                    elif dtype == torch.bfloat16 or dtype == torch.float32:
                        loss.backward()
            if step_plus % accumulate_times == 0:
                if dtype == torch.float16:
                    scaler.unscale_(opt)
                model.clip_grad_norm_(args.solver.grad_clip)
                if dtype == torch.float16:
                    scaler.step(opt)
                    scaler.update()
                elif dtype == torch.bfloat16 or dtype == torch.float32:
                    opt.step()
                lr_scheduler.step()
                opt.zero_grad()

            if step_plus % args.experiment.log_every == 0:
                with timer.log:
                    loss_str = ''
                    for avg_loss_key, avg_loss in avg_loss_dict.items():
                        avg_loss = torch.tensor(
                            [avg_loss],
                            device=device) / args.experiment.log_every
                        dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                        avg_loss = avg_loss.item() / world_size
                        loss_str += f'step: {step_plus}, {avg_loss_key}: {avg_loss} '
                        tb_tracker.add_scalar(
                            tag=f'train/{avg_loss_key}',
                            scalar_value=avg_loss,
                            global_step=step_plus)
                        avg_loss_dict[avg_loss_key] = 0
                    tb_tracker.add_scalar(
                        tag='train/lr',
                        scalar_value=opt.param_groups[0]['lr'],
                        global_step=step_plus,
                    )
                    logger.info(
                        loss_str + f'rank {global_rank:02d} '
                        f'Peak Mem: {torch.cuda.max_memory_allocated() / 1024 / 1024:,.2f} MiB ',
                        main_process_only=True,
                    )

            if step_plus % args.experiment.checkpoint_every == 0:
                with timer.ckpt:
                    trainer.save_ckpt(
                        global_dp_rank,
                        load_save_dict,
                        model_for_opt=model_for_opt,
                        optimizer=None,
                        global_step=step_plus,
                        dcp=False)
            if step_plus % args.experiment.ema_every == 0:
                with timer.ema:
                    if train_transformer:
                        update_ema(ema, model, decay=args.experiment.ema_decay)

        # Add validation after logging and EMA updates
        if (not is_training) or ((step_plus % args.experiment.visualize_every)== 0):
            for test_batch_idx in range(test_dataloader_len):
                with timer.data:
                    batch = next(test_dataloader)
                    latents = batch.pop('motion').to(device=device, dtype=dtype)  # [B, T, C]
                    latents_mask = batch.pop('motion_mask').to(device=device, dtype=dtype)    # [B, T]
                    prompt_emb = batch.pop('prompt_emb').to(device=device, dtype=dtype)   # [B, L, C]
                    prompt_emb_null = batch.pop('prompt_emb_null').to(device=device, dtype=dtype)   # [B, L, C]
                    text = batch.pop('text')
                    motion_mean = batch.pop('motion_mean').to(device=device)
                    motion_std = batch.pop('motion_std').to(device=device)
                    motion_dim_mask = batch.pop('motion_dim_mask').to(device=device)    # [B, C]
                    attend_to_text_mask = batch.pop('attend_to_text_mask').to(device=device)
                    ref_latents_original = batch.pop('ref_motion_original').to(device=device, dtype=dtype)  # [B, T, C]
                    ref_latents = batch.pop('ref_motion').to(device=device, dtype=dtype)  # [B, T, C]
                    ref_latents_mask = batch.pop('ref_motion_mask').to(device=device, dtype=dtype)  # [B, T]

                torch.cuda.empty_cache()
                logger.info(
                    f'step: {step_plus}, generating validation samples',
                    main_process_only=True)

                ref_latents_visual = ref_latents.clone()
                ref_latents_visual_mask = ref_latents_mask.clone()

                attend_to_text_mask_bool = attend_to_text_mask.bool()
                text_mask = attend_to_text_mask_bool
                motion_mask = ~attend_to_text_mask_bool
                condition_names = ['text' if flag else 'motion' for flag in attend_to_text_mask_bool.tolist()]

                gen_latents_full = torch.zeros_like(latents)
                for condition_name, sample_mask in (('text', text_mask), ('motion', motion_mask)):
                    if not sample_mask.any().item():
                        continue
                    condition_ref_latents = (torch.zeros_like(ref_latents_visual[sample_mask])
                                             if condition_name == 'text' else ref_latents_visual[sample_mask])
                    condition_gen_latents = generate_pipe(
                        model=model,
                        prompt_emb=prompt_emb[sample_mask],
                        prompt_emb_null=prompt_emb_null[sample_mask],
                        latents=latents[sample_mask],
                        latents_mask=latents_mask[sample_mask],
                        ref_latents=condition_ref_latents,
                        ref_latents_mask=ref_latents_visual_mask[sample_mask],
                        num_inference_steps=args.experiment.get('validation_steps', 50),
                        cfg_scale=args.experiment.get('cfg_scale', 5.0),
                        use_ema=False,
                        device=device,
                        dtype=dtype,
                        scheduler=wan_scheduler,
                        seed=seed,
                        logger=logger,
                        condition_on_text=(condition_name == 'text'),
                        attend_to_text_mask=attend_to_text_mask_bool[sample_mask],
                    )
                    gen_latents_full[sample_mask] = condition_gen_latents.to(gen_latents_full.dtype)
                
                # Visualization
                motion_dict, txt_dict = {}, {}
                # for batch_idx in tqdm(range(1), desc="Saving Visualization Data", disable=logger is None):
                # vis_num = 5 if is_training else gen_latents.shape[0]
                gen_latents = gen_latents_full
                vis_num = gen_latents.shape[0]
                if vis_num < gen_latents.shape[0]:
                    vis_idx = torch.randint(0, gen_latents.shape[0], (vis_num,))
                else:
                    vis_idx = torch.arange(gen_latents.shape[0])
                for batch_idx in tqdm(vis_idx.tolist(), desc="Saving Visualization Data", disable=logger is None):
                    test_sample_id = batch.get('test_sample_id')[batch_idx]
                    txt_dict[f'step{step_plus:08d}/{test_sample_id}/prompt.txt'] = text[batch_idx]
                    latents_mask_ = latents_mask[batch_idx].bool()  # [T]
                    ref_latents_mask_ = ref_latents_mask[batch_idx].bool()  # [T]
                    ref_latents_visual_mask_ = ref_latents_visual_mask[batch_idx].bool()
                    condition_name = condition_names[batch_idx]
                    if torch.any(latents_mask_):
                        motion_dict[f'step{step_plus:08d}/{test_sample_id}/motion_gen_condition_on_{condition_name}.pt'] = gen_latents[batch_idx:batch_idx+1, latents_mask_]
                        if condition_name == 'motion':
                            motion_dict[f'step{step_plus:08d}/{test_sample_id}/motion_ref.pt'] = ref_latents_original[batch_idx:batch_idx+1, ref_latents_mask_]

                trainer.save_motion_dict(motion_dict, mean=motion_mean[0:1], std=motion_std[0:1], device=device, result_folder=result_folder)
                trainer.save_txt_dict(txt_dict, result_folder=result_folder)
                logger.info(f'Saved visualization data for step {step_plus}', main_process_only=True)

                torch.cuda.empty_cache()    
                
            if not is_training:
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ViMoGen Training and Evaluation Script')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/tm2m_train',
        help='config file',
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='train',
        help='choose is training or evaluating')
    parser.add_argument(
        '--mbench_name',
        type=str,
        default='mbench')
    args = parser.parse_args()
    main_args = OmegaConf.load(args.config)
    main_args.mode = args.mode
    main_args.mbench_name = args.mbench_name
    main(main_args)
