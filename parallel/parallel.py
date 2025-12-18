import torch
import torch.nn as nn
from functools import partial
# from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
#     CheckpointImpl,
#     apply_activation_checkpointing,
#     checkpoint_wrapper,
# )
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.api import BackwardPrefetch
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy

strategy_mapping = {
    'no_shard': ShardingStrategy.NO_SHARD,
    'op_grad': ShardingStrategy.SHARD_GRAD_OP,
    'full_shard': ShardingStrategy.FULL_SHARD,
    'hybrid': ShardingStrategy.HYBRID_SHARD,
}


def fsdp_text_encoder(model: nn.Module, device_mesh: DeviceMesh,
                      dtype: torch.dtype) -> FSDP:
    if hasattr(model, 'encoder'):
        fn = lambda m: m in list(model.encoder.block)
    elif hasattr(model, 'text_model'):
        fn = lambda m: m in list(model.text_model.encoder.layers)
    else:
        raise ValueError('unknown text encoder')

    model = FSDP(
        model,
        auto_wrap_policy=partial(
            lambda_auto_wrap_policy,
            lambda_fn=fn,
        ),
        device_mesh=device_mesh,
        sharding_strategy=ShardingStrategy.HYBRID_SHARD,
        mixed_precision=MixedPrecision(param_dtype=dtype, ),
        device_id=torch.cuda.current_device(),
        sync_module_states=True,
        limit_all_gathers=True,
        use_orig_params=True,
    )
    torch.cuda.synchronize()
    return model


def fsdp_transformer_ulysses(
    model: nn.Module,
    device_mesh: DeviceMesh,
    global_device_mesh: DeviceMesh,
    dtype: torch.dtype,
    grad_dtype: torch.dtype = None,
    strategy: str = 'op_grad',
) -> FSDP:
    tp_mesh = device_mesh['tp']
    if tp_mesh.size() > 1:
        model = model.__class__.sequence_parallelize_ulysses(model, tp_mesh)

    fn = lambda m: m in model.get_fsdp_wrap_module_list()
    strategy = strategy_mapping[strategy]
    model = FSDP(
        model,
        auto_wrap_policy=partial(
            lambda_auto_wrap_policy,
            lambda_fn=fn,
        ),
        device_mesh=global_device_mesh,
        sharding_strategy=strategy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        mixed_precision=MixedPrecision(
            param_dtype=dtype,
            reduce_dtype=grad_dtype or dtype,
            buffer_dtype=dtype,
        ),
        device_id=torch.cuda.current_device(),
        sync_module_states=True,
        limit_all_gathers=True,
        use_orig_params=True,
    )
    # non_reentrant_wrapper = partial(
    #     checkpoint_wrapper,
    #     checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    # )
    # apply_activation_checkpointing(
    #     model,
    #     checkpoint_wrapper_fn=non_reentrant_wrapper,
    #     check_fn=lambda m: m in model.get_fsdp_wrap_module_list()
    # )
    # model.enable_gradient_checkpointing()
    torch.cuda.synchronize()
    return model
