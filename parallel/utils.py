import os
import torch.distributed as dist
# from accelerate.tracking import GeneralTracker, on_main_process
from torch.distributed.device_mesh import init_device_mesh


def is_main_process():
    return dist.get_rank() == 0


def do_nothing(*args, **kwargs):
    return None


def on_main_process(function):
    """Decorator to selectively run the decorated function on the main process
    only based on the `main_process_only` attribute in a class.

    Checks at function execution rather than initialization time, not
    triggering the initialization of the `PartialState`.
    """

    def execute_on_main_process(self, *args, **kwargs):
        if getattr(self, 'main_process_only', False):
            if is_main_process():
                return function(self, *args, **kwargs)
            else:
                return do_nothing(self, *args, **kwargs)
        else:
            return function(self, *args, **kwargs)

    return execute_on_main_process


def get_local_rank():
    return int(os.environ['LOCAL_RANK'])


def get_local_world_size():
    return int(os.environ['LOCAL_WORLD_SIZE'])


def get_rank():
    return int(os.environ['RANK'])


def get_world_size():
    # can be called before dist initialization
    return int(os.environ['WORLD_SIZE'])


def get_nodes():
    return int(os.environ['GROUP_WORLD_SIZE'])


def get_node_rank():
    return int(os.environ['GROUP_RANK'])


def get_device_mesh(use_hybrid: bool = False, tp_size: int = None):
    world_size = get_world_size()
    n_nodes = get_nodes()
    replicate_size = n_nodes if use_hybrid else 1

    assert (world_size % ((tp_size or 1) * replicate_size)) == 0
    shard_size = world_size // (replicate_size * (tp_size or 1))
    if use_hybrid:
        if tp_size is None:
            mesh_shape = (replicate_size, shard_size)
            mesh_dim_names = ('replicate', 'dp')
        else:
            mesh_shape = (replicate_size, shard_size, tp_size)
            mesh_dim_names = ('replicate', 'dp', 'tp')
    else:
        if tp_size is None:
            mesh_shape = (shard_size, )
            mesh_dim_names = ('dp', )
        else:
            mesh_shape = (shard_size, tp_size)
            mesh_dim_names = ('dp', 'tp')
    device_mesh = init_device_mesh(
        'cuda', mesh_shape, mesh_dim_names=mesh_dim_names)
    return device_mesh
