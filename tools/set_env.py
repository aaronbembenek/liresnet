import os
import subprocess

import torch
from torch import distributed as torch_dist


def setup_print(is_master):
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        if is_master:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def _backend_for(device):
    """NCCL only exists for CUDA; everything else (CPU, MPS) uses gloo."""
    return 'nccl' if device.type == 'cuda' else 'gloo'


def _world_size():
    return int(os.environ.get('WORLD_SIZE', 1))


def init_DDP(launcher, device):
    """Set up distributed training, if any.

    Args:
        launcher (str): `none` for a single process (no process group),
            or `pytorch` / `slurm` for multi-process training.
        device (torch.device): the device this process trains on.

    Returns:
        tuple: (rank, local_rank, world_size)
    """
    if launcher == 'none':
        print(f'Single-process training on {device}.')
        return 0, 0, 1

    print('Init distributed training...')
    if launcher == 'slurm':
        rank, local_rank, world_size = _init_dist_slurm(device)
    elif launcher == 'pytorch':
        rank, local_rank, world_size = _init_dist_pytorch(device)
    else:
        raise TypeError('Launcher not supported')
    setup_print(local_rank == 0)
    return rank, local_rank, world_size


def _init_dist_pytorch(device):
    rank = int(os.environ['RANK'])
    world_size = _world_size()
    local_rank = int(os.environ['LOCAL_RANK'])
    if device.type == 'cuda':
        torch.cuda.set_device(local_rank)
    torch_dist.init_process_group(backend=_backend_for(device))
    return rank, local_rank, world_size


def _init_dist_slurm(device):
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = max(torch.cuda.device_count(), 1)
    local_rank = proc_id % num_gpus
    if device.type == 'cuda':
        torch.cuda.set_device(local_rank)
    addr = subprocess.getoutput(
        f'scontrol show hostname {node_list} | head -n1')

    # specify master port
    if 'MASTER_PORT' not in os.environ:
        # 29500 is torch.distributed default port
        os.environ['MASTER_PORT'] = '29500'

    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = addr

    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['LOCAL_RANK'] = str(local_rank)
    os.environ['RANK'] = str(proc_id)
    torch_dist.init_process_group(backend=_backend_for(device))

    return proc_id, local_rank, _world_size()
