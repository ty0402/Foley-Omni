
import os
import torch
import torch.distributed as dist


def get_global_rank() -> int:
    """
    Get the global rank, the global index of the GPU.
    """
    return int(os.environ.get("RANK", "0"))


def get_local_rank() -> int:
    """
    Get the local rank, the local index of the GPU.
    """
    lr = int(os.environ.get("LOCAL_RANK", "0"))
    # 支持每张 GPU 上启动多个进程：当 LOCAL_RANK 超过物理 GPU 数量时，对设备数取模
    try:
        num_devs = torch.cuda.device_count()
        if num_devs > 0:
            lr = lr % num_devs
    except Exception:
        pass
    return lr


def get_world_size() -> int:
    """
    Get the world size, the total amount of GPUs.
    """
    return int(os.environ.get("WORLD_SIZE", "1"))


def get_device() -> torch.device:
    """
    Get current rank device.
    """
    return torch.device("cuda", get_local_rank())

def get_sequence_parallel_group():
    """Get the sequence parallel group the caller rank belongs to."""
    return _SEQUENCE_PARALLEL_GROUP

def initialize_sequence_parallelism(sequence_parallel_size):
    assert int(get_world_size()) % sequence_parallel_size == 0
    sequence_parallel_num_groups = int(get_world_size()) // sequence_parallel_size
    global _SEQUENCE_PARALLEL_GROUP
    for i in range(sequence_parallel_num_groups):
        ranks = range(i * sequence_parallel_size,
                    (i + 1) * sequence_parallel_size)
        group = torch.distributed.new_group(ranks)
        if int(get_global_rank()) in ranks:
            print(f"Rank {get_global_rank()} joined group with ranks {list(ranks)}")
            _SEQUENCE_PARALLEL_GROUP = group