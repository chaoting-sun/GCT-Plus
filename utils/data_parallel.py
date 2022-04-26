import os

from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch

num_gpus = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
