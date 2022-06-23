import os
import random
import numpy as np
import torch


def set_seed(seed=0):
    """
    souce: https://blog.csdn.net/weixin_43646592/article/details/121234595
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    torch.cuda.manual_seed_all(seed) # all GPUs
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False
