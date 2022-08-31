import os
import torch
import torch.distributed as dist


def allocate_gpu(id=None):
    v = torch.empty(1)
    if id is not None:
        return torch.device("cuda:{}".format(str(id)))
    else:
        for i in range(4):
            try:
                dev_name = "cuda:{}".format(str(i))
                v = v.to(dev_name)
                print("Allocating cuda:{}.".format(i))
                return v.device
            except Exception as e:
                pass
        print("CUDA error: all CUDA-capable devices are busy or unavailable")
        return v.device


def allocate_multiple_gpu(id_list):
    '''
    choose the free gpu in the node
    '''
    if not isinstance(id_list[0], str):
        id_list = [str(i) for i in id_list]

    return torch.device("cuda:{}".format(','.join(id_list)))


def get_number_gpus():
    """get the number of GPUs """
    num_gpus = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    return num_gpus


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    - ref: https://github.com/Res2Net/Res2Net-maskrcnn/blob/master/maskrcnn_benchmark/utils/comm.py
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    else:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    if world_size == 1:
        return 
    dist.barrier()
    return rank, world_size


def initialize_process_group(local_rank):
    """
    set up the current GPUs and initialize the group.
    nccl is used for backend communication, which should be downlaoded first.
    The official guide recommends os.environ["CUDA_VISIBLE_DEVICES"] = "0" more.
    """
    print("@ initialize process group:")

    print("  - CUDA set device :", local_rank)
    torch.cuda.set_device(local_rank)
    
    print("  - initialize process group")
    torch.distributed.init_process_group(backend="nccl", init_method="env://")

    print("  - synchronize")
    synchronize()