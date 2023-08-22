import os
import torch
import argparse
import pandas as pd

# modules for distributed data-parallel training (DDP)
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from Train.trainer1 import train_model
from Configuration.config import train_opts
from Model.build_model import get_model
from Utils import (
    set_seed,
    get_logger,
    allocate_gpu,
    smiles_field,
    DataloaderPreparation
)


def get_fields(model_type, file_path):
    if model_type == 'vaetf' or model_type == 'pvaetf':
        SRC, TRG = smiles_field(file_path, add_sep=False)
    elif model_type == 'scavaetf' or model_type == 'pscavaetf':
        SRC, TRG = smiles_field(file_path, add_sep=True)
    return SRC, TRG


def main(rank, world_size):
    if world_size > 1:
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size
        )
    parser = argparse.ArgumentParser()
    train_opts(parser)
    args = parser.parse_args()

    set_seed(args.seed)

    os.makedirs(args.model_folder, exist_ok=True)
    os.makedirs(args.prepared_folder, exist_ok=True)
    os.makedirs(args.util_folder, exist_ok=True)

    logger = get_logger()

    LOG = logger(name='train', log_path=os.path.join(
        args.model_folder, "records.log"))

    LOG.info('random seed: %s', args.seed)
    
    if rank == 0:
        LOG.info(args)
        LOG.info(f'world size: {world_size}')
    LOG.info(f'Get GPU: rank = {rank}')

    SRC, TRG = get_fields(args.model_type, args.util_folder)

    LOG.info(f'SRC: {SRC.vocab.stoi}')
    LOG.info(f'TRG: {TRG.vocab.stoi}')
    
    LOG.info('Get dataloader...')

    if args.model_type == 'vaetf' or args.model_type == 'pvaetf':
        train = pd.read_csv(os.path.join(args.prepared_folder, 'train.csv'))
        valid = pd.read_csv(os.path.join(args.prepared_folder, 'test.csv'))
    elif args.model_type == 'scavaetf' or args.model_type == 'pscavaetf':
        train = pd.read_csv(os.path.join(args.prepared_folder, 'train_sca.csv'))
        valid = pd.read_csv(os.path.join(args.prepared_folder, 'test_sca.csv'))

    if args.debug:
        args.batch_size = 4
        train = train[:32]
        valid = valid[:32]

    dp = DataloaderPreparation(rank, SRC, TRG, args.model_type,
                               args.property_list, world_size,
                               args.randomize_prob, args.use_scaffold)
    train_loader = dp.get_dataloader(train, batch_size=args.batch_size, is_train=True)
    valid_loader = dp.get_dataloader(valid, batch_size=args.batch_size, is_train=False)

    if rank == 0:
        LOG.info(f'# train: {len(train)}, # validation: {len(valid)}')    
        LOG.info(f'# train / validation loader: {len(train_loader)} / {len(valid_loader)}')

    args.sos_id = TRG.vocab.stoi['<sos>']
    args.eos_id = TRG.vocab.stoi['<eos>']
    args.pad_id = SRC.vocab.stoi['<pad>']
    args.TRG = TRG

    LOG.info('Get model...')

    if args.start_epoch > 1:
        args.model_path = os.path.join(args.model_folder, f'model_{args.start_epoch-1}.pt')
    model = get_model(args, len(SRC.vocab), len(TRG.vocab), rank)
    
    torch.cuda.set_device(rank)
    model = model.cuda()
    # model = model.to(rank)
    
    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert train_params > 0, '# trainable parameters = 0'
    
    LOG.info(f'# total params: {total_params}, # train params: {train_params}')

    if world_size > 1:
        model = DDP(model, device_ids=[rank], output_device=rank)

    LOG.info('Get optimizer...')

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, betas=(args.lr_beta1, args.lr_beta2), eps=args.lr_eps
    )
    
    # args.lr = args.lr * np.sqrt(world_size)
    # args.lr = args.lr * np.sqrt(int(world_size*(args.batch_size/128)))
    # rescale lr: https://github.com/Lightning-AI/lightning/discussions/3706#discussioncomment-238302

    if args.start_epoch > 1:
        model_path = os.path.join(args.model_folder, f'model_{args.start_epoch-1}.pt')        
        checkpoint = torch.load(model_path, map_location={ 'cuda:%d' % 0: 'cuda:%d' % rank })
        optim_dict = checkpoint['opt_state_dict']
        optimizer.load_state_dict(optim_dict)

    train_model(args, model, optimizer, train_loader,
                valid_loader, rank, world_size, LOG)


def cleanup():
    """
    is_initialized() to check if a process group exists before
    calling dist.barrier() to synchronize all processes and
    dist.destroy_process_group() to destroy the process group.

    dist.barrier() is used to ensure that all processes have
    completed their work before destroying the process group.
    This is important because destroying the process group
    while some processes are still working can lead to
    undefined behavior.
    """
    if torch.distributed.is_initialized():
        dist.barrier()  
        dist.destroy_process_group()


if __name__ == "__main__":
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
    
    device_count = torch.cuda.device_count()
    print('device count:', device_count)
    
    if device_count > 1:
        world_size = device_count
        os.environ["MASTER_ADDR"] = "localhost"
        # os.environ["MASTER_PORT"] = "29500"

        mp.spawn(main,
                 args=(world_size,),
                 nprocs=world_size,
                 join=True)
        cleanup()

    else:
        device = allocate_gpu()
        main(rank=device, world_size=1)
