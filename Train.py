# import configuration.opts_mlpcvae as opts
# # import configuration.opts as opts

# from trainer.mlptransformer_trainer import TransformerTrainer
# # from trainer.transformer_trainer import TransformerTrainer

import os
import torch
import argparse
import pandas as pd
import numpy as np
from functools import partial
from torchtext import data
from torch.utils.data import DataLoader
from Utils.dataset import SmilesDataset

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from Model.collate_fn import get_collate_fn

# from Train.train import train
# from Train.mlp_train import mlp_train
# from Train.plot_results import plot_results
# from Train.attencvaetf_train import attencvaetf_train
# from Train.mlpcvaetf_encoder_train import mlpcvaetf_encoder_train
# from Train.cvaetfcut_train import cvaetfcut_train
# from Train.mlpcvaetf_train import mlpcvaetf_train
# from Train.sepcvaetf_train import sepcvaetf_train
# from Train.sepcvaetf2_train import sepcvaetf2_train
# from Train.ctf_train import ctf_train
# from Train.attenctf_train import attenctf_train
# from Configuration.config import options, hard_constraints_opts

from Utils.seed import set_seed
from Utils.log import get_logger
from Utils.gpu import allocate_gpu
from Utils.scaler import get_scaler
from Utils.field import get_iter_field, smiles_field
from Utils.dataset import get_dataset, get_iterator
from Train.trainer import train_model
from Train.train_attencvaetf import train_model as train_attencvaetf
from Train.train_cvaetf_bk import train_model as train_cvaetf_bk

from Model.build_model import get_model, freeze_params
from Configuration.config import model_opts, \
    klAnnealing_opts, optimTasks_opts


def train_opts(parser):
    model_opts(parser)
    klAnnealing_opts(parser)
    optimTasks_opts(parser)

    """main settings"""
    parser.add_argument('-benchmark', type=str, default='moses')
    parser.add_argument('-start_epoch', type=int, default=1)
    parser.add_argument('-num_epoch', type=int, default=30)
    parser.add_argument('-max_strlen', type=int, default=80)
    parser.add_argument('-property_list', nargs='+',
                        default=['logP', 'tPSA', 'QED'])
    parser.add_argument('-original_model_path', type=str)
    parser.add_argument('-model_folder', type=str, required=True)

    """sub settings"""
    parser.add_argument('-similarity', type=float, default=1)
    # parser.add_argument('-tolerance', type=float, default=0)
    parser.add_argument('-batch_size', type=int, default=32)
    parser.add_argument('-train_params', type=str, nargs='+')
    parser.add_argument('-randomize', action='store_true')
    parser.add_argument('-use_scaffold', action='store_true')
    parser.add_argument('-pad_to_same_len', action='store_true')
    parser.add_argument('-data_folder', type=str,
                        default='/fileserver-gamma/chaoting/ML/dataset/')

    parser.add_argument('-debug', action='store_true')


def save_prepared_data(raw_data, property_list, save_path):
    src = raw_data.loc[:, ['smiles']].rename(columns={'smiles': 'src'})
    trg = raw_data.loc[:, ['smiles']].rename(columns={'smiles': 'trg'})

    prop = raw_data.loc[:, property_list]
    src_prop = prop.rename(columns={p: f'src_{p}' for p in property_list})
    trg_prop = prop.rename(columns={p: f'trg_{p}' for p in property_list})

    prepared_data = pd.concat([src, src_prop, trg, trg_prop], axis=1)
    prepared_data.to_csv(save_path, index=False)


def prepare_input_data(property_list, benchmark, raw_folder,
                       prepared_folder, util_folder, debug=False):
    if benchmark == 'moses':
        data_type_list = ('train', 'test')
    elif benchmark == 'chembl_02':
        data_type_list = ('train', 'validation')
    elif benchmark == 'guacamol':
        exit('不知道')

    for data_type in data_type_list:
        raw_data = pd.read_csv(os.path.join(raw_folder, f'{data_type}.csv'),
                               index_col=[0])
        if debug:
            raw_data = raw_data[:1000]

        raw_smiles = raw_data.loc[:, ['smiles']]
        raw_prop = raw_data.loc[:, property_list]

        if data_type == 'train':
            scaler = get_scaler(property_list, util_folder,
                                raw_prop, rebuild=True)

        raw_prop.loc[:, property_list] = scaler.transform(raw_prop)
        raw_data = pd.concat([raw_smiles, raw_prop], axis=1)

        save_prepared_data(raw_data, property_list,
                           os.path.join(prepared_folder, f'{data_type}.csv'))


# def init_distributed():

#     # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
#     dist_url = "env://" # default

#     # only works with torch.distributed.launch // torch.run

#     print(os.environ)
#     rank = int(os.environ["RANK"])
#     world_size = int(os.environ['WORLD_SIZE']) # number of GPUs
#     local_rank = int(os.environ['LOCAL_RANK'])

#     print('rank:', rank)
#     print('world_size:', world_size)
#     print('local_rank:', local_rank)

#     dist.init_process_group(
#             backend="nccl",
#             init_method=dist_url,
#             world_size=world_size,
#             rank=rank)

#     # this will make all .cuda() calls work properly
#     torch.cuda.set_device(local_rank)

#     # synchronizes all the threads to reach this point before moving on
#     dist.barrier()


def get_data_name(benchmark, similarity, use_scaffold=False, debug=False):
    if benchmark == 'moses':
        data_name = ['train', 'test']
    elif benchmark == 'chembl_02':
        data_name = ['train', 'validation']
    
    for i in range(len(data_name)):
        data_name[i] += '_sca' if use_scaffold else ''
        data_name[i] += f'-s{similarity:.2f}'
        data_name[i] += '_debug' if debug else ''        
    return data_name


def prepare_dataloader(dataset, rank, world_size, batch_size, shuffle,
                       collate_fn, pin_memory=False, num_workers=0):
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size,
                                     rank=rank, shuffle=shuffle, drop_last=False)
        shuffle_loader = False  # use sampler -> turn off shuffle in dataloader
    else:
        sampler = None
        shuffle_loader = shuffle

    dataloader = DataLoader(dataset, batch_size=batch_size,
                            pin_memory=pin_memory, num_workers=num_workers,
                            drop_last=False, sampler=sampler, shuffle=shuffle_loader,
                            collate_fn=collate_fn)
    return dataloader


def main(rank, world_size):    
    if world_size > 1:
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size
        )

    set_seed(0)  # 0, 100, 200, 400

    parser = argparse.ArgumentParser()
    train_opts(parser)
    args = parser.parse_args()

    raw_folder = os.path.join(args.data_folder, args.benchmark, 'raw')
    prepared_folder = os.path.join(
        args.data_folder, args.benchmark, 'prepared')
    util_folder = os.path.join(args.data_folder, args.benchmark, 'utils')

    os.makedirs(args.model_folder, exist_ok=True)
    os.makedirs(prepared_folder, exist_ok=True)
    os.makedirs(util_folder, exist_ok=True)

    logger = get_logger()

    LOG = logger(name='train', log_path=os.path.join(
        args.model_folder, "records.log"))

    if rank == 0:
        LOG.info(args)
        LOG.info(f'world size: {world_size}')

    LOG.info(f'Get GPU: rank = {rank}')
    LOG.info('Prepare input data...')

    # prepare_input_data(args.property_list, args.benchmark, raw_folder,
    #                    prepared_folder, util_folder, args.debug)
    
    if args.model_type == 'scacvaetfv1' or args.model_type == 'scacvaetfv2':
        SRC, TRG = smiles_field(args.property_list, util_folder, suffix='molgct')
    elif args.model_type == 'scacvaetfv3':
        SRC, TRG = smiles_field(args.property_list, util_folder)        
    
    # iter_fields, SRC, TRG = get_iter_field(args.property_list, util_folder)

    LOG.info(f'SRC: {SRC.vocab.stoi}')
    LOG.info(f'TRG: {TRG.vocab.stoi}')
    
    train_name, valid_name = get_data_name(args.benchmark,
        args.similarity, args.use_scaffold, args.debug)
    
    train = pd.read_csv(os.path.join(prepared_folder,
                                     f'{train_name}.csv'))
    valid = pd.read_csv(os.path.join(prepared_folder,
                                     f'{valid_name}.csv'))

    train = SmilesDataset(train, args.property_list, SRC, TRG,
                          use_scaffold=args.use_scaffold,
                          randomize=args.randomize)
    valid = SmilesDataset(valid, args.property_list, SRC, TRG,
                          use_scaffold=args.use_scaffold,
                          randomize=args.randomize)

    if rank == 0:
        LOG.info(f'# train: {len(train)}, # validation: {len(valid)}')

    if args.debug:
        args.batch_size = 2

    LOG.info('Prepare dataloader...')
    
    collate_fn = get_collate_fn(args.model_type, SRC, TRG, rank)
    train_loader = prepare_dataloader(train, rank, world_size, args.batch_size,
                                      shuffle=True, collate_fn=collate_fn)
    valid_loader = prepare_dataloader(valid, rank, world_size, args.batch_size,
                                      shuffle=False, collate_fn=collate_fn)
    
    if rank == 0:
        LOG.info(
            f'# train loader: {len(train_loader)}, # validation loader: {len(valid_loader)}')

    args.sos_id = TRG.vocab.stoi['<sos>']
    args.eos_id = TRG.vocab.stoi['<eos>']
    args.pad_id = SRC.vocab.stoi['<pad>']
    args.TRG = TRG

    LOG.info(f'rank = {rank}, Get model...')

    if args.start_epoch == 1:
        model = get_model(args, len(SRC.vocab), len(TRG.vocab), rank)
    else:
        # TODO: fix bug - attencvaetf should be original_model_path
        args.model_path = os.path.join(args.model_folder, f'model_{args.start_epoch-1}.pt')
        model = get_model(args, len(SRC.vocab), len(TRG.vocab), rank)
        
    model = model.to(rank)
    
    if world_size > 1:
        model = DDP(model, device_ids=[rank], output_device=rank)

    # for name, params in model.named_parameters():
        # print(name, params.size(), params.requires_grad)
        # params.requires_grad = True
    # exit()

    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert train_params > 0, '# trainable parameters = 0'

    LOG.info(f'# total params: {total_params}, # train params: {train_params}')

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

    # if args.model_type == 'cvaetf':
    #     train_cvaetf(args, model, optimizer, train_iter,
    #                  valid_iter, rank, LOG)

    if args.model_type == 'attencvaetf':
        train_attencvaetf(args, model, optimizer,
                          train_loader, valid_loader, rank,
                          world_size, LOG)

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
        os.environ["MASTER_PORT"] = "29500"

        mp.spawn(main,
                 args=(world_size,),
                 nprocs=world_size,
                 join=True)
        cleanup()

    else:
        device = allocate_gpu()
        main(rank=device, world_size=1)
