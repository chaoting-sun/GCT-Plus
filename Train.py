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
from torch.utils.data import DataLoader
from Utils.dataset import SmilesDataset

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


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
from Utils.field import get_iter_field
from Utils.dataset import get_dataset, get_iterator
from Train.train_cvaetf import train_model as train_cvaetf
from Train.train_attencvaetf import train_model as train_attencvaetf

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
    parser.add_argument('-property_list', nargs='+', default=['logP', 'tPSA', 'QED'])
    parser.add_argument('-original_model_path', type=str)
    parser.add_argument('-model_folder', type=str, required=True)

    """sub settings"""
    parser.add_argument('-similarity_threshold', type=float, default=1)
    # parser.add_argument('-tolerance', type=float, default=0)
    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-train_params', type=str, nargs='+')
    parser.add_argument('-pad_to_same_len', action='store_true')
    parser.add_argument('-data_folder', type=str, default='/fileserver-gamma/chaoting/ML/dataset/')

    parser.add_argument('-debug', action='store_true')


def save_prepared_data(raw_data, property_list, save_path):
    src = raw_data.loc[:, ['smiles']].rename(columns={ 'smiles': 'src' })
    trg = raw_data.loc[:, ['smiles']].rename(columns={ 'smiles': 'trg' })
    
    prop = raw_data.loc[:, property_list]
    src_prop = prop.rename(columns={ p: f'src_{p}' for p in property_list })
    trg_prop = prop.rename(columns={ p: f'trg_{p}' for p in property_list })        

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


def prepare_dataloader(dataset, rank, world_size,
                       batch_size, shuffle, collate_fn,
                       pin_memory=False, num_workers=0):
    sampler = DistributedSampler(dataset, num_replicas=world_size,
        rank=rank, shuffle=shuffle, drop_last=False)
    
    dataloader = DataLoader(dataset, batch_size=batch_size,
        pin_memory=pin_memory, num_workers=num_workers, 
        drop_last=False, shuffle=False, sampler=sampler,
        collate_fn=collate_fn)

    return dataloader


def main(rank, world_size):
    if world_size > 1:
        dist.init_process_group(
            backend="gloo",
            rank=rank,
            world_size=world_size
        )
    
    set_seed(0) # 0, 100, 200, 400

    parser = argparse.ArgumentParser()
    train_opts(parser)
    args = parser.parse_args()

    raw_folder = os.path.join(args.data_folder, args.benchmark, 'raw')
    prepared_folder = os.path.join(args.data_folder, args.benchmark, 'prepared')
    util_folder = os.path.join(args.data_folder, args.benchmark, 'utils')
    
    os.makedirs(args.model_folder, exist_ok=True)
    os.makedirs(prepared_folder, exist_ok=True)
    os.makedirs(util_folder, exist_ok=True)

    logger = get_logger()
    LOG = logger(name='augment data by conditions',
                 log_path=os.path.join(args.model_folder, "records.log"))
    LOG.info(args)

    LOG.info('Prepare input data...')

    prepare_input_data(args.property_list, args.benchmark, raw_folder,
                       prepared_folder, util_folder, args.debug)
    iter_fields, SRC, TRG = get_iter_field(args.property_list, util_folder)

    if args.benchmark == 'moses':
        train_name = f'train-s{args.similarity_threshold:.2f}'
        valid_name = f'test-s{args.similarity_threshold:.2f}'
    elif args.benchmark == 'chembl_02':
        train_name = f'train-s{args.similarity_threshold:.2f}'
        valid_name = f'validation-s{args.similarity_threshold:.2f}'
    
    if args.debug:
        train_name += '_debug'
        valid_name += '_debug'

    train = pd.read_csv(os.path.join(prepared_folder,
                                     f'{train_name}.csv'))
    valid = pd.read_csv(os.path.join(prepared_folder,
                                     f'{valid_name}.csv'))

    train = SmilesDataset(train, args.property_list, rank, SRC, TRG,
                          include_mconds=True, debug=False)
    valid = SmilesDataset(train, args.property_list, rank, SRC, TRG,
                          include_mconds=True, debug=False)
    
    collate_fn = partial(SmilesDataset.collate_fcn,
                         SRC=SRC, TRG=TRG, device=rank)
    
    if args.debug:
        args.batch_size = 2
        
    train_loader = prepare_dataloader(train, rank,
        world_size, args.batch_size, shuffle=True,
        collate_fn=collate_fn)

    valid_loader = prepare_dataloader(valid, rank,
        world_size, args.batch_size, shuffle=False,
        collate_fn=collate_fn)


    exit()

    LOG.info(f'# train: {len(train_loader)}, # validation: {len(valid_loader)}')
    
    # train_iter, valid_iter = get_iterator(train, valid, args.batch_size)

    args.sos_id = TRG.vocab.stoi['<sos>']
    args.eos_id = TRG.vocab.stoi['<eos>']
    args.pad_id = SRC.vocab.stoi['<pad>']

    LOG.info('Get model...')

    if args.start_epoch == 1:
        model = get_model(args, len(SRC.vocab), len(TRG.vocab))
    else:
        args.model_path = os.path.join(args.model_folder,
                                       f'model_{args.start_epoch-1}.pt')
        model = get_model(args, len(SRC.vocab), len(TRG.vocab))
    
    model = model.to(rank)
    if world_size > 1:
        model = DDP(model, device_ids=[rank])
    
    # for name, params in model.named_parameters():
    #     print(name, params.size(), params.requires_grad)
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

    if args.start_epoch > 1:
        checkpoint = torch.load(args.model_path, map_location='cuda:0')
        optim_dict = checkpoint['opt_state_dict']
        optimizer.load_state_dict(optim_dict)

    LOG.info('Train model...')
    
    # if args.model_type == 'cvaetf':
    #     train_cvaetf(args, model, optimizer, train_iter,
    #                  valid_iter, rank, LOG)
        
    if args.model_type == 'attencvaetf':
        train_attencvaetf(args, model, optimizer, train_loader,
                          valid_loader, rank, rank, LOG)


def cleanup():
    dist.destroy_process_group()


if __name__ == "__main__":
    multi_gpus = False

    if multi_gpus:    
        world_size = 4
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
    
        mp.spawn(main,
                args=(world_size,),
                nprocs=world_size,
                join=True)
        cleanup()

    else:
        device = allocate_gpu()
        main(device, 1)