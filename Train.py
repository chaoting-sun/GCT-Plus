import os
import torch
import argparse
import pandas as pd

# modules for distributed data-parallel training (DDP)
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from Train.trainer import train_model
from Configuration.config import model_opts, \
    klAnnealing_opts, optimTasks_opts

from Configuration.config import train_opts
from Model.build_model import get_model
from Utils import (
    set_seed,
    get_logger,
    allocate_gpu,
    smiles_field,
    DataloaderPreparation
)
from Utils.field import train_field
from torch.utils.data.distributed import DistributedSampler


def save_prepared_data(raw_data, property_list, save_path):
    src = raw_data.loc[:, ['smiles']].rename(columns={'smiles': 'src'})
    trg = raw_data.loc[:, ['smiles']].rename(columns={'smiles': 'trg'})

    prop = raw_data.loc[:, property_list]
    src_prop = prop.rename(columns={p: f'src_{p}' for p in property_list})
    trg_prop = prop.rename(columns={p: f'trg_{p}' for p in property_list})

    prepared_data = pd.concat([src, src_prop, trg, trg_prop], axis=1)
    prepared_data.to_csv(save_path, index=False)


# def prepare_input_data(property_list, benchmark, raw_folder,
#                        prepared_folder, util_folder, debug=False):
#     if benchmark == 'moses':
#         data_type_list = ('train', 'test')
#     elif benchmark == 'chembl_02':
#         data_type_list = ('train', 'validation')
#     elif benchmark == 'guacamol':
#         exit('不知道')

#     for data_type in data_type_list:
#         raw_data = pd.read_csv(os.path.join(raw_folder, f'{data_type}.csv'),
#                                index_col=[0])
#         if debug:
#             raw_data = raw_data[:1000]

#         raw_smiles = raw_data.loc[:, ['smiles']]
#         raw_prop = raw_data.loc[:, property_list]

#         if data_type == 'train':
#             scaler = get_scaler(property_list, util_folder,
#                                 raw_prop, rebuild=True)

#         raw_prop.loc[:, property_list] = scaler.transform(raw_prop)
#         raw_data = pd.concat([raw_smiles, raw_prop], axis=1)

#         save_prepared_data(raw_data, property_list,
#                            os.path.join(prepared_folder, f'{data_type}.csv'))


# def prepare_dataloader(dataset, rank, world_size, batch_size, shuffle,
#                        collate_fn, pin_memory=False, num_workers=0):
#     if world_size > 1:
#         sampler = DistributedSampler(dataset, num_replicas=world_size,
#                                      rank=rank, shuffle=shuffle, drop_last=False)
#         shuffle_loader = False  # use sampler -> turn off shuffle in dataloader
#     else:
#         sampler = None
#         shuffle_loader = shuffle

#     dataloader = DataLoader(dataset, batch_size=batch_size,
#                             pin_memory=pin_memory, num_workers=num_workers,
#                             drop_last=False, sampler=sampler, shuffle=shuffle_loader,
#                             collate_fn=collate_fn)
#     return dataloader


def get_data_name(benchmark, model_type, property_list):
    if benchmark == 'moses':
        data_name = ['train', 'test']
    elif benchmark == 'chembl_02':
        data_name = ['train', 'validation']
    
    if model_type == 'vaetf':
        data_name = [name+'_v0' for name in data_name]
    elif model_type =='cvaetf':
        data_name = [name+'_v0' for name in data_name]
    elif model_type == 'scacvaetfv3':
        data_name = [name+'_sca' for name in data_name]

    # for i in range(len(data_name)):
    #     # data_name[i] += f'-s{similarity:.2f}'
    #     data_name[i] += '_debug' if debug else ''
    return data_name


def get_fields(model_type, property_list, field_path):
    if model_type in ('vaetf', 'cvaetf', 'scacvaetfv1', 'scacvaetfv2'):
        SRC, TRG = smiles_field(field_path, add_sep=False)
        # SRC, TRG = smiles_field(properties=property_list,
        #                         field_path=field_path,
        #                         suffix='molgct')
    elif model_type in ('scacvaetfv3'):
        SRC, TRG = smiles_field(field_path, add_sep=True)
    return SRC, TRG


from torchtext import data
from time import time

class MyIterator(data.Iterator):
    def __init__(self, dataset, batch_size, sort_key, device, sampler=None, **kwargs):
        self.sampler = sampler
        super(MyIterator, self).__init__(dataset, batch_size, sort_key, device=device, **kwargs)

    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(sorted(p, key=self.sort_key), self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b
            if self.sampler:
                self.batches = pool([self.data()[i] for i in self.sampler],
                                     self.random_shuffler)
            else:
                self.batches = pool(self.data(), self.random_shuffler)
        else:
            if self.sampler:
                idxs = list(self.sampler)
                self.batches = [sorted([self.data()[i] for i in b], key=self.sort_key)
                                for b in data.batch(idxs, self.batch_size)]
            else:
                self.batches = []
                for b in data.batch(self.data(), self.batch_size,
                                          self.batch_size_fn):
                    self.batches.append(sorted(b, key=self.sort_key))


# class MyIterator(data.Iterator):
#     def create_batches(self):
#         if self.train:
#             def pool(d, random_shuffler):
#                 for p in data.batch(d, self.batch_size * 100):
#                     p_batch = data.batch(sorted(p, key=self.sort_key), self.batch_size, self.batch_size_fn)
#                     for b in random_shuffler(list(p_batch)):
#                         yield b
#             self.batches = pool(self.data(), self.random_shuffler)
#             # print('self.batches:', len([i for i in self.batches]))
            
#         else:
#             self.batches = []
#             for b in data.batch(self.data(), self.batch_size,
#                                           self.batch_size_fn):
#                 self.batches.append(sorted(b, key=self.sort_key))


global max_src_in_batch, max_tgt_in_batch

def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    effective_batch_size = max(src_elements, tgt_elements)
    print(f"count: {count}, max_src: {max_src_in_batch}, max_tgt: {max_tgt_in_batch}, effective_batch_size: {effective_batch_size}")
    return max(src_elements, tgt_elements)


import random
from torch.utils.data import BatchSampler
from torch.utils.data import DataLoader

class PoolingBatchSampler(BatchSampler):
    def __init__(self, sampler, batch_size, drop_last, shuffle=True, pool_factor=100):
        super().__init__(sampler, batch_size, drop_last)
        self.shuffle = shuffle
        self.pool_factor = pool_factor

    def __iter__(self):
        if self.shuffle:
            for p in BatchSampler(self.sampler, self.batch_size * self.pool_factor, False):
                p_batch = list(BatchSampler(p, self.batch_size, self.drop_last))
                random.shuffle(p_batch)
                for batch in p_batch:
                    yield batch
        else:
            for batch in BatchSampler(self.sampler, self.batch_size, self.drop_last):
                yield batch


def main(rank, world_size):
    if world_size > 1:
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size
        )
    seed_no = 1
    # set_seed(0) # 0
    # set_seed(100) # 100
    # set_seed(200) # 200
    # set_seed(1000) # 1000
    set_seed(seed_no)

    parser = argparse.ArgumentParser()
    train_opts(parser)
    args = parser.parse_args()

    prepared_folder = os.path.join(args.data_folder, args.benchmark, 'prepared')
    util_folder = os.path.join(args.data_folder, args.benchmark, 'utils')

    os.makedirs(args.model_folder, exist_ok=True)
    os.makedirs(prepared_folder, exist_ok=True)
    os.makedirs(util_folder, exist_ok=True)

    logger = get_logger()

    LOG = logger(name='train', log_path=os.path.join(
        args.model_folder, "records.log"))

    LOG.info(f'seed: {seed_no}')

    if rank == 0:
        LOG.info(args)
        LOG.info(f'world size: {world_size}')
    LOG.info(f'Get GPU: rank = {rank}')

    # field

    data_fields, SRC, TRG = train_field[args.model_type](args.property_list,
                                                         args.use_scaffold,
                                                         util_folder)
    args.SRC = SRC
    args.TRG = TRG

    LOG.info(f'SRC: {SRC.vocab.stoi}')
    LOG.info(f'TRG: {TRG.vocab.stoi}')
    
    if args.debug:
        args.batch_size = 128

    LOG.info('Get dataloader...')

    if args.model_type == 'scacvaetfv3':
        train_name = 'train_sca'
        valid_name = 'test_sca'
        
    else:
        train_name = 'train'
        valid_name = 'test'

    if args.debug:
        train_name = valid_name

    LOG.info(f'train/valid name: {train_name}, {valid_name}')

    train, valid = data.TabularDataset.splits(
        path=prepared_folder, train=f'{train_name}.csv',
        validation=f'{valid_name}.csv', format='csv',
        skip_header=True, fields=data_fields)

    LOG.info('fields: %s', data_fields)

    train_sampler = None
    if world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train,
                                                                        num_replicas=world_size,
                                                                        rank=rank,
                                                                        shuffle=True)

    train_loader = MyIterator(train, args.batch_size, device=f'cuda:{rank}',
                              sort_key=lambda x: (len(x.src), len(x.trg)),
                              sort_within_batch=True, sampler=train_sampler,
                              shuffle=False)
    
    # train_loader = MyIterator(train, args.batch_size, device=f'cuda:{rank}',
    #                           sort_key=lambda x: (len(x.src), len(x.trg)),
    #                           sort_within_batch=True)

    valid_loader = MyIterator(valid, args.batch_size, device=f'cuda:{rank}',
                              sort_key=lambda x: (len(x.src), len(x.trg)),
                              sort_within_batch=True, shuffle=False)

    # train_loader = MyIterator(train, args.batch_size, device=f'cuda:{rank}',
    #                           sort_key=lambda x: (len(x.src), len(x.trg)),
    #                           sampler=train_sampler, sort_within_batch=False)

    # valid_loader = MyIterator(valid, args.batch_size, device=f'cuda:{rank}',
    #                           sort_key=lambda x: (len(x.src), len(x.trg)),
    #                           sampler=None, sort_within_batch=False)


    # train_loader, valid_loader = MyIterator.splits(
    #     (train, valid), batch_sizes=(args.batch_size, args.batch_size),
    #     device=device, sort_key=lambda x: (len(x.src), len(x.trg)),
    #     repeat=True, sort=False, shuffle=True, sort_within_batch=False,
    #     )

    #  = data.TabularDataset(os.path.join(prepared_folder, 'train_debug.csv'),
    #                             format='csv', fields=data_fields, skip_header=True)

    # train_sampler = None
    # if world_size > 1:
    #     train_sampler = torch.utils.data.distributed.DistributedSampler(train,
    #                                                                     num_replicas=world_size,
    #                                                                     rank=rank,
    #                                                                     shuffle=True)
    # train_iter = MyIterator(train, batch_size=args.batch_size, device=f'cuda:{rank}',
    #                         repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
    #                         batch_size_fn=batch_size_fn, train=True)

    # train_iter = MyIterator(train, batch_size=args.batch_size, device=f'cuda:{rank}',
    #                         repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
    #                         batch_size_fn=batch_size_fn, train=True, sampler=train_sampler)

    # train_sampler = torch.utils.data.distributed.DistributedSampler(train,
    #                                                                 num_replicas=world_size,
    #                                                                 rank=rank,
    #                                                                 shuffle=True)
    
    # train_iter = MyIterator(train, batch_size=args.batch_size, device=f'cuda:{rank}',
    #                         repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
    #                         batch_size_fn=batch_size_fn, train=True, shuffle=False,
    #                         sampler=train_sampler)

    # data_loader = DataLoader(SmilesDataset, collate_fn=lambda ins: vaetf_collate_fn(ins, SRC, TRG, device), batch_sampler=batch_sampler)

    if rank == 0:
        LOG.info(f'# train: {len(train)}, # validation: {len(valid)}')    

    args.sos_id = TRG.vocab.stoi['<sos>']
    args.eos_id = TRG.vocab.stoi['<eos>']
    args.pad_id = SRC.vocab.stoi['<pad>']
    args.TRG = TRG

    LOG.info('Get model...')

    if args.start_epoch > 1:
        args.model_path = os.path.join(args.model_folder, f'model_{args.start_epoch-1}.pt')
    model = get_model(args, len(SRC.vocab), len(TRG.vocab), rank)
    model = model.to(rank)
    
    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert train_params > 0, '# trainable parameters = 0'
    
    LOG.info(f'# total params: {total_params}, # train params: {train_params}')

    if world_size > 1:
        model = DDP(model, device_ids=[rank], output_device=rank)

    # for name, params in model.named_parameters():
        # print(name, params.size(), params.requires_grad)
        # params.requires_grad = True
    # exit()

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
        os.environ["MASTER_PORT"] = "29500"

        mp.spawn(main,
                 args=(world_size,),
                 nprocs=world_size,
                 join=True)
        cleanup()

    else:
        device = allocate_gpu()
        main(rank=device, world_size=1)
