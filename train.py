import os
import torch
import random
import argparse
import pandas as pd
from torchtext import data
from torch.utils.data import BatchSampler

# modules for distributed data-parallel training (DDP)
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from Train.trainer import train_model

from Configuration.config import train_opts
from Model.build_model import get_model
from Utils import set_seed, get_logger, allocate_gpu, smiles_field
from Utils.field import get_train_field


def save_prepared_data(raw_data, property_list, save_path):
    src = raw_data.loc[:, ['smiles']].rename(columns={'smiles': 'src'})
    trg = raw_data.loc[:, ['smiles']].rename(columns={'smiles': 'trg'})

    prop = raw_data.loc[:, property_list]
    src_prop = prop.rename(columns={p: f'src_{p}' for p in property_list})
    trg_prop = prop.rename(columns={p: f'trg_{p}' for p in property_list})

    prepared_data = pd.concat([src, src_prop, trg, trg_prop], axis=1)
    prepared_data.to_csv(save_path, index=False)


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

    parser = argparse.ArgumentParser()
    train_opts(parser)
    args = parser.parse_args()

    set_seed(args.seed)

    os.makedirs(args.model_folder, exist_ok=True)
    os.makedirs(args.prepared_folder, exist_ok=True)
    os.makedirs(args.util_folder, exist_ok=True)

    logger = get_logger()

    LOG = logger(name='train', log_path=os.path.join(args.model_folder, "records.log"))

    LOG.info(f'seed: {args.seed}')

    if rank == 0:
        LOG.info(args)
        LOG.info(f'world size: {world_size}')
    LOG.info(f'Get GPU: rank = {rank}')

    # field

    data_fields, SRC, TRG = get_train_field(args.property_list, args.util_folder)
    args.SRC = SRC
    args.TRG = TRG

    LOG.info(f'SRC: {SRC.vocab.stoi}')
    LOG.info(f'TRG: {TRG.vocab.stoi}')
    
    if args.debug:
        args.batch_size = 128

    LOG.info('Get dataloader...')

    train_name = 'train'
    valid_name = 'test'

    if args.debug:
        train_name = valid_name

    LOG.info(f'train/valid name: {train_name}, {valid_name}')

    train, valid = data.TabularDataset.splits(
        path=args.prepared_folder, train=f'{train_name}.csv',
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

    valid_loader = MyIterator(valid, args.batch_size, device=f'cuda:{rank}',
                              sort_key=lambda x: (len(x.src), len(x.trg)),
                              sort_within_batch=True, shuffle=False)

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
        os.environ["MASTER_PORT"] = "29500"

        mp.spawn(main,
                 args=(world_size,),
                 nprocs=world_size,
                 join=True)
        cleanup()

    else:
        device = allocate_gpu()
        main(rank=device, world_size=1)
