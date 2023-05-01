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


# def save_prepared_data(raw_data, property_list, save_path):
#     src = raw_data.loc[:, ['smiles']].rename(columns={'smiles': 'src'})
#     trg = raw_data.loc[:, ['smiles']].rename(columns={'smiles': 'trg'})

#     prop = raw_data.loc[:, property_list]
#     src_prop = prop.rename(columns={p: f'src_{p}' for p in property_list})
#     trg_prop = prop.rename(columns={p: f'trg_{p}' for p in property_list})

#     prepared_data = pd.concat([src, src_prop, trg, trg_prop], axis=1)
#     prepared_data.to_csv(save_path, index=False)


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
        data_name = [name+'_scasmi' for name in data_name]
        # data_name = [name+'_sca' for name in data_name]

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


def main(rank, world_size):
    if world_size > 1:
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size
        )

    # set_seed(0) # 0
    # set_seed(100) # 100
    # set_seed(200) # 200
    set_seed(1000) # 1000

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

    if rank == 0:
        LOG.info(args)
        LOG.info(f'world size: {world_size}')
    LOG.info(f'Get GPU: rank = {rank}')

    SRC, TRG = get_fields(args.model_type, args.property_list, util_folder)

    LOG.info(f'SRC: {SRC.vocab.stoi}')
    LOG.info(f'TRG: {TRG.vocab.stoi}')
    
    if args.debug:
        args.batch_size = 64

    LOG.info('Get dataloader...')

    train_name, valid_name = get_data_name(args.benchmark, args.model_type, args.property_list)
    if args.debug:
        train_name = valid_name

    train = pd.read_csv(os.path.join(prepared_folder, f'{train_name}.csv'))
    valid = pd.read_csv(os.path.join(prepared_folder, f'{valid_name}.csv'))
    
    dp = DataloaderPreparation(rank, SRC, TRG, args.model_type,
                               args.property_list, world_size,
                               args.randomize, args.use_scaffold)
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
        # os.environ["MASTER_PORT"] = "29500"

        mp.spawn(main,
                 args=(world_size,),
                 nprocs=world_size,
                 join=True)
        cleanup()

    else:
        device = allocate_gpu()
        main(rank=device, world_size=1)
