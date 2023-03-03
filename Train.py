# import configuration.opts_mlpcvae as opts
# # import configuration.opts as opts

# from trainer.mlptransformer_trainer import TransformerTrainer
# # from trainer.transformer_trainer import TransformerTrainer

import os
import argparse
import pandas as pd
import numpy as np
import torch

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
from Model.build_model import get_model, freeze_params

from Configuration.config import model_opts, \
    klAnnealing_opts, optimTasks_opts


def train_opts(parser):
    """main settings"""

    parser.add_argument('-benchmark', type=str, default='moses')
    parser.add_argument('-start_epoch', type=int)
    parser.add_argument('-num_epoch', type=int, default=30)
    parser.add_argument('-max_strlen', type=int, default=80)
    parser.add_argument('-property_list', nargs='+', default=['logP', 'tPSA', 'QED'])
    parser.add_argument('-model_folder', type=str, required=True)

    parser.add_argument('-debug', action='store_true')

    """sub settings"""

    model_opts(parser)
    klAnnealing_opts(parser)
    optimTasks_opts(parser)

    parser.add_argument('-similarity', type=float, default=1) # X
    parser.add_argument('-tolerance', type=float, default=0) # X
    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-train_params', type=str, nargs='+')
    parser.add_argument('-pad_to_same_len', action='store_true')
    parser.add_argument('-data_folder', type=str, default='/fileserver-gamma/chaoting/ML/dataset/')


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


if __name__ == "__main__":
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

    LOG.info('Allocate GPU...')

    device = allocate_gpu()

    LOG.info('Prepare input data...')

    prepare_input_data(args.property_list, args.benchmark, raw_folder,
                       prepared_folder, util_folder, args.debug)
    iter_fields, SRC, TRG = get_iter_field(args.property_list, util_folder)
    if args.benchmark == 'moses':
        train, valid = get_dataset(prepared_folder, iter_fields,
                                   ['train', 'test', None])
    else:
        train, valid = get_dataset(prepared_folder, iter_fields,
                                   ['train', 'validation', None])        

    LOG.info(f'# train: {len(train)}, # validation: {len(valid)}')

    train_iter, valid_iter = get_iterator(train, valid, args.batch_size)

    args.train_nbatches = int(np.ceil(len(train) / args.batch_size))
    args.valid_nbatches = int(np.ceil(len(valid) / args.batch_size))

    args.sos_id = TRG.vocab.stoi['<sos>']
    args.eos_id = TRG.vocab.stoi['<eos>']
    args.pad_id = SRC.vocab.stoi['<pad>']

    LOG.info('Get model...')

    if args.start_epoch == 1:
        model = get_model(args, len(SRC.vocab), len(TRG.vocab))
    else:
        args.model_path = os.path.join(args.model_folder, f'model_{args.start_epoch-1}.pt')
        model = get_model(args, len(SRC.vocab), len(TRG.vocab), args.model_path)
    model = model.to(device)

    if args.train_params:
        freeze_params(model, train_names=args.train_params)
    
    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert train_params > 0, '# trainable parameters = 0'

    LOG.info(f'# total params: {total_params}, # train params: {train_params}')

    LOG.info('Get optimizer...')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
        betas=(args.lr_beta1, args.lr_beta2), eps=args.lr_eps
    )

    if args.start_epoch > 1:
        checkpoint = torch.load(args.model_path, map_location='cuda:0')
        optim_dict = checkpoint['opt_state_dict']
        optimizer.load_state_dict(optim_dict)

    LOG.info('Train model...')
    
    if args.model_type == 'cvaetf':
        train_cvaetf(args, model, optimizer, train_iter,
                     valid_iter, device, LOG)
