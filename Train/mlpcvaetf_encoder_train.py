import os
import gc

import torch
from time import time
import numpy as np
import pandas as pd
from torchtext import data
import torch.nn.functional as F

from Utils import allocate_gpu
from Utils.field import get_cvaetfencoder_fields, save_fields
# from Utils.dataset import get_dataloader, to_dataloader
from Model.modules import create_source_mask, create_target_mask
from Model.build_model import freeze_params, get_model
from Model.loss import LossCompute, KLDiv


class Batch:
    def __init__(self, src, trg_en, trg, device,
                 econds=None, dconds=None, mconds=None):
        # input of encoder
        self.src = src.to(device)
        self.trg_en = trg_en.to(device)
        # output of the decoder
        self.trg_y = trg[:, 1:].to(device)
        self.trg = trg[:, :-1].to(device)
        # conditions
        self.econds = econds.to(device)
        self.dconds = dconds.to(device)
        self.mconds = mconds.to(device)


def padding(obj, max_strlen, cond_len, pad_idx):
    obj_pad = torch.ones(obj.size(0), abs(max_strlen - obj.size(1)
                       - cond_len), dtype=torch.long) * pad_idx
    return torch.cat([obj, obj_pad], dim=1)


def rebatch(batch, conds, pad_idx, max_strlen, device):
    src = padding(batch.src.transpose(0,1), max_strlen, len(conds), pad_idx)
    trg = padding(batch.trg.transpose(0,1), max_strlen, len(conds), pad_idx)
    trg_en = padding(batch.trg_en.transpose(0,1), max_strlen, len(conds), pad_idx)

    econds = torch.zeros((len(batch),3))
    dconds = torch.zeros((len(batch),3))    
    mconds = torch.zeros((len(batch),6))

    for i, c in enumerate(conds):
        econds[:, i] = getattr(batch, f"src_{c}")
        dconds[:, i] = getattr(batch, f"trg_{c}")
    mconds[:, :3] = econds[:]
    mconds[:, 3:6] = torch.sub(econds, dconds)
    return Batch(src, trg_en, trg, device, econds, dconds, mconds)


def to_dataloader(data_iter, conditions, pad_idx, max_strlen, device):
    return (rebatch(batch, conditions, pad_idx, max_strlen, device)
            for batch in data_iter)


def KLAnnealer(epoch, KLA_ini_beta, KLA_inc_beta, KLA_beg_epoch):
    beta = KLA_ini_beta + KLA_inc_beta * ((epoch + 1) - KLA_beg_epoch)
    return beta


def loss_function(z_prediction, z_truth):
    return KLDiv()(z_prediction, z_truth)


def save_checkpoint(args, model, optimizer, save_path):
    hyper_param_names = ('N', 'd_model', 'd_ff', 'H', 'latent_dim',
                         'dropout', 'use_cond2dec', 'use_cond2lat',
                         'variational', 'nconds')
    hyper_param_values = {}
    for name in hyper_param_names:
        hyper_param_values[name] = getattr(args, name)
    save_dict = {
        'model_state_dict': model.state_dict(),
        'opt_state_dict': optimizer.state_dict(),
        'model_params': hyper_param_values
    }
    torch.save(save_dict, save_path)


def run_epoch(args, model, optimizer, dataloader, nbatches, LOG, train):
    n_samples = 0
    n_printevery = 1
    tot_loss = 0
    history = { 'LOSS': [] }    
    cost_time = -time()

    for i, batch in enumerate(dataloader):
        # dim of out: (batch_size, max_trg_seq_length-1, d_model)
        src_pad_mask = create_source_mask(batch.src, args.pad_id, batch.econds)
        trg_pad_mask = create_source_mask(batch.trg_en, args.pad_id, batch.dconds)
        # trg_mask = create_target_mask(trg_input, args.pad_id,
        #                               batch.dconds, args.use_cond2dec)

        trg_z_pred, trg_z_truth = model.forward(
            src=batch.src,
            trg_en=batch.trg_en,
            econds=batch.econds,
            mconds=batch.mconds,
            dconds=batch.dconds,
            src_pad_mask=src_pad_mask,
            trg_pad_mask=trg_pad_mask,
        )
        
        if train:
            optimizer.zero_grad()
        loss = loss_function(trg_z_pred, trg_z_truth)
        if train:
            loss.backward()
            optimizer.step()
        
        tot_loss += loss.item()        
        n_onebatch = len(batch.src)
        n_samples += n_onebatch

        history['LOSS'].append(loss.item() / n_onebatch)

        torch.cuda.empty_cache()

        total_cost_time = time() + cost_time

        details = f'{i+1}/{nbatches:<10}\t'                \
                  f'LOSS: {history["LOSS"][-1]:.5f}\t'     \
                  f'TIME(s): {total_cost_time:.1f}\t'      \

        if (i + 1) % n_printevery == 0:
            LOG.info(details)

    avg_loss = tot_loss / n_samples    
    return history, avg_loss
    

def train_model(args, model, optimizer, train_iter,
                valid_iter, device, LOG):

    for epoch in range(args.start_epoch, args.num_epoch+1):
        LOG.info(f'run epoch: {epoch}')

        LOG.info(f'training start, epoch: {epoch}')
        dataloader = to_dataloader(train_iter,
                                   args.conditions,
                                   args.pad_id,
                                   args.max_strlen,
                                   device)
        model.train()

        loss_history, avg_loss = run_epoch(
            args,
            model,
            optimizer,
            dataloader,
            args.train_nbatches,
            LOG,
            train=True
        )
        df = pd.DataFrame(loss_history)
        df.to_csv(os.path.join(args.save_directory, f'train_{epoch}.csv'))
        LOG.info(f'training end\tloss: {avg_loss}')

        LOG.info(f'validation start, epoch: {epoch}')
        dataloader = to_dataloader(valid_iter,
                                   args.conditions,
                                   args.pad_id,
                                   args.max_strlen,
                                   device)
        model.eval()
        with torch.no_grad():
            loss_history, avg_loss = run_epoch(
                args,
                model,
                optimizer,
                dataloader,
                args.valid_nbatches,
                LOG,
                train=False
            )
        df = pd.DataFrame(loss_history)
        df.to_csv(os.path.join(args.save_directory, f'valid_{epoch}.csv'))
        LOG.info(f'validation end\tloss: {avg_loss}')

        LOG.info(f'save model...')  
        model_path = os.path.join(args.save_directory, f'model_{epoch}.pt')
        save_checkpoint(args, model, optimizer, model_path)


def mlpcvaetf_encoder_train(args, logger):
    os.makedirs(args.save_directory, exist_ok=True)

    data_path = os.path.join(args.data_path, 'aug',
                             f'data_sim{args.similarity:.2f}_tol{args.tolerance:.2f}')
    # data_path = os.path.join(args.data_path, 'aug', 'data_sim0.80_tiny')

    LOG = logger(name='augment data by conditions',
                 log_path=os.path.join(args.save_directory, "records.log"))

    LOG.info('allocate GPU...')
    device = allocate_gpu()

    LOG.info('get feilds & SRC & TRG...')
    fields, SRC, TRG = get_cvaetfencoder_fields(args.conditions, args.molgct_path)

    LOG.info('prepare train & valid dataset...')
    train, valid = data.TabularDataset.splits(
        path=data_path, train='train_en.csv', validation='validation_en.csv', # change
        test=None, format='csv', fields=fields, skip_header=True
    )

    LOG.info('prepare train & valid dataset...')
    train_iter, valid_iter = data.BucketIterator.splits(
        (train, valid), batch_sizes=(args.batch_size, args.batch_size),
        sort_key=lambda x: (len(x.src), len(x.trg))
    )

    LOG.info(f'# train: {len(train)}, # validation: {len(valid)}')
    if args.load_field is False:
        SRC.build_vocab(train)
        TRG.build_vocab(valid)
        save_fields(SRC, TRG, args.molgct_path)

    args.train_nbatches = int(np.ceil(len(train) / args.batch_size))
    args.valid_nbatches = int(np.ceil(len(valid) / args.batch_size))

    args.sos_id = TRG.vocab.stoi['<sos>']
    args.eos_id = TRG.vocab.stoi['<eos>']
    args.pad_id = SRC.vocab.stoi['<pad>']

    assert SRC.vocab.stoi['<pad>'] == TRG.vocab.stoi['<pad>']

    del train, valid
    gc.collect()

    LOG.info('prepare model...')
    
    model = get_model(args, len(SRC.vocab), len(TRG.vocab))

    for name, params in model.named_parameters():
        print(name, params.size())    

    model = model.to(device)

    if args.train_params:
        freeze_params(model, train_names=args.train_params)

    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    LOG.info(f'# total params: {total_params}, # train params: {train_params}')

    assert train_params > 0, '# trainable parameters = 0'

    LOG.info(f'Get optimizer...')

    # model_folder, model_path, start_epoch
        
    model_train_params = filter(lambda p: p.requires_grad, model.parameters())
    
    if args.optimizer_choice == 'sgd':
        optimizer = torch.optim.SGD(model_train_params, lr=1E-4)
    elif args.optimizer_choice == 'rmsprop':
        optimizer = torch.optim.RMSprop(model_train_params, lr=1E-4)
    elif args.optimizer_choice == 'adagrad':
        optimizer = torch.optim.Adagrad(model_train_params, lr=1E-4)
    elif args.optimizer_choice == 'adam':
        optimizer = torch.optim.Adam(model_train_params, lr=args.lr,
            betas=(args.lr_beta1, args.lr_beta2), eps=args.lr_eps
        )
    elif args.optimizer_choice == 'original':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
            betas=(args.lr_beta1, args.lr_beta2), eps=args.lr_eps
        )

    if args.use_model_path and args.optimizer_choice == 'original':
        checkpoint = torch.load(args.use_model_path, map_location='cuda:0')
        optim_dict = checkpoint['opt_state_dict']
        optimizer.load_state_dict(optim_dict)
            
    LOG.info(f'train model...')
    
    train_model(args, model, optimizer, train_iter,
                valid_iter, device, LOG)


# import os
# import gc

# from numpy import ceil
# from time import time

# import torch
# from torchtext import data

# from Train.mlpcvae_trainer import MLPCVAE_Trainer
# from Train.att_trainer import ATT_Trainer
# from Model.modules import NoamOpt as moptim
# from Utils import set_seed, allocate_gpu, get_fields, save_fields
# from Model.build_model import build_model


# def train(args, debug=False):
#     set_seed(100)

#     print('Getting GPU')
#     device = allocate_gpu()    

#     print('Getting feilds / SRC / TRG')

#     fields, SRC, TRG = get_fields(args.conditions, args.molgct_path)

#     print('Preparing training / validation dataset')

#     train_data, valid_data = data.TabularDataset.splits(
#         path=os.path.join(args.data_path, 'aug', f'data_sim{args.similarity:.2f}'),
#         train='train.csv', validation='validation.csv', test=None, format='csv',
#         fields=fields, skip_header=True
#     )

#     print(f'#pairs in training / validation dataset: {len(train_data)}/{len(valid_data)}')

#     if args.load_field is False:
#         SRC.build_vocab(train_data)
#         TRG.build_vocab(valid_data)
#         save_fields(SRC, TRG, args.molgct_path)

#     args.train_nbatches, args.valid_nbatches = int(ceil(len(train_data) / args.batch_size)), \
#                                                int(ceil(len(valid_data) / args.batch_size))

#     print('Preparing training / validation dataloader')

#     train_iter, valid_iter = data.BucketIterator.splits(
#         (train_data, valid_data), batch_sizes=(args.batch_size, args.batch_size),
#         sort_key=lambda x: (len(x.src), len(x.trg))
#     )

#     del train_data, valid_data
#     gc.collect()

#     args.sos_idx = TRG.vocab.stoi['<sos>']
#     args.eos_idx = TRG.vocab.stoi['<eos>']
#     args.pad_idx = SRC.vocab.stoi['<pad>']

#     assert SRC.vocab.stoi['<pad>'] == TRG.vocab.stoi['<pad>']

#     print(f'Preparing model with starting epoch: {args.start_epoch}')

#     if args.model_type == 'att_encoder':
#         model = build_model(args, len(SRC.vocab), len(TRG.vocab), att_type='ATT_v5').to(device)
#     elif args.model_type == 'mlp_encoder':
#         model = build_model(args, len(SRC.vocab), len(TRG.vocab)).to(device)

#     # for n, p in model.named_parameters():
#     #     if p.requires_grad:
#     #         print(n, p.size())

#     total_parameters = sum(p.numel() for p in model.parameters())
#     trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

#     assert trainable_parameters > 0

#     print('Parameters / Trainable Parameters:', f'{total_parameters:<11}/{trainable_parameters:<11}\t')
    
#     if args.model_type == 'att_encoder':
#         trainer = ATT_Trainer(args.conditions, args.save_directory, args.pad_idx, args.max_strlen)
#         trainer.train(args, model, train_iter, valid_iter, SRC, TRG, device)
#     elif args.model_type == 'mlp_encoder':
#         trainer = MLPCVAE_Trainer(args)
#         trainer.train(model, train_iter, valid_iter, SRC, TRG, device)


# def mlpcvae_train(args, logger):
#     pass