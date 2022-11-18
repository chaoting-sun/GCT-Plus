import os
import gc

import torch
from time import time
import numpy as np
import pandas as pd
from torchtext import data
import torch.nn.functional as F

from Model.build_model import build_model, freeze_params
from Utils import allocate_gpu, save_fields
from Utils.field import get_tf_fields
from Utils.dataset import get_dataloader
from Model.modules import create_source_mask, create_target_mask

# from Train.tf_trainer import Trainer


def KLAnnealer(epoch, KLA_ini_beta, KLA_inc_beta, KLA_beg_epoch):
    beta = KLA_ini_beta + KLA_inc_beta * ((epoch + 1) - KLA_beg_epoch)
    return beta


def loss_function(beta, preds_prop, preds_mol, ys_cond,
                  ys_mol, mu, log_var, use_cond2dec, pad_id):
    RCE_mol = F.cross_entropy(preds_mol.contiguous().view(-1, preds_mol.size(-1)),
                              ys_mol, ignore_index=pad_id, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    if use_cond2dec == True:
        RCE_prop = F.mse_loss(preds_prop, ys_cond, reduction='sum')
        loss = RCE_mol + RCE_prop + beta * KLD
    else:
        RCE_prop = torch.zeros(1)
        loss = RCE_mol + beta * KLD
    return loss, RCE_mol, RCE_prop, KLD


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


def run_epoch(args, model, optimizer, dataloader,
              current_step, beta, nbatches, LOG):
    n_samples = 0
    n_printevery = 1
    tot_loss = tot_rce = tot_kld = 0
    history = {'RCE': [], 'KLD': [], 'LOSS': [],
               'lr': [], 'beta': []}
    
    model_cost_time = update_cost_time = 0
    cost_time = -time()

    for i, batch in enumerate(dataloader):
        current_step += 1

        trg_input = batch.trg[:, :-1]

        # dim of out: (batch_size, max_trg_seq_length-1, d_model)
        src_mask = create_source_mask(batch.src, args.pad_id, batch.econds)
        trg_mask = create_target_mask(trg_input, args.pad_id,
                                      batch.dconds, args.use_cond2dec)

        model_cost_time -= time()
        preds_prop, preds_mol, mu, log_var, _ = model.forward(batch.src,
                                                              trg_input,
                                                              batch.econds,
                                                              batch.dconds,
                                                              src_mask,
                                                              trg_mask
                                                              )
        model_cost_time += time()

        ys_mol = batch.trg[:, 1:].contiguous().view(-1)
        ys_cond = torch.unsqueeze(batch.dconds, 2).contiguous(
        ).view(-1, len(args.conditions), 1)

        update_cost_time -= time()
        optimizer.zero_grad()
        loss, RCE_mol, RCE_prop, KLD = loss_function(beta, preds_prop, preds_mol,
                                                     ys_cond, ys_mol, mu, log_var,
                                                     args.use_cond2dec, args.pad_id)
        loss.backward()
        optimizer.step()
        update_cost_time += time()

        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
        
        tot_rce += RCE_mol.item()
        tot_kld += KLD.item()
        tot_loss += loss.item()
        
        history['RCE'].append(RCE_mol.item())
        history['KLD'].append(KLD.item())
        history['LOSS'].append(loss.item())
        history['lr'].append(current_lr)
        history['beta'].append(beta)

        # modify learning rate
        head = np.float(np.power(np.float(current_step), -0.5))
        tail = np.float(current_step) * \
            np.power(np.float(args.lr_WarmUpSteps), -1.5)
        lr = np.float(np.power(np.float(args.d_model), -0.5)) * min(head, tail)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # torch.cuda.empty_cache()

        n_onebatch = len(batch.src)
        n_samples += n_onebatch
        total_cost_time = time() + cost_time

        details = f'{i+1}/{nbatches:<10}\t' \
                  f'MSE: {history["RCE"][-1]/n_onebatch:.5f}\t'  \
                  f'KLD: {history["RCE"][-1]/n_onebatch:.5f}\t'  \
                  f'LOSS: {history["RCE"][-1]/n_onebatch:.5f}\t' \
                  f'TIME(s): {total_cost_time:.1f}\t' \
                  f'MODELTIME(s): {model_cost_time:.1f}\t' \
                  f'UPDATETIME(s): {update_cost_time:.1f}' \

        if (i + 1) % n_printevery == 0:
            LOG.info(details)
            # print(details)

    avg_loss = tot_loss / n_samples
    avg_rce = tot_rce / n_samples
    avg_kld = tot_kld / n_samples
    return history, (avg_loss, avg_rce, avg_kld)
    

def train_model(args, model, optimizer, train_iter,
                valid_iter, data_path, device, LOG):
    beta = current_step = 0

    for epoch in range(args.use_epoch+1, args.num_epoch+1):
        LOG.info(f'run epoch: {epoch}')

        LOG.info('KL anealing...')
        if args.use_KLA == True:
            if epoch + 1 >= args.KLA_beg_epoch and beta < args.KLA_max_beta:
                beta = KLAnnealer(epoch, args.KLA_ini_beta,
                                  args.KLA_inc_beta, args.KLA_beg_epoch)
        else:
            beta = 1
        LOG.info(f'beta: {beta:4f}')

        LOG.info(f'training start, epoch: {epoch}')
        
        model.train()
        dataloader = get_dataloader(train_iter, args.conditions,
                                    args.pad_id, device)
        train_his, avg_loss = run_epoch(args, model, optimizer, dataloader,
                              current_step, beta, args.train_nbatches, LOG)
        df = pd.DataFrame(train_his)
        df.to_csv(os.path.join(data_path, f'train_{epoch}.csv'))
        
        LOG.info(f'training end\tloss: {avg_loss[0]},\t'
                 f'RCE: {avg_loss[1]},\tKLD: {avg_loss[2]}')

        LOG.info(f'validation start, epoch: {epoch}')
        
        model.eval()
        dataloader = get_dataloader(valid_iter, args.conditions,
                                    args.pad_id, device)
        valid_hist, avg_loss = run_epoch(args, model, optimizer, dataloader,
                               current_step, beta, args.valid_nbatches, LOG)
        df = pd.DataFrame(valid_hist)
        df.to_csv(os.path.join(data_path, f'valid_{epoch}.csv'))
        
        LOG.info(f'validation end\tloss: {avg_loss[0]},\t'
                 f'RCE: {avg_loss[1]},\tKLD: {avg_loss[2]}')

        LOG.info(f'save model...')
        save_path = os.path.join(args.model_path, f'model_{epoch}.pt')
        save_checkpoint(args, model, optimizer, save_path)


def tf_train(args, logger):
    os.makedirs(args.model_path, exist_ok=True)
    
    # data_path = os.path.join(args.data_path, 'aug',
    #                          f'data_tol{args.tolerance:.2f}')
    data_path = os.path.join(args.data_path, 'aug',
                             f'data_tol{args.tolerance:.2f}')

    LOG = logger(name='augment data by conditions',
                 log_path=os.path.join(args.model_path, "records.log"))

    LOG.info('allocate GPU...')
    device = allocate_gpu()

    LOG.info('get feilds & SRC & TRG...')
    fields, SRC, TRG = get_tf_fields(args.conditions, args.molgct_path)

    LOG.info('prepare train & valid dataset...')
    train, valid = data.TabularDataset.splits(
        path=data_path, train='train.csv', validation='validation.csv',
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

    LOG.info(f'prepare model with start epoch: {args.use_epoch}')
    
    model = build_model(args, len(SRC.vocab), len(TRG.vocab))
    model = model.to(device)

    freeze_params(model, train_names=['decoder', 'out'])

    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    LOG.info(f'# total params: {total_params}, # train params: {train_params}')

    assert train_params > 0, f'# trainable parameters = 0'

    LOG.info(f'Get optimizer. Starting epochs is {args.use_epoch}')

    if args.use_epoch == 1:
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr, betas=(args.lr_beta1, args.lr_beta2), eps=args.lr_eps
        )
    else:
        cp_path = os.path.join(args.model_path,
                               f'model_{args.use_epoch-1}.pt')
        checkpoint = torch.load(cp_path, map_location='cuda:0')
        optim_dict = checkpoint['opt_state_dict']
        optimizer.load_state_dict(optim_dict)
        
    LOG.info(f'train model...')
    
    # for name, params in model.named_parameters():
    #     print(name, params.size())

    # exit()

    train_model(args, model, optimizer, train_iter,
                valid_iter, data_path, device, LOG)



    