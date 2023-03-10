import os
import gc

import torch
from time import time
import numpy as np
import pandas as pd
from torchtext import data
import torch.nn.functional as F

from Utils import allocate_gpu
from Utils.field import save_fields, get_tf_fields
from Utils.dataset import get_loader
from Model.modules import create_source_mask, create_target_mask
from Model.build_model import freeze_params, get_model


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
                         'variational')
    hyper_param_values = { 'nconds': len(args.property_list) }
    for name in hyper_param_names:
        hyper_param_values[name] = getattr(args, name)

    save_dict = {
        'model_state_dict': model.state_dict(),
        'opt_state_dict': optimizer.state_dict(),
        'model_params': hyper_param_values
    }
    torch.save(save_dict, save_path)


def run_epoch(args, model, optimizer, dataloader,
              current_step, beta, LOG, train):
    n_samples = 0
    n_printevery = 1
    tot_loss = tot_rce = tot_kld = 0
    history = { 'RCE': [], 'KLD': [], 'LOSS': [], 'BETA': [], 'LR': [] }
    
    model_cost_time = update_cost_time = 0
    cost_time = -time()

    for i, batch in enumerate(dataloader):
        current_step += 1

        trg_input = batch['trg'][:, :-1]

        # dim of out: (batch_size, max_trg_seq_length-1, d_model)
        src_mask = create_source_mask(batch['src'],
                                      args.pad_id,
                                      batch['econds'])
        trg_mask = create_target_mask(trg_input,
                                      args.pad_id,
                                      batch['dconds'],
                                      args.use_cond2dec)

        model_cost_time -= time()
        preds_prop, preds_mol, mu, log_var, _ = model.forward(
            src=batch['src'],
            trg=trg_input,
            econds=batch['econds'],
            mconds=batch['mconds'],
            dconds=batch['dconds'],
            src_mask=src_mask,
            trg_mask=trg_mask,
        )

        model_cost_time += time()

        ys_mol = batch['trg'][:, 1:].contiguous().view(-1)
        ys_cond = torch.unsqueeze(batch['dconds'], 2).contiguous(
        ).view(-1, len(args.property_list), 1)

        update_cost_time -= time()
        
        if train:
            optimizer.zero_grad(set_to_none=True)

        loss, RCE_mol, RCE_prop, KLD = loss_function(beta, preds_prop, preds_mol,
                                                     ys_cond, ys_mol, mu, log_var,
                                                     args.use_cond2dec, args.pad_id)
        if train:
            loss.backward()
            optimizer.step()

        update_cost_time += time()

        # if args.optimizer_choice == 'original' and args.lr_scheduler == "WarmUpDefault":
        #     k1, k2, k3 = 0.5, 1.5, 0.5 # original
        #     # k1, k2, k3 = 0.6, 1.6, 0.5 # train decoder & out
        #     head = np.float(np.power(np.float(current_step), -k1))
        #     tail = np.float(current_step) * \
        #         np.power(np.float(args.lr_WarmUpSteps), -k2)
        #     lr = np.float(np.power(np.float(args.d_model), -k3)) * \
        #          min(head, tail)

        # if args.optimizer_choice == 'original' and train:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr

        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
        
        tot_loss += loss.item()
        tot_rce += RCE_mol.item()
        tot_kld += KLD.item()
        
        n_onebatch = len(batch['src'])
        n_samples += n_onebatch
        
        history['RCE'].append(RCE_mol.item() / n_onebatch)
        history['KLD'].append(KLD.item() / n_onebatch)
        history['LOSS'].append(loss.item() / n_onebatch)
        history['BETA'].append(beta)        
        history['LR'].append(current_lr)

        torch.cuda.empty_cache()

        total_cost_time = time() + cost_time

        details = f'{i+1}/{len(dataloader):<10}\t'                \
                  f'RCE: {history["RCE"][-1]:.5f}\t'       \
                  f'KLD: {history["KLD"][-1]:.5f}\t'       \
                  f'LOSS: {history["LOSS"][-1]:.5f}\t'     \
                  f'TIME(s): {total_cost_time:.1f}\t'      \
                  f'MODELTIME(s): {model_cost_time:.1f}\t' \
                  f'UPDATETIME(s): {update_cost_time:.1f}' \

        if (i + 1) % n_printevery == 0:
            LOG.info(details)
            # print(details)

    avg_loss = tot_loss / n_samples
    avg_rce = tot_rce / n_samples
    avg_kld = tot_kld / n_samples
    
    if train: # training phase
        return history, (avg_loss, avg_rce, avg_kld), current_step
    else: # validation phase
        return history, (avg_loss, avg_rce, avg_kld)


def train_model(args, model, optimizer, train_loader, valid_loader,
                device, rank, LOG):
    beta = 0
    # current_step = args.use_epoch * args.train_nbatches
    current_step = (args.start_epoch-1) * args.train_nbatches

    for epoch in range(args.start_epoch, args.num_epoch+1):
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

        # dataloader = get_loader(
        #     train_iter, args.property_list, args.pad_id, args.max_strlen,
        #     args.pad_to_same_len, device, include_mconds=True
        # )
        model.train()

        loss_history, avg_loss, current_step = run_epoch(args,
                                                         model,
                                                         optimizer,
                                                         train_loader,
                                                         current_step,
                                                         beta,
                                                         LOG,
                                                         train=True
                                                         )

        df = pd.DataFrame(loss_history)
        df.to_csv(os.path.join(args.model_folder, f'train_{epoch}.csv'))
        LOG.info(f'training end\tloss: {avg_loss[0]},\t'
                 f'RCE: {avg_loss[1]},\tKLD: {avg_loss[2]}')

        LOG.info(f'validation start, epoch: {epoch}')

        # dataloader = get_loader(
        #     valid_iter, args.property_list, args.pad_id, args.max_strlen,
        #     args.pad_to_same_len, device, include_mconds=True
        # )

        model.eval()
        with torch.no_grad():
            loss_history, avg_loss = run_epoch(args,
                                               model,
                                               optimizer,
                                               valid_loader,
                                               current_step,
                                               beta,
                                               LOG,
                                               train=False
                                               )
        df = pd.DataFrame(loss_history)
        df.to_csv(os.path.join(args.model_folder, f'valid_{epoch}.csv'))
        LOG.info(f'validation end\tloss: {avg_loss[0]},\t'
                 f'RCE: {avg_loss[1]},\tKLD: {avg_loss[2]}')

        LOG.info(f'save model...')  
        model_path = os.path.join(args.model_folder, f'model_{epoch}.pt')

        if rank == 0:
            save_checkpoint(args, model, optimizer, model_path)

