import os
import gc

import torch
from time import time
import numpy as np
import pandas as pd
from functools import reduce
from torchtext import data
import torch.nn.functional as F
import torch.distributed as dist
from Utils.dataset import get_loader
from Model.modules import create_source_mask, create_target_mask


def reduce_val(val, size, method):
    group = dist.new_group(list(range(size)))
    tensor = torch.tensor([val])

    if method == 'sum':
        # result of reduction is stored in tensor and return None
        dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM, group=group)
    elif method == 'avg':
        dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM, group=group)
        tensor /= size
    return tensor.item()


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
    hyper_param_values = {'nconds': len(args.property_list)}
    for name in hyper_param_names:
        hyper_param_values[name] = getattr(args, name)

    try:
        model_state_dict = model.module.state_dict()
    except AttributeError:
        model_state_dict = model.state_dict()
    # ref: https://github.com/pytorch/pytorch/issues/9176

    save_dict = {
        'model_state_dict': model_state_dict,
        'opt_state_dict'  : optimizer.state_dict(),
        'model_params'    : hyper_param_values
    }
    torch.save(save_dict, save_path)


def run_epoch(args, model, optimizer, dataloader,
              current_step, beta, LOG, train):

    n_samples = 0
    n_printevery = 1
    history = {'RCE': [], 'KLD': [], 'LOSS': [], 'BETA': [], 'LR': []}

    model_cost_time = update_cost_time = 0
    cost_time = -time()

    for i, batch in enumerate(dataloader):
        current_step += 1

        trg_input = batch['trg'][:, :-1]

        # dim of out: (batch_size, max_trg_seq_length-1, d_model)
        src_mask = create_source_mask(
            batch['src'], args.pad_id, batch['econds'])
        trg_mask = create_target_mask(
            trg_input, args.pad_id, batch['dconds'],
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

        loss, RCE_mol, RCE_prop, KLD = loss_function(beta, preds_prop, preds_mol, ys_cond, ys_mol, mu,
                                                     log_var, args.use_cond2dec, args.pad_id)

        if train:
            loss.backward()
            optimizer.step()

        update_cost_time += time()

        if args.lr_scheduler == "WarmUpDefault":
            k1, k2, k3 = 0.5, 1.5, 0.5
            head = np.float(np.power(np.float(current_step), -k1))
            tail = np.float(current_step) * \
                np.power(np.float(args.lr_WarmUpSteps), -k2)
            lr = np.float(np.power(np.float(args.d_model), -k3)) * \
                min(head, tail)

        if train:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']

        n_onebatch = len(batch['src'])
        n_samples += n_onebatch

        history['RCE'].append(RCE_mol.item() / n_onebatch)
        history['KLD'].append(KLD.item() / n_onebatch)
        history['LOSS'].append(loss.item() / n_onebatch)
        history['BETA'].append(beta)
        history['LR'].append(current_lr)

        torch.cuda.empty_cache()

        total_cost_time = time() + cost_time

        details = f'{i+1}/{len(dataloader):<10}\t'         \
                  f'RCE: {history["RCE"][-1]:.5f}\t'       \
                  f'KLD: {history["KLD"][-1]:.5f}\t'       \
                  f'LOSS: {history["LOSS"][-1]:.5f}\t'     \
                  f'TIME(s): {total_cost_time:.1f}\t'      \
                  f'MODELTIME(s): {model_cost_time:.1f}\t' \
                  f'UPDATETIME(s): {update_cost_time:.1f}' \

        if (i + 1) % n_printevery == 0:
            LOG.info(details)
            # print(details)

    if train:  # training phase
        return history,  current_step
    else:  # validation phase
        return history


def set_barrier(world_size):
    if world_size > 1:
        return


def train_model(args, model, optimizer, train_loader, valid_loader,
                rank, world_size, LOG):
    beta = 0
    current_step = (args.start_epoch-1) * len(train_loader)

    for epoch in range(args.start_epoch, args.num_epoch+1):
        if world_size > 1:
            train_loader.sampler.set_epoch(epoch)
        # randomize the dataloader each epoch

        LOG.info(f'run epoch: {epoch}')

        if args.use_KLA == True:
            if epoch + 1 >= args.KLA_beg_epoch and beta < args.KLA_max_beta:
                beta = KLAnnealer(epoch,
                                  args.KLA_ini_beta,
                                  args.KLA_inc_beta,
                                  args.KLA_beg_epoch)
        else:
            beta = 1
        # KL annealing

        LOG.info(f'training start, epoch: {epoch}')

        if world_size > 1:
            dist.barrier()  # blocking

        model.train()

        train_history, current_step = run_epoch(args,
                                                model, optimizer, train_loader, current_step,
                                                beta, LOG, train=True)

        LOG.info('Save training results...')

        df_train = pd.DataFrame(train_history)
        if world_size > 1:
            df_train.to_csv(os.path.join(args.model_folder,
                            f'train_{epoch}_r{rank}.csv'))
        else:
            df_train.to_csv(os.path.join(args.model_folder,
                            f'train_{epoch}.csv'))

        LOG.info(f'validation start, epoch: {epoch}')

        if world_size > 1:
            dist.barrier()  # blocking

        model.eval()

        with torch.no_grad():
            valid_history = run_epoch(args,
                                      model, optimizer, valid_loader, current_step,
                                      beta, LOG, train=False)

        LOG.info('Save validation results...')

        df_valid = pd.DataFrame(valid_history)
        if world_size > 1:
            df_valid.to_csv(os.path.join(args.model_folder,
                            f'valid_{epoch}_r{rank}.csv'))
        else:
            df_valid.to_csv(os.path.join(args.model_folder,
                            f'valid_{epoch}.csv'))

        if world_size > 1:
            dist.barrier()  # blocking

        if rank == 0:
            LOG.info('Save model...')

            model_path = os.path.join(args.model_folder, f'model_{epoch}.pt')
            save_checkpoint(args, model, optimizer, model_path)

        if rank == 0:
            for data_type in ('train', 'valid'):
                his_list = []
                for rank_i in range(world_size):
                    res_path = os.path.join(args.model_folder,
                                            f'{data_type}_{epoch}_r{rank_i}.csv')
                    his_list.append(pd.read_csv(res_path, index_col=[0]))
                    if not args.debug:
                        os.remove(res_path)

                df_rce = reduce(
                    lambda x, y: x[['RCE']] + y[['RCE']], his_list) / world_size
                df_kld = reduce(
                    lambda x, y: x[['KLD']] + y[['KLD']], his_list) / world_size
                df_loss = reduce(
                    lambda x, y: x[['LOSS']] + y[['LOSS']], his_list) / world_size
                df = pd.concat([df_rce, df_kld, df_loss,
                                his_list[0]['BETA'],
                                his_list[0]['LR']], axis=1)

                df.to_csv(os.path.join(args.model_folder,
                          f'{data_type}_{epoch}.csv'))

        if world_size > 1:
            dist.barrier()  # blocking
