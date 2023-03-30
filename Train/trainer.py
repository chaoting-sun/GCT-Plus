import os
import torch
import numpy as np
import pandas as pd
from time import time
import torch.nn.functional as F
import torch.distributed as dist
import gc
from functools import reduce
from Utils.dataset import get_loader
from Model.modules import create_source_mask, create_target_mask
# from GPUtil import showUtilization as gpu_usage
# from torch.cuda.amp import GradScaler, autocast


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

    save_dict = {
        'model_state_dict': model.state_dict(),
        'opt_state_dict': optimizer.state_dict(),
        'model_params': hyper_param_values
    }
    torch.save(save_dict, save_path)


def decode_check(preds_mol, TRG):
    from Utils.smiles import get_mol
    
    prob = F.softmax(preds_mol, dim=-1)
    decoded_strings = []
    valid_mols = 0
    for batch_id in range(prob.size(0)):
        decoded_str = []
        for position_id in range(prob.size(1)):
            token_id = torch.multinomial(prob[batch_id, position_id], 1)[0]
            if token_id == TRG.vocab.stoi['<eos>']:
                break
            decoded_char = TRG.vocab.itos[token_id]
            decoded_str.append(decoded_char)
        smiles = "".join(decoded_str)
        decoded_strings.append(smiles)
        if get_mol(smiles) is not None:
            valid_mols += 1
    print('valid ratio (%):', valid_mols / prob.size(0) * 100)
    return decoded_strings
        
        
def run_epoch(args, model, optimizer, dataloader, current_step,
              beta, LOG, world_size, train):
    n_samples = 0
    n_printevery = 1
    history = {'RCE': [], 'KLD': [], 'LOSS': [], 'BETA': [], 'LR': []}
    model_cost_time = update_cost_time = 0
    cost_time = -time()

    # Create a GradScaler for mixed precision training
    # if train:
    #     scaler = GradScaler()

    for i, batch in enumerate(dataloader):
        current_step += 1
        n_onebatch = batch['src'].size(0)
        trg_input = batch['trg'][:, :-1]
        
        model_cost_time -= time()
        
        # Handle different model types
        if args.model_type in ('cvaetf', 'scacvaetfv1', 'scacvaetfv3'):
            src_mask = create_source_mask(
                batch['src'], args.pad_id, batch['econds'])
            trg_mask = create_target_mask(
                trg_input, args.pad_id, batch['dconds'],
                args.use_cond2dec)

            # with autocast():
            preds_prop, preds_mol, mu, log_var, _ = model.forward(
                src=batch['src'],
                trg=trg_input,
                econds=batch['econds'],
                dconds=batch['dconds'],
                src_mask=src_mask,
                trg_mask=trg_mask,
            )
        
        elif args.model_type == 'scacvaetfv2':
            src_enc_mask = create_source_mask(
                torch.cat((batch['src_scaffold'], batch['src']), 1),
                args.pad_id, batch['econds'])
            # src_enc_mask: (bs, 1, nc+sca+src)
            src_dec_mask = create_source_mask(
                batch['trg_scaffold'],
                args.pad_id, batch['dconds'])
            # src_dec_mask: (bs, 1, nc+<sos>sca<eos>)
            trg_mask = create_target_mask(
                trg_input, args.pad_id, batch['dconds'],
                args.use_cond2dec)

            # with autocast():
            preds_prop, preds_mol, mu, log_var, _ = model.forward(
                src=batch['src'],
                trg=trg_input,
                src_scaffold=batch['src_scaffold'],
                trg_scaffold=batch['trg_scaffold'],
                econds=batch['econds'],
                dconds=batch['dconds'],
                src_enc_mask=src_enc_mask,
                src_dec_mask=src_dec_mask,
                trg_mask=trg_mask,
            )
            
        model_cost_time += time()

        ys_mol = trg_input.contiguous().view(-1)
        ys_cond = torch.unsqueeze(batch['dconds'], 2).contiguous(
            ).view(-1, len(args.property_list), 1)
        
        update_cost_time -= time()
        
        if train:
            optimizer.zero_grad(set_to_none=True)    

        # Calculate loss
        # with autocast():
        loss, RCE_mol, RCE_prop, KLD = loss_function(
            beta, preds_prop, preds_mol, ys_cond, ys_mol,
            mu, log_var, args.use_cond2dec, args.pad_id)
        
        # Update gradients and optimizer for training
        if train:
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            loss.backward()
            optimizer.step()
            
        update_cost_time += time()

        # Learning rate scheduler
        if args.lr_scheduler == "WarmUpDefault":
            k1, k2, k3 = 0.5, 1.5, 0.5
            head = np.float(np.power(np.float(current_step), -k1))
            tail = np.float(current_step) * \
                np.power(np.float(args.lr_WarmUpSteps), -k2)
            lr = np.float(np.power(np.float(args.d_model), -k3)) * \
                      min(head, tail)
            # lr = base_lr*np.sqrt(int(world_size*(args.batch_size/128)))
        
        if train:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']

        n_samples += n_onebatch

        history['RCE'].append(RCE_mol.item() / n_onebatch)
        history['KLD'].append(KLD.item() / n_onebatch)
        history['LOSS'].append(loss.item() / n_onebatch)
        # history['LOSS'].append(loss.detach().item() / n_onebatch)
        history['BETA'].append(beta)
        history['LR'].append(current_lr)

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

    if train:  # training phase
        return history,  current_step
    else:  # validation phase
        return history


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
                                                beta, LOG, world_size, train=True)

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
                    # if not args.debug:
                    #     os.remove(res_path)

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
