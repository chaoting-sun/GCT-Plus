import os
import gc
import numpy as np
import time
from timeit import default_timer as timer
from datetime import timedelta
from Utils.field import condition_fields
# import pandas as pd
# import pickle as pkl
# from itertools import tee

import torch
import torch.nn as nn
import torch.nn.functional as F
from Tokenize import moltokenize
# import torch.nn as nn


# from Model.mlp_transformer import decode
from Model.mlp_encoder import decode
from Model.loss import LossCompute, MSE_KLDiv, MSE, KLDiv, JSD
from Model.modules import NoamOpt as moptim
from Utils.dataset import to_dataloader
from Utils.log import get_logger
from Model.modules import create_source_mask
# import Process.data_preparation as pdp
# import Process.vocabulary as mv

DEBUG = True
N_SAMPLES = 80 if DEBUG is True else None


class ATT_Trainer(object):
    def __init__(self, conditions, save_directory, pad_idx, max_strlen):
        self.pad_idx = pad_idx
        self.conditions = conditions
        self.max_strlen = max_strlen

        os.makedirs(save_directory, exist_ok=True)
        self.results_path = os.path.join(save_directory, 'train_results.log')
        self.details_path = os.path.join(save_directory, 'train_details.log')

        self.LOG_results = get_logger(name="train_results",
                                      log_path=self.results_path)
        self.LOG_details = get_logger(name="train_details", 
                                      log_path=self.details_path)
    
    def get_optimization(self, args, trainable_parameters, opt_name):
        """
        name: opt_state_dict_mu or opt_state_dict_logvar
        """
        if args.start_epoch == 1:
            optimizer = torch.optim.Adam(trainable_parameters, lr=0,
                                         betas=(args.adam_beta1,
                                                args.adam_beta2),
                                         eps=args.adam_eps)
            optim = moptim(args.d_model, args.factor,
                           args.warmup_steps, optimizer)
        else:
            checkpoint = torch.load(os.path.join(args.save_directory,
                                    f'model_{args.start_epoch-1}.pt'),
                                    map_location='cuda:0')
            optim_dict = checkpoint[opt_name]
            optim = moptim(optim_dict['model_size'],
                           optim_dict['factor'], 
                           optim_dict['warmup'],
                           torch.optim.Adam(trainable_parameters, lr=0))
            optim.load_state_dict(optim_dict)
        return optim


    def save_checkpoint(self, args, model, model_name, opt_mu, opt_logvar,
                        src_vocab_size, trg_vocab_size):
        os.makedirs(self.save_directory, exist_ok=True)
        file_name = os.path.join(self.save_directory, model_name)

        model_param_names = [
            'N', 'H', 'd_ff', 'nconds', 'd_model',
            'dropout', 'latent_dim', 'variational',
            'use_cond2dec', 'use_cond2lat'
        ]

        model_params = { name: getattr(args, name) for name in model_param_names }
        model_params['src_vocab_size'] = src_vocab_size
        model_params['trg_vocab_size'] = trg_vocab_size

        save_dict = {
            'model_state_dict': model.state_dict(),
            'opt_state_dict_mu': opt_mu.save_state_dict(),
            'opt_state_dict_logvar': opt_logvar.save_state_dict(),
            'model_parameters': model_params
        }
        torch.save(save_dict, file_name)


    def run_epoch(self, model, data_iter, nbatches, TRG,
                  compute_mu_loss, compute_logvar_loss, device):
        """
        The following variables records the total values from the dataset
        """
        n_samples = 0
        sum_mu_mse = sum_logvar_mse = sum_mu_kld = sum_logvar_kld = 0
        total_model_time = total_update_time = total_clear_time = 0

        start_time = timer()

        def compute_loss(pred, truth):
            mse, kld = MSE(), KLDiv()
            return mse(pred, truth), kld(pred, truth)

        dataloader = to_dataloader(data_iter, self.conditions, self.pad_idx,
                                   self.max_strlen, device)

        for i, batch in enumerate(dataloader):
            # dim of out: (batch_size, max_trg_seq_length-1, d_model)
            total_model_time -= timer()
            
            src_pad_mask = create_source_mask(batch.src, self.pad_idx, batch.econds)
            trg_pad_mask = create_source_mask(batch.trg_en, self.pad_idx, batch.dconds)

            mu_pred, mu_truth, logvar_pred, logvar_truth = model.forward(batch.src,
                                                                         batch.trg_en,
                                                                         batch.econds,
                                                                         batch.mconds,
                                                                         batch.dconds,
                                                                         src_pad_mask,
                                                                         trg_pad_mask
                                                                         )
            total_model_time += timer()

            # Compute loss (rec-loss + KL-div) and update
            total_update_time -= timer()

            _mu_mse, _mu_kld = compute_loss(mu_pred, mu_truth)
            _logvar_mse, _logvar_kld = compute_loss(logvar_pred, logvar_truth)

            compute_mu_loss(mu_pred, mu_truth)
            compute_logvar_loss(logvar_pred, logvar_truth)

            total_update_time += timer()

            total_clear_time -= timer()
            torch.cuda.empty_cache()
            total_clear_time += timer()
            
            sum_mu_mse += float(_mu_mse)
            sum_logvar_mse += float(_logvar_mse)
            sum_mu_kld += float(_mu_kld)
            sum_logvar_kld += float(_logvar_kld)

            n_samples += len(batch.src)

            end_time = timer()

            details = f'{i+1}/{nbatches:<10}\t' \
                      f'MSE_mu: {float(_mu_mse)/len(batch.src):.6f}\t' \
                      f'MSE_logvar: {float(_logvar_mse)/len(batch.src):.6f}\t' \
                      f'KLD_mu: {float(_mu_kld)/len(batch.src):.6f}\t' \
                      f'KLD_logvar: {float(_logvar_kld)/len(batch.src):.6f}\t' \
                      f'TotalT(s): {end_time-start_time:.1f}\t'

            self.LOG_details.info(details)

        return (sum_mu_mse/n_samples, sum_logvar_mse/n_samples, 
                sum_mu_kld/n_samples, sum_logvar_kld/n_samples)

    
    def train(self, args, model, train_iter, valid_iter, SRC, TRG, device):
        if args.start_epoch == 1:
            if os.path.exists(self.results_path):
                os.remove(self.results_path)
            if os.path.exists(self.details_path):
                os.remove(self.details_path)

        if args.loss_fcn == 'mse':
            criterion = MSE()
        elif args.loss_fcn == 'kld':
            criterion = KLDiv()

        opt_mu = self.get_optimization(args, filter(lambda p: p.requires_grad,
                                       model.parameters()), 'state_dict_mu')
        opt_logvar = self.get_optimization(args, filter(lambda p: p.requires_grad,
                                           model.parameters()), 'state_dict_logvar')

        lowest_kldiv = 1000
        early_stop, stop_cnt = 4, 0
        epoch_best = args.start_epoch

        self.LOG_results.info(args)

        for epoch in range(args.start_epoch, args.num_epoch+1):
            self.LOG_results.info(f"Start EPOCH {epoch}")
            
            model.train()

            self.LOG_details.info(f"Training Start EPOCH: {epoch}")
            loss = self.run_epoch(model,
                                  train_iter,
                                  args.train_nbatches, TRG,
                                  LossCompute(criterion, opt_mu),
                                  LossCompute(criterion, opt_logvar),
                                  device
                                  )
            train_mu_mse, train_logvar_mse, train_mu_kld, train_logvar_kld = loss
            
            self.LOG_details.info("Training End")

            """ Validation """
            model.eval()

            self.LOG_details.info(f"Validation Start EPOCH: {epoch}")
            with torch.no_grad():
                loss = self.run_epoch(model,
                                      valid_iter,
                                      args.valid_nbatches, TRG,
                                      LossCompute(criterion, None),
                                      LossCompute(criterion, None),
                                      device
                                      )
                valid_mu_mse, valid_logvar_mse, valid_mu_kld, valid_logvar_kld = loss
            self.LOG_details.info("Validation End")

            """ Recording """
            self.LOG_results.info(f"Train: MSE_mu/MSE_logvar/KLD_mu/KLD_logvar "
                                  f"{train_mu_mse:.6f}/{train_logvar_mse:.6f}/{train_mu_kld:.6f}/{train_logvar_kld:.6f}\t"
                                  f"Validate: MSE_mu/MSE_logvar/KLD_mu/KLD_logvar "
                                  f"{valid_mu_mse:.6f}/{valid_logvar_mse:.6f}/{valid_mu_kld:.6f}/{valid_logvar_kld:.6f}")

            """ Recording the best model """
            if lowest_kldiv > (valid_mu_kld*valid_logvar_kld)**0.5:
                # store lowest-loss new model every time is saferx
                if os.path.exists(os.path.join(self.save_directory, f"best_{epoch_best}.pt")):
                    os.remove(os.path.join(self.save_directory, f"best_{epoch_best}.pt"))
                    
                self.save_checkpoint(args, model, f"best_{epoch}.pt", opt_mu, opt_logvar,
                                     len(SRC.vocab), len(TRG.vocab))
                lowest_kldiv = (valid_mu_kld*valid_logvar_kld)**0.5
                epoch_best = epoch
                stop_cnt = 0
            else:
                stop_cnt += 1

            self.save_checkpoint(model, f"model_{epoch}.pt", opt_mu, opt_logvar,
                                 len(SRC.vocab), len(TRG.vocab))
            
            if stop_cnt >= early_stop:
                break