import os
import gc
import numpy as np
import time
from timeit import default_timer as timer
from datetime import timedelta
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


class Trainer(object):
    def __init__(self, args):
        # parameters for training
        self.args = args
        os.makedirs(args.save_directory, exist_ok=True)

        results_path = os.path.join(args.save_directory, 'train_results.log')
        details_path = os.path.join(args.save_directory, 'train_details.log')

        if args.start_epoch == 1:
            if os.path.exists(results_path):
                os.remove(results_path)
            if os.path.exists(details_path):
                os.remove(details_path)

        self.LOG_results = get_logger(name="train_results",
                                      log_path=results_path)
        self.LOG_results.info(args)
        self.LOG_details = get_logger(name="train_details", 
                                      log_path=details_path)
    
    def get_optimization(self, trainable_parameters):
        if self.args.start_epoch == 1:
            optimizer = torch.optim.Adam(trainable_parameters, lr=0,
                                         betas=(self.args.adam_beta1, 
                                                self.args.adam_beta2), 
                                         eps=self.args.adam_eps)
            optim = moptim(self.args.d_model, self.args.factor,
                           self.args.warmup_steps, optimizer)
        else:
            checkpoint = torch.load(os.path.join(self.args.save_directory,
                                    f'model_{self.args.start_epoch-1}.pt'), map_location='cuda:0')
            optim_dict = checkpoint['optimizer_state_dict']
            optim = moptim(optim_dict['model_size'], optim_dict['factor'], 
                           optim_dict['warmup'], torch.optim.Adam(trainable_parameters, lr=0))
            optim.load_state_dict(optim_dict)
        return optim


    def save_checkpoint(self, model, optim, model_name, src_vocab_size, trg_vocab_size):
        # save optimizer and hyperparametdecodeers
        os.makedirs(self.args.save_directory, exist_ok=True)
        file_name = os.path.join(self.args.save_directory, model_name)
        model_params = { 
            'N': self.args.N, 'H': self.args.H, 'd_ff': self.args.d_ff, 'nconds': self.args.nconds, 
            'd_model': self.args.d_model, 'dropout': self.args.dropout, 'latent_dim': self.args.latent_dim, 
            'variational': self.args.variational, 'use_cond2dec': self.args.use_cond2dec, 
            'use_cond2lat': self.args.use_cond2lat, 'src_vocab_size': src_vocab_size, 'trg_vocab_size': trg_vocab_size
        }
        save_dict = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.save_state_dict(),
            'model_parameters': model_params
        }
        torch.save(save_dict, file_name)


    def run_epoch(self, model, data_iter, nbatches, loss_compute, device, TRG):
        """
        The following variables records the total values from the dataset
        """
        n_samples = sum_mse = sum_kld = 0
        total_model_time = total_update_time = total_clear_time = 0

        start_time = timer()
        mse, kld = MSE(), KLDiv()

        dataloader = to_dataloader(data_iter,
                                   self.args.conditions,
                                   self.args.pad_idx,
                                   self.args.max_strlen,
                                   device)

        for i, batch in enumerate(dataloader):
            # dim of out: (batch_size, max_trg_seq_length-1, d_model)
            total_model_time -= timer()
            
            src_pad_mask = create_source_mask(batch.src, self.args.pad_idx, batch.econds)
            trg_pad_mask = create_source_mask(batch.trg_en, self.args.pad_idx, batch.dconds)

            trg_z_pred, trg_z_truth = model.forward(batch.src,
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
            
            _kld = kld(trg_z_pred, trg_z_truth)
            _mse = mse(trg_z_pred, trg_z_truth)

            _loss = loss_compute(trg_z_pred, trg_z_truth)

            total_update_time += timer()

            total_clear_time -= timer()
            torch.cuda.empty_cache()
            total_clear_time += timer()

            sum_kld += float(_kld)
            sum_mse += float(_mse)
            n_samples += len(batch.src)

            end_time = timer()

            details = f'{i+1}/{nbatches:<10}\t' \
                      f'MSE: {float(_mse)/len(batch.src):.6f}\t' \
                      f'KLDiv: {float(_kld)/len(batch.src):.6f}\t' \
                      f'TotalT(s): {end_time-start_time:.1f}\t' \
                      f'ModelT(s): {total_model_time:.1f}\t' \
                      f'UpdateT(s): {total_update_time:.1f}\t' \
                      f'ClearT(s): {total_clear_time:.1f}\t'

            self.LOG_details.info(details)
            print(details)

        return sum_mse/n_samples, sum_kld/n_samples

    
    def train(self, model, train_iter, valid_iter, SRC, TRG, device):
        if self.args.loss_fcn == 'mse':
            criterion = MSE()
        elif self.args.loss_fcn == 'kld':
            criterion = KLDiv()

        optim = self.get_optimization(filter(lambda p: p.requires_grad, model.parameters()))
        
        lowest_kldiv = 1000
        early_stop, stop_cnt = 4, 0
        epoch_best = self.args.start_epoch

        for epoch in range(self.args.start_epoch, self.args.num_epoch+1):
            self.LOG_results.info(f"Start EPOCH {epoch}")

            """ Train """
            model.train()

            self.LOG_details.info(f"Training Start EPOCH: {epoch}")
            train_mse, train_kldiv = self.run_epoch(model, train_iter, self.args.train_nbatches,
                                                    LossCompute(criterion, optim), device, TRG)
            self.LOG_details.info("Training End")

            """ Validation """
            model.eval()

            self.LOG_details.info(f"Validation Start EPOCH: {epoch}")
            with torch.no_grad():
                valid_mse, valid_kldiv = self.run_epoch(model, valid_iter, self.args.valid_nbatches,
                                                        LossCompute(criterion, None), device, TRG)
            self.LOG_details.info("Validation End")

            """ Recording """
            self.LOG_results.info(f"Train:RMSE/KLDiv {train_mse:.6f}/{train_kldiv:.6f}\t"
                                   f"Valid:RMSE/KLDiv {valid_mse:.6f}/{valid_kldiv:.6f}")

            """ Recording the best model """
            if lowest_kldiv > valid_kldiv:
                # store lowest-loss new model every time is saferx
                if os.path.exists(os.path.join(self.args.save_directory, f"best_{epoch_best}.pt")):
                    os.remove(os.path.join(self.args.save_directory, f"best_{epoch_best}.pt"))
                    
                self.save_checkpoint(model, optim, f"best_{epoch}.pt", len(SRC.vocab), len(TRG.vocab))
                lowest_kldiv = valid_kldiv
                epoch_best = epoch
                stop_cnt = 0
            else:
                stop_cnt += 1

            self.save_checkpoint(model, optim, f"model_{epoch}.pt", len(SRC.vocab), len(TRG.vocab))
            
            if stop_cnt >= early_stop:
                break

    # def train(self, model, train_iter, valid_iter, SRC, TRG, device):
    #     # criterion = MSELoss() # 1
    #     criterion = KLDiv()

    #     optim = self.get_optimization(filter(lambda p: p.requires_grad, model.parameters()))
        
    #     lowest_mse = 1000
    #     early_stop, stop_cnt = 4, 0
    #     epoch_best = self.args.start_epoch

    #     for epoch in range(self.args.start_epoch, self.args.num_epoch+1):
    #         self.LOG_results.info(f"Start EPOCH {epoch}")

    #         """ Train """
    #         model.train()

    #         self.LOG_details.info(f"Training Start EPOCH: {epoch}")
    #         train_mse, train_kldiv = self.run_epoch(train_iter, self.args.train_nbatches, model, 
    #                                                  LossCompute(criterion, optim), device, TRG)
    #         self.LOG_details.info("Training End")

    #         """ Validation """
    #         model.eval()

    #         self.LOG_details.info(f"Validation Start EPOCH: {epoch}")
    #         with torch.no_grad():
    #             valid_mse, valid_kldiv = self.run_epoch(valid_iter, self.args.valid_nbatches, model,
    #                                                       LossCompute(criterion, None), device, TRG)
    #         self.LOG_details.info("Validation End")

    #         """ Recording """
    #         self.LOG_results.info(f"Train:RMSE/KLDiv(10^6) {train_mse:.3f}/{train_kldiv*10**6:.3f}\t"
    #                                f"Valid:RMSE/KLDiv(10^6) {valid_mse:.3f}/{valid_kldiv*10**6:.3f}")

    #         """ Recording the best model """
    #         if lowest_mse > valid_mse:
    #             # store lowest-loss new model every time is saferx
    #             if os.path.exists(os.path.join(self.args.save_directory, f"best_{epoch_best}.pt")):
    #                 os.remove(os.path.join(self.args.save_directory, f"best_{epoch_best}.pt"))
                    
    #             self.save_checkpoint(model, optim, f"best_{epoch}.pt", len(SRC.vocab), len(TRG.vocab))
    #             lowest_mse = valid_mse
    #             epoch_best = epoch
    #             stop_cnt = 0
    #         else:
    #             stop_cnt += 1

    #         self.save_checkpoint(model, optim, f"model_{epoch}.pt", len(SRC.vocab), len(TRG.vocab))
            
    #         if stop_cnt >= early_stop:
    #             break