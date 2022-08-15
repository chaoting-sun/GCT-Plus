import os
import gc
from timeit import default_timer as timer
# from time import time
import glob
from datetime import timedelta
from line_profiler import LineProfiler
# import pandas as pd
# import pickle as pkl
# from itertools import tee

import torch
import torch.nn as nn
import torch.nn.functional as F
from Tokenize import moltokenize


from Model.mlp import decode
from Model.modules import NoamOpt as moptim
from Model.loss import LossCompute, Criterion, MSELoss
from Utils.dataset import to_dataloader
from Utils.log import get_logger
from Utils.dataset import pickle_load, torch_load, np_load, memmap_tp_torch, sqlite_select, to_device
# import Process.data_preparation as pdp
# import Process.vocabulary as mv

DEBUG = True
N_SAMPLES = 80 if DEBUG is True else None


# from pynvml import *
# def gpu_utilization():
#     nvmlInit()
#     handle = nvmlDeviceGetHandleByIndex(0)
#     info = nvmlDeviceGetMemoryInfo(handle)
#     return info.used//1024**2 # MB


class Trainer(object):
    def __init__(self, args, SRC, TRG):
        # parameters for training
        self.args = args
        self.SRC = SRC
        self.TRG = TRG
        
        self.save_directory = args.save_directory
        os.makedirs(self.save_directory, exist_ok=True)
        self.LOG_progress = get_logger(name="train_progress", 
                   log_path=os.path.join(self.save_directory, 'train_progress.log'))
        self.LOG_progress.info(args)
        self.LOG_details = get_logger(name="train_details", 
                           log_path=os.path.join(self.save_directory, 'train_details.log'))

    
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


    def save_checkpoint(self, optim, model, model_name):
        # save optimizer and hyperparametdecodeers
        os.makedirs(self.save_directory, exist_ok=True)
        file_name = os.path.join(self.save_directory, model_name)
        model_params = { 
            'N': self.args.N, 'H': self.args.H, 'd_ff': self.args.d_ff,
            'nconds': self.args.nconds, 'd_model': self.args.d_model,
            'dropout': self.args.dropout, 'latent_dim': self.args.latent_dim, 
            'variational': self.args.variational, 'use_cond2dec': self.args.use_cond2dec, 
            'use_cond2lat': self.args.use_cond2lat, 'src_vocab_size': len(self.SRC.vocab), 
            'trg_vocab_size': len(self.TRG.vocab)
        }
        save_dict = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.save_state_dict(),
            'model_parameters': model_params
        }
        torch.save(save_dict, file_name)


    def remove_checkpoint(self, model_name):
        checkpoint_path = os.path.join(self.save_directory, model_name)
        os.remove(checkpoint_path)


    def run_epoch(self, dataloader, model, loss_compute, device):
        """
        The following variables records the total values from the dataset
        """
        total_model_time = total_update_time = total_clear_time = 0
        clear_time_last = load_time_last = 0

        time_tolerance = 2
        save_interval = 100

        start_time = timer()
        kl_loss_fcn = nn.KLDivLoss(reduction='batchmean')
    
        for i, batch in enumerate(dataloader):
            nb = i+1 # nb-th batch
            if nb <= self.last_batch:
                del batch
                start_time = timer() 
                continue
            
            batch['src'] = batch['src'].to(device)
            batch['trg'] = batch['trg'].to(device)
            batch['mconds'] = batch['mconds'].to(device)
    
            # dim of out: (batch_size, max_trg_seq_length-1, d_model)
            
            total_model_time -= timer()
            trg_z_pred, trg_z_truth = model.forward(batch['src'],
                                                    batch['trg'],
                                                    batch['mconds'])
            total_model_time += timer()

            # Compute loss (rec-loss + KL-div) and update
            total_update_time -= timer()
            loss = loss_compute(trg_z_pred, trg_z_truth)
            kl_loss = kl_loss_fcn(F.log_softmax(trg_z_pred.contiguous().view(-1), dim=-1), 
                                  F.softmax(trg_z_truth.contiguous().view(-1), dim=-1))
            total_update_time += timer()
            
            total_clear_time -= timer()
            torch.cuda.empty_cache()
            del trg_z_pred, trg_z_truth, batch
            gc.collect()
            total_clear_time += timer()

            total_time = timer() - start_time
            details = f'{i+1}/{len(dataloader):<10}\t' \
                      f'KLdiv(*10^): {float(kl_loss)*10**7:.3f}\t' \
                      f'RMSE: {float(loss):.3f}\t' \
                      f'TotalT(s): {total_time:.2f}\t' \
                      f'ModelT(s): {total_model_time:.2f}\t' \
                      f'UpdateT(s): {total_update_time:.2f}\t' \
                      f'ClearT(s): {total_clear_time:.2f}\t' \
                      f'IOT(s): {torch_load.cummulative_time:.2f}\t' \
                    #   f'IOT(s): {pickle_load.cummulative_time:.2f}\t' \

            self.LOG_details.info(details)
            print(details)

            if nb % save_interval == 0:
                self.save_checkpoint(loss_compute.optim, model, f"model_{self.last_epoch}_{nb}.pt")
                if self.last_batch > -1:
                    self.remove_checkpoint(f"model_{self.last_epoch}_{self.last_batch}.pt")
                self.last_batch = nb
            if (total_clear_time-clear_time_last) + \
               (torch_load.cummulative_time-load_time_last) > time_tolerance:
                self.save_checkpoint(loss_compute.optim, model, f"model_{self.last_epoch}_{nb}.pt")
                if self.last_batch > -1:
                    self.remove_checkpoint(f"model_{self.last_epoch}_{self.last_batch}.pt")
                self.last_batch = nb
                raise Exception('Training too slow!')

            clear_time_last = total_clear_time
            load_time_last = torch_load.cummulative_time
            

    def train(self, model, train_dl, valid_dl, last_batch, device):
        criterion = MSELoss()
        optim = self.get_optimization(filter(lambda p: p.requires_grad, model.parameters()))
        
        # lowest_loss = 1000
        # early_stop, stop_cnt = 10, 0
        # epoch_best = self.args.start_epoch

        for epoch in range(self.args.start_epoch, self.args.num_epoch+1):
            self.last_epoch = epoch
            self.last_batch = last_batch
            self.LOG_progress.info(f"Start EPOCH {epoch}")
            
            """ Train """
            model.train()
            if last_batch == len(train_dl.dataset):
                self.LOG_details.info(f"TrainingStart EPOCH: {epoch}")
            self.run_epoch(train_dl, model, LossCompute(criterion, optim), device)
            self.LOG_details.info(f"TrainingEnd")

            """ Validation """
            model.eval()
            if last_batch == len(valid_dl.dataset):
                self.LOG_details.info(f"ValidationStart EPOCH: {epoch}")
            with torch.no_grad():
                self.run_epoch(valid_dl, model, LossCompute(criterion, None), device)
            self.LOG_details.info(f"ValidationEnd")

            # """ Recording """
            # self.LOG_progress.info(f"TrainLoss {loss_train:.4f}\tValidationLoss {loss_val:.4f}")

            # """ Recording the best model """
            # if lowest_loss > loss_val:
            #     # store lowest-loss new model every time is saferx
            #     if os.path.exists(os.path.join(self.args.save_directory, f"best_{epoch_best}.pt")):
            #         os.remove(os.path.join(self.args.save_directory, f"best_{epoch_best}.pt"))
                    
            #     self.save_checkpoint(optim, model, f"best_{epoch}.pt")
            #     lowest_loss = loss_val
            #     epoch_best = epoch
            #     stop_cnt = 0
            # else:
            #     stop_cnt += 1

            # self.save_checkpoint(optim, model, f"model_{epoch}.pt")
            
            # epoch += 1
            # if stop_cnt >= early_stop:
            #     break