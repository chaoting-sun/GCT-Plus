import os
import gc
import time
from tqdm import tqdm
from datetime import timedelta
# import pandas as pd
# import pickle as pkl
# from itertools import tee

import torch
# import torch.nn as nn

from Model.mlpcvae_Transformer import decode
from Model.mlpcvae_Transformer.noam_opt import NoamOpt as moptim
from Model.mlpcvae_Transformer.loss import LossCompute, Criterion
from Utils.dataset import to_dataloader
from Utils.log import get_logger
# import Process.data_preparation as pdp
# import Process.vocabulary as mv

DEBUG = True
N_SAMPLES = 80 if DEBUG is True else None


class Trainer(object):
    def __init__(self, args):
        # parameters for training
        self.args = args
        self.save_path = os.path.join('Experiment', args.save_directory)
        self.LOG = get_logger(name="train_model", log_path=os.path.join(self.save_path, 'train_model.log'))
        self.LOG.info(args)


    def get_optimization(self, trainable_parameters):
        if self.args.starting_epoch == 1:
            optimizer = torch.optim.Adam(trainable_parameters, lr=0,
                                         betas=(self.args.adam_beta1, 
                                                self.args.adam_beta2), 
                                         eps=self.args.adam_eps)
            optim = moptim(self.args.d_model, self.args.factor,
                           self.args.warmup_steps, optimizer)           
        else:
            checkpoint = torch.load(os.path.join(self.save_path, f'model_{self.args.starting_epoch-1}.pt'), map_location='cuda:0')
            optim_dict = checkpoint['optimizer_state_dict']
            optim = moptim(optim_dict['model_size'], optim_dict['factor'], 
                           optim_dict['warmup'], torch.optim.Adam(trainable_parameters, lr=0))
            optim.load_state_dict(optim_dict)
        return optim


    def save_checkpoint(self, model, optim, model_name, src_vocab_size, trg_vocab_size):
        # save optimizer and hyperparameters
        os.makedirs(self.save_path, exist_ok=True)
        file_name = os.path.join(self.save_path, model_name)
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


    def run_epoch(self, iter, model, loss_compute, device):
        """
        The following variables records the total values from the dataset
        """
        sum_loss = 0
        n_tokens = 0
        n_samples = 0
        n_correct = 0
        
        dataloader = to_dataloader(iter, self.args.conditions)
        
        for i, batch in tqdm(enumerate(dataloader), total=len(iter)):
            src = batch.src.to(device)
            trg_y = batch.trg_y.to(device)
            trg = batch.trg.to(device)
            econds = batch.econds.to(device) 
            mconds = batch.mconds.to(device)
            dconds = batch.dconds.to(device) 
            # dim of out: (batch_size, max_trg_seq_length-1, d_model)
            _, out, _, _, _ = model.forward(src, trg, econds, mconds, dconds)

            # Compute loss (rec-loss + KL-div) and update
            loss = loss_compute(out, trg_y)
            sum_loss += float(loss)
            smiles = decode.decode(model, src, econds, mconds, dconds, self.args.max_strlen, 
                                   type='greedy', use_cond2dec=self.args.use_cond2dec)
            n_samples += batch.trg.size(0)
            n_tokens += float((batch.trg_y != self.args.src_pad_idx).data.sum())

            for b in range(batch.trg.size(0)):
                if torch.equal(smiles[b, :], trg[b]):
                    n_correct += 1

        # the accuracy in a epoch & average loss of each predicted token
        return n_correct*1.0/n_samples, sum_loss/n_tokens


    def train(self, model, train_iter, valid_iter, SRC, TRG, device):
        src_vocab_size, trg_vocab_size = len(SRC.vocab), len(TRG.vocab)

        model = model.to(device)
        
        print(">>> GET OPTIMIZER")
        optim = self.get_optimization(filter(lambda p: p.requires_grad, model.parameters()))

        print(">>> GET CRITERION")
        criterion = Criterion(size=len(TRG.vocab), padding_idx=SRC.vocab.stoi['<pad>'],
                              smoothing=self.args.label_smoothing)
        
        lowest_loss = 1000000
        early_stop, stop_cnt = 8, 0
        epoch = epoch_best = self.args.starting_epoch

        while epoch <= self.args.num_epoch:
            print(f">>> EPOCHS: ", epoch)

            self.LOG.info("Starting EPOCH #%d", epoch)

            """ Train """
            self.LOG.info("Training start")
            model.train()
            acc_train, loss_train = self.run_epoch(train_iter, model,
                                    LossCompute(criterion, optim), device)
            print(">>> training accuracy: {:.6}\ttraining loss: {:.6}".format(acc_train, loss_train))

            self.LOG.info("Training end")

            """ Validation """
            self.LOG.info("Validation start")
            model.eval()
            with torch.no_grad():
                acc_val, loss_val = self.run_epoch(valid_iter, model,
                                    LossCompute(criterion, None), device)
            print(">>> validation accuracy: {:.6}\tvalidation loss: {:.6}".format(acc_val, loss_val))

            self.LOG.info("Validation end")

            """ Recording """
            self.LOG.info("Train:Acc/loss {:.6},{:.6}\tValidation:Acc/loss {:.6},{:.6}".
                          format(acc_train, loss_train, acc_val, loss_val))

            """ Recording the best """
            if lowest_loss > loss_val:
                # store lowest-loss new model every time is safer
                if os.path.exists(os.path.join(self.save_path, f"best_{epoch_best}.pt")):
                    os.remove(os.path.join(self.save_path, f"best_{epoch_best}.pt"))
                    
                self.save_checkpoint(model, optim, f"best_{epoch}.pt", src_vocab_size, trg_vocab_size)
                lowest_loss = loss_val
                epoch_best = epoch
                stop_cnt = 0
            else:
                stop_cnt += 1

            self.save_checkpoint(model, optim, f"model_{epoch}.pt", src_vocab_size, trg_vocab_size)
            
            epoch += 1
            if stop_cnt >= early_stop:
                break