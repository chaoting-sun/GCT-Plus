from genericpath import exists
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
        os.makedirs(args.save_directory, exist_ok=True)
        self.LOG = get_logger(name="train_model", 
                              log_path=os.path.join(args.save_directory, 'train_model.log'))
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
            checkpoint = torch.load(os.path.join(self.args.save_directory, 
                                    f'model_{self.args.starting_epoch-1}.pt'), map_location='cuda:0')
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


    def run_epoch(self, data_iter, model, loss_compute, device, TRG):
        """
        The following variables records the total values from the dataset
        """
        sum_loss, n_tokens, n_samples, n_correct = 0, 0, 0, 0
        dataloader = to_dataloader(data_iter, self.args.conditions, TRG.vocab.stoi['<pad>'], device)

        # torch.set_printoptions(threshold=10_000)

        # for i, batch in tqdm(enumerate(dataloader), total=len(data_iter)):
        for i, batch in enumerate(dataloader):
            # dim of out: (batch_size, max_trg_seq_length-1, d_model)
            trg_z_pred, trg_z_truth = model.forward(batch.src,
                                                    batch.trg_en,
                                                    batch.econds, 
                                                    batch.mconds, 
                                                    batch.dconds)
            # Compute loss (rec-loss + KL-div) and update
            loss = loss_compute(trg_z_pred, trg_z_truth)
            # loss = loss_compute(out, batch.trg_y)
            
            smiles = decode.decode(model, batch.src,
                                   batch.econds,
                                   batch.mconds,
                                   batch.dconds,
                                   self.args.sos_idx,
                                   self.args.eos_idx,
                                   self.args.max_strlen, 
                                   decode_type='greedy', 
                                   use_cond2dec=self.args.use_cond2dec)

            sum_loss += float(loss)
            n_samples += batch.trg.size(0)
            n_tokens += float((batch.trg_y != self.args.src_pad_idx).data.sum())

            # correctness: all tokens of a SMILES as a unit
            for b in range(batch.trg.size(0)):
                if torch.equal(smiles[b, :], batch.trg[b]):
                    n_correct += 1
            
            print('average_accuracy: {:.6f}\taverage_loss_each_token: {:.6f}'.
                  format(n_correct*1.0/n_samples, sum_loss/n_tokens))
            
            if i == len(data_iter) - 1:
                sample = smiles[:6]

        def printsmiles(smiles):
            smiles = smiles.cpu().numpy()
            for i in range(len(smiles)):
                outs = ''.join([TRG.vocab.itos[tok] for tok in smiles[i]])
                print(i, outs)
        print('>>> SAMPLE SMILES:')
        printsmiles(sample)

        print("--- accuracy: {:.6}\tloss: {:.6}".format(
            n_correct*1.0/n_samples, sum_loss/n_tokens))

        # the accuracy in a epoch & average loss of each predicted token
        return n_correct*1.0/n_samples, sum_loss/n_tokens


    def train(self, model, train_iter, valid_iter, SRC, TRG, device):
        print(">>> PREPARING OPTIMIZER")
        optim = self.get_optimization(filter(lambda p: p.requires_grad, model.parameters()))

        print(">>> PREPARING LOSS FUNCTION")
        criterion = Criterion()
        
        lowest_loss = 1000000
        early_stop, stop_cnt = 10, 0
        epoch = epoch_best = self.args.starting_epoch

        print('>>> TRAINING THE MODEL')
        while epoch <= self.args.num_epoch:
            print(f"{epoch} EPOCH: ")

            self.LOG.info("Starting EPOCH #%d", epoch)

            """ Train """
            model.train()

            self.LOG.info("Training start")
            acc_train, loss_train = self.run_epoch(train_iter, model,
                                    LossCompute(criterion, optim), device, TRG)
            self.LOG.info("Training end")

            """ Validation """
            model.eval()

            self.LOG.info("Validation start")
            with torch.no_grad():
                acc_val, loss_val = self.run_epoch(valid_iter, model,
                                    LossCompute(criterion, None), device, TRG)
            self.LOG.info("Validation end")

            """ Recording """
            self.LOG.info("Train:Acc/loss {:.6},{:.6}\tValidation:Acc/loss {:.6},{:.6}".
                          format(acc_train, loss_train, acc_val, loss_val))

            """ Recording the best """
            if lowest_loss > loss_val:
                # store lowest-loss new model every time is safer
                if os.path.exists(os.path.join(self.args.save_directory, f"best_{epoch_best}.pt")):
                    os.remove(os.path.join(self.args.save_directory, f"best_{epoch_best}.pt"))
                    
                self.save_checkpoint(model, optim, f"best_{epoch}.pt", len(SRC.vocab), len(TRG.vocab))
                lowest_loss = loss_val
                epoch_best = epoch
                stop_cnt = 0
            else:
                stop_cnt += 1

            self.save_checkpoint(model, optim, f"model_{epoch}.pt", len(SRC.vocab), len(TRG.vocab))
            
            epoch += 1
            if stop_cnt >= early_stop:
                break