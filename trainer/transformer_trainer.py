# Molecular Optimization

import os
import torch
import pandas as pd
import pickle as pkl
from tqdm import tqdm

import configuration.config_default as cfgd

import utils.log as ul
import utils.file as uf
import utils.torch_util as ut
import models.dataset as md
import Process.vocabulary as mv

from models.VAETransformer import decode, transformer, mask
from models.VAETransformer.noam_opt import NoamOpt as moptim
from models.VAETransformer.loss import LossCompute, Criterion, KLAnnealer

import Process.data_preparation as pdp
import Process.batch as pb
from itertools import tee


DEBUG = False

class TransformerTrainer(object):

    def __init__(self, opt):
        # parameters for training
        self.opt = opt
        self.save_path = os.path.join('experiments', opt.save_directory)
        self.pad_idx = cfgd.DATA_DEFAULT['padding_value']
        self.max_len_trg = cfgd.DATA_DEFAULT['max_sequence_length']
        # for logging
        self.LOG = ul.get_logger(name="train_model", log_path=os.path.join(self.save_path, 'train_model.log'))
        self.LOG.info(opt)

    
    def get_dataIterator(self, data_type, SRC=None, TRG=None):
        if self.opt.dataset == 'moses' and data_type == 'validation':
            data_type = 'test'

        dataset = pdp.get_dataset(self.opt.dataset, data_type)
        
        dataset = dataset[:1000]

        if self.opt.nconds > 0:
            df_conds = pdp.get_property(dataset, self.opt.lang_format, self.opt.cond_list)
        else:
            df_conds = None

        if data_type == 'train':
            if self.opt.load_field:
                SRC, TRG = pdp.create_fields(self.opt.lang_format, self.opt.field_path)
            else:
                SRC, TRG = pdp.create_fields(self.opt.lang_format, None)
        elif data_type == 'validation':
                SRC, TRG = pdp.create_fields(self.opt.lang_format, self.opt.field_path)

        data_iter = pdp.create_dataset(self.opt, data_type, dataset, dataset, SRC, TRG, df_conds)

        return data_iter, SRC, TRG


    def get_dataloader(self, data_iter):
        """ return a data generator """
        return (pb.rebatch(self.opt.src_pad, b, self.opt.cond_list) for b in data_iter)


    def get_model(self, src_len_tokens, trg_len_tokens):
        # build a model from scratch or load a model from a given epoch

        if self.opt.starting_epoch == 1:
            model = transformer.build_transformer(src_len_tokens, trg_len_tokens,
                                                  self.opt.N, self.opt.d_model, self.opt.d_ff, 
                                                  self.opt.H, self.opt.latent_dim, self.opt.dropout, 
                                                  self.opt.nconds, use_cond2dec=True, use_cond2lat=False)
        else:
            file_path = os.path.join(self.save_path, f'checkpoint/model_{self.opt.starting_epoch-1}.pt')
            model = transformer.load_from_file(file_path)
        model.to(self.opt.device)
        return model


    def get_optimization(self, model):
        if self.opt.starting_epoch == 1:
            optim = moptim(self.opt.d_model, self.opt.factor, self.opt.warmup_steps,
                           torch.optim.Adam(model.parameters(), lr=0, betas=(self.opt.adam_beta1, self.opt.adam_beta2), eps=self.opt.adam_eps))            
        else:
            checkpoint = torch.load(os.path.join(self.save_path, f'checkpoint/model_{self.opt.starting_epoch-1}.pt'), map_location='cuda:0')
            optim_dict = checkpoint['optimizer_state_dict']
            optim = moptim(optim_dict['model_size'], optim_dict['factor'], 
                           optim_dict['warmup'], torch.optim.Adam(model.parameters(), lr=0))
            optim.load_state_dict(optim_dict)
        return optim
    

    def train_epoch(self, train_iter, model, loss_compute, device):
        """
        The following variables records the total values from the dataset
        """
        total_loss = 0     # total loss
        total_rec_loss = 0 # total reconstruction loss
        total_kl_div = 0   # total KL divergence
        total_tokens = 0   # total non-padded tokens
        n_correct = 0      # total number of correct predicted smiles
        total_n = 0        # total number of target smiles

        rec_loss_list = []
        kl_div_list = []
        beta_list = []

        # parameter of kl divergence
        klannealer = KLAnnealer(self.opt.kl_beta_init, self.opt.kl_beta, self.opt.kl_cycle)

        for i, batch in tqdm(enumerate(train_iter), total=self.opt.train_len):
            # CPU to GPU
            src = batch.src.to(device)
            trg_y = batch.trg_y.to(device)
            trg = batch.trg.to(device)
            conds = batch.conds.to(device) 
            
            # dim of out: (batch_size, max_trg_seq_length-1, d_model)
            _, out, mu, log_var, _, _, _, _ = model.forward(src, trg, conds)

            beta = klannealer(i)

            # Compute loss (rec-loss + KL-div) and update
            rec_loss, kl_div = loss_compute(out, trg_y, total_n, mu, log_var, beta)

            total_loss += float(rec_loss + kl_div)
            total_rec_loss += float(rec_loss)
            total_kl_div += float(kl_div)

            if self.opt.train_verbose:
                print('The model decodes to SMILES.')

            smiles = decode.decode(model, src, conds, self.max_len_trg, 
                                   type='greedy', use_cond2dec=True)

            batch_n = batch.trg.size(0)
            total_n += batch_n
            total_tokens += float((batch.trg_y != self.pad_idx).data.sum())
            n_correct = sum([1 if torch.equal(smiles[b, :], trg[b]) else 0 for b in range(batch_n)])
            
            rec_loss_list.append(float(rec_loss))
            kl_div_list.append(float(kl_div))
            beta_list.append(float(beta))

        print("reconstruction loss:\n", rec_loss_list)
        print("KL divergence:\n", kl_div_list)
        print("Beta values:\n", beta_list)

        return (n_correct*1.0 / total_n,       # the overall accuracy
                total_rec_loss / total_tokens, # the average reconstruction loss per token
                total_kl_div / total_n)        # the average KL divergence per prediction (smiles)


    def validation_stat(self, valid_iter, model, loss_compute, device):
        total_loss = 0     # total loss
        total_rec_loss = 0 # total reconstruction loss
        total_kl_div = 0   # total KL divergence
        total_tokens = 0   # total non-padded tokens
        n_correct = 0      # total number of correct predicted smiles
        total_n = 0        # total number of smiles
        
        rec_loss_list = []
        kl_div_list = []
        beta_list = []

        # parameter of kl divergence
        klannealer = KLAnnealer(self.opt.kl_beta_init, self.opt.kl_beta, self.opt.kl_cycle)


        for i, batch in tqdm(enumerate(valid_iter), total=self.opt.test_len):
            # CPU to GPU
            src = batch.src.to(device)
            trg_y = batch.trg_y.to(device)
            trg = batch.trg.to(device)
            conds = batch.conds.to(device) 

            with torch.no_grad():
                # dim of out: (batch_size, max_trg_seq_length-1, d_model)
                _, out, mu, log_var, _, _, _, _ = model.forward(src, trg, conds)
                
                beta = klannealer(i)

                rec_loss, kl_div = loss_compute(out, trg_y, total_n, mu, log_var, beta)

                total_loss += float(rec_loss + kl_div)
                total_rec_loss += float(rec_loss)
                total_kl_div += float(kl_div)

                smiles = decode.decode(model, src, conds, self.max_len_trg, 
                                       type='greedy', use_cond2dec=True)

                batch_n = batch.trg.size(0)
                total_n += batch_n
                total_tokens += float((batch.trg_y != self.pad_idx).data.sum())
                n_correct = sum([1 if torch.equal(smiles[b, :], trg[b]) else 0 for b in range(batch_n)])

                rec_loss_list.append(float(rec_loss))
                kl_div_list.append(float(kl_div))
                beta_list.append(float(beta))

        print("reconstruction loss:\n", rec_loss_list)
        print("KL divergence:\n", kl_div_list)
        print("Beta values:\n", beta_list)

        return (n_correct / total_n,           # the overall accuracy
                total_rec_loss / total_tokens, # the average reconstruction loss per token
                total_kl_div / total_n)        # the average KL divergence per prediction (smiles)


    def _get_model_parameters(self, src_vocab_size, trg_vocab_size):
        return {
            'N': self.opt.N,
            'H': self.opt.H,
            'd_model': self.opt.d_model,
            'd_ff': self.opt.d_ff,
            'latent_dim': self.opt.latent_dim,
            'dropout': self.opt.dropout,
            'nconds': self.opt.nconds,
            'use_cond2dec': self.opt.use_cond2dec,
            'use_cond2lat': self.opt.use_cond2lat,
            'src_vocab_size': src_vocab_size,
            'trg_vocab_size': trg_vocab_size
        }


    def save(self, model, optim, model_name, src_vocab_size, trg_vocab_size):
        # save optimizer and hyperparameters
        file_name = os.path.join(self.save_path, "checkpoint", model_name)
        uf.make_directory(file_name, is_dir=False)
        save_dict = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.save_state_dict(),
            'model_parameters': self._get_model_parameters(src_vocab_size, trg_vocab_size)
        }
        torch.save(save_dict, file_name)


    def train(self):
        if DEBUG:
            torch.set_printoptions(threshold=10_000)

        self.opt.device = ut.allocate_gpu()

        if self.opt.train_verbose:
            print("\n - get training iterator...")

        train_iterator, SRC, TRG = self.get_dataIterator('train', None, None)

        if self.opt.train_verbose:
            print("\n - get validation iterator...")

        valid_iterator, _, _ = self.get_dataIterator('validation', SRC, TRG)

        exit(0)

        model = self.get_model(len(SRC.vocab), len(TRG.vocab))
        optim = self.get_optimization(model)
        criterion = Criterion(size=len(TRG.vocab), padding_idx=self.pad_idx,
                              smoothing=self.opt.label_smoothing)

        loss_acc_writer = ul.get_logger(name="loss_accuracy",
                                        log_path=os.path.join(self.save_path, 'loss_accuracy.log'))
        
        # early stop parameter
        early_stop_cnt = 0
        early_stop = 8
        lowest_loss = 10000.

        # record the best model and its present optimizer
        model_best = model
        optim_best = optim
        
        epoch = self.opt.starting_epoch
        epoch_best = self.opt.starting_epoch
        n_epochs = self.opt.starting_epoch + self.opt.num_epoch

        while epoch <= n_epochs:
            self.LOG.info("Starting EPOCH #%d", epoch)
            self.LOG.info("Training start")
            model.train()

            train_dl = self.get_dataloader(train_iterator)
            valid_dl = self.get_dataloader(valid_iterator)

            acc_train, rec_loss_train, KL_div_train = self.train_epoch(train_dl, model,
                                                      LossCompute(criterion, optim), self.opt.device)
            self.LOG.info("Training end")
            self.LOG.info("Validation start")
            
            model.eval()

            acc_val, rec_loss_val, KL_div_val = self.validation_stat(valid_dl, model, 
                                                LossCompute(criterion, None), self.opt.device)

            self.LOG.info("Validation end")
            self.LOG.info("Train:Acc/rec_loss/kl_div {:.6},{:.6},{:.6}  Validation:Acc/rec_loss/kl_div {:.6},{:.6},{:.6}".
                          format(acc_train, rec_loss_train, KL_div_train, acc_val, rec_loss_val, KL_div_val))

            loss_acc_writer.info("Train:Acc/rec_loss/kl_div {:.6},{:.6},{:.6}  Validation:Acc/rec_loss/kl_div {:.6},{:.6},{:.6}".
                            format(acc_train, rec_loss_train, KL_div_train, acc_val, rec_loss_val, KL_div_val))

            if lowest_loss > rec_loss_val + KL_div_val:
                # store lowest-loss new model every time is safer
                if os.path.exists(os.path.join(self.save_path, "checkpoint", f"best_{epoch_best}.pt")):
                    os.remove(os.path.join(self.save_path, "checkpoint", f"best_{epoch_best}.pt"))
                    
                self.save(model, optim, f"best_{epoch}.pt", len(SRC.vocab), len(TRG.vocab))
                lowest_loss = rec_loss_val + KL_div_val
                model_best = model
                optim_best = optim
                epoch_best = epoch
                early_stop_cnt = 0
            else:
                early_stop_cnt += 1

            self.save(model, optim, f"model_{epoch}.pt", len(SRC.vocab), len(TRG.vocab))
            epoch += 1

            if early_stop_cnt > early_stop:
                break

        self.save(model, optim, f"model_{epoch}.pt", len(SRC.vocab), len(TRG.vocab))
