import os
import gc
from tqdm import tqdm
import time
from datetime import timedelta
# import pandas as pd
# import pickle as pkl
# from itertools import tee

import torch
# import torch.nn as nn


import utils.log as ul
import utils.file as uf
import utils.gpu as ug
# import models.dataset as md

from models.cvae_Transformer import decode, transformer, mask
from models.cvae_Transformer.noam_opt import NoamOpt as moptim
from models.cvae_Transformer.loss import LossCompute, Criterion, KLAnnealer

from models.mlpcvae_Transformer import mlptransformer
import configuration.config_default as cfgd


from Process import dataset as pdataset
from Process import field as pfield
from Process import create_fields, create_total_fields, save_fields

# import Process.data_preparation as pdp
# import Process.vocabulary as mv
import Process.batch as pb

DEBUG = True
N_SAMPLES = 80 if DEBUG is True else None


# get_dataset_dataframe(data_name, data_type, condition_list, condition_path=None,
#                           similarity=1, n_jobs=1, n_samples=None)


# def get_dataset(data_name, datatype, cond_list, similarity=0, n_samples=None):
#     dataset = pdp.get_dataset(data_name, datatype)
#     df = pd.DataFrame({
#         'src': [line for line in dataset],
#         'trg': [line for line in dataset]},
#         columns=["src", "trg"]
#     )


#     assert 0 <= similarity and similarity <= 1
#     if similarity != 0:
        

#     conditions = pdp.get_conditions(dataset, cond_list)
#     if n_samples is not None:
#         return dataset[:n_samples], conditions[:n_samples]
#     return dataset, conditions
    

# def get_dataset_iterator(opt, SRC, TRG, n_samples=None):
#     dataset, conditions = get_dataset(opt.dataset, opt.cond_list, 
#                                     'train', n_samples)
#     train_iterator = pdp.create_iterator(opt, 'train', dataset, dataset, 
#                                          SRC, TRG, conditions, debug=DEBUG)
#     validation_iterator = pdp.create_iterator(opt, 'train', dataset, dataset,
#                                               SRC, TRG, conditions, debug=DEBUG)
#     return train_iterator, validation_iterator


class TransformerTrainer(object):

    def __init__(self, opt):
        # parameters for training
        self.opt = opt
        self.save_path = os.path.join('experiments', opt.save_directory)
        self.pad_idx = cfgd.PAD
        self.max_len_trg = cfgd.MAX_STRLEN
        # for logging
        self.LOG = ul.get_logger(name="train_model", log_path=os.path.join(self.save_path, 'train_model.log'))
        self.LOG.info(opt)


    def get_dataloader(self, data_iter):
        """ return a data generator """
        return (pb.rebatch(self.opt.src_pad, b, self.opt.cond_list) for b in data_iter)


    def get_model(self, src_len_tokens, trg_len_tokens, train_stage=2):
        # build a model from scratch or load a model from a given epoch
        
        if train_stage == 1:
            if self.opt.starting_epoch == 1:
                model = transformer.build_transformer(src_len_tokens, trg_len_tokens,
                                                      self.opt.N, self.opt.d_model, self.opt.d_ff, 
                                                      self.opt.H, self.opt.latent_dim, self.opt.dropout, 
                                                      self.opt.nconds, self.opt.use_cond2dec, self.opt.use_cond2lat)
            else:
                file_path = os.path.join(self.save_path, f'checkpoint/model_{self.opt.starting_epoch-1}.pt')
                model = transformer.load_from_file(file_path)      
        
        elif train_stage == 2:
            file_path = os.path.join('molGCT', 'molgct.pt')
            # file_path = os.path.join(self.save_path, f'checkpoint/model_{self.opt.starting_epoch-1}.pt')
            model = transformer.load_from_file(self.opt, file_path, src_len_tokens, trg_len_tokens)
            # model = transformer.load_from_file(file_path)
            
            if self.opt.starting_epoch == 1:
                model = transformer.build_transformer(src_len_tokens, trg_len_tokens, self.opt.N, 
                                                      self.opt.d_model, self.opt.d_ff, self.opt.H, 
                                                      self.opt.latent_dim, self.opt.dropout, self.opt.nconds, 
                                                      self.opt.use_cond2dec, self.opt.use_cond2lat)
                model = mlptransformer.build_mlptransformer(model, src_len_tokens, trg_len_tokens, self.opt.N, 
                                                            self.opt.d_model, self.opt.d_ff, self.opt.H, 
                                                            self.opt.latent_dim, self.opt.dropout, self.opt.nconds, 
                                                            self.opt.use_cond2dec, self.opt.use_cond2lat, self.opt.variational)
            else:
                file_path = os.path.join(self.save_path, f'checkpoint/model_{self.opt.mlp_starting_epoch-1}.pt')
                model = mlptransformer.load_from_file(model, file_path)
                
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
            'variational': self.opt.variational,
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

        # self.opt.distributed = int(os.environ['WORLD_SIZE'])

        # print("@ distributed ? ", self.opt.distributed)
        # if self.opt.distributed > 1:
        #     ug.initialize_process_group(local_rank=os.eviron['LOCAL_RANK'])
        # print(f"local rank {self.opt.local_rank} exits.")

        print(">>> Get GPU...")
        self.opt.device = ug.allocate_gpu()

        print(">>> Preprocess & Save the Data...")
        df = pdataset.get_dataset_dataframe(
            data_name=self.opt.data_name,
            data_type='train', 
            condition_list=self.opt.cond_list, 
            condition_path=None,
            similarity=self.opt.similarity, # 補 
            lang_format=self.opt.lang_format,
            n_jobs=self.opt.n_jobs,
            n_samples=N_SAMPLES
        )
        df.to_csv(self.opt.data_path)
        del df
        gc.collect()

        print(">>> Create/Load Fields...")
        SRC, TRG = pfield.create_fields(self.opt.lang_format, self.opt.field_path)
        total_fields = create_total_fields(SRC, TRG, self.opt.cond_list)
        self.opt.src_pad = SRC.vocab.stoi['<pad>']
        self.opt.trg_pad = TRG.vocab.stoi['<pad>']
        assert self.opt.src_pad == self.opt.trg_pad

        print(">>> Create Dataset...")
        dataset = pdataset.to_dataset(self.opt.data_path, total_fields)

        print(">>> Save Fields...")
        if self.opt.load_field is False:
            SRC.build_vocab(dataset)
            TRG.build_vocab(dataset)
            save_fields(SRC, TRG, self.opt.field_path)
        
        print(">>> Transform Dataset")
        

        
        exit(0)

        print("- Get dataIterator...")
        n_samples = 100
        dataset, conditions = get_dataset(self.opt.dataset, 'train', 
                                          self.opt.cond_list, n_samples=n_samples)

        train_iterator, valid_iterator = get_dataset_iterator(self.opt, SRC, TRG, n_samples=80)
        
        print("- Get model...")
        model = self.get_model(len(SRC.vocab), len(TRG.vocab), self.opt.train_stage)

        print("- Get optimizer...")
        optim = self.get_optimization(model)

        print("- Get optimizer...")
        criterion = Criterion(size=len(TRG.vocab), padding_idx=self.pad_idx,
                              smoothing=self.opt.label_smoothing)

        print("- Total parameters:", 
              sum(p.numel() for p in model.parameters()))
        print("- Total trainable parameters:", 
              sum(p.numel() for p in model.parameters() if p.requires_grad))

        for n, p in model.named_parameters():
            if p.requires_grad == True:
                print(n)

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
                if os.path.exists(os.path.join(self.save_path, "checkpoint", f"mlpbest_{epoch_best}.pt")):
                    os.remove(os.path.join(self.save_path, "checkpoint", f"mlpbest_{epoch_best}.pt"))
                    
                self.save(model, optim, f"best_{epoch}.pt", len(SRC.vocab), len(TRG.vocab))
                lowest_loss = rec_loss_val + KL_div_val
                model_best = model
                optim_best = optim
                epoch_best = epoch
                early_stop_cnt = 0
            else:
                early_stop_cnt += 1

            self.save(model, optim, f"mlpmodel_{epoch}.pt", len(SRC.vocab), len(TRG.vocab))
            epoch += 1

            if early_stop_cnt > early_stop:
                break

        self.save(model, optim, f"mlpmodel_{epoch}.pt", len(SRC.vocab), len(TRG.vocab))
