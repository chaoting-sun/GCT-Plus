# Molecular Optimization

import os
import torch
import pandas as pd
import pickle as pkl

import configuration.config_default as cfgd

import utils.log as ul
import utils.file as uf
import utils.torch_util as ut
import models.dataset as md
import preprocess.vocabulary as mv

from models.VAETransformer import decode, transformer, mask
from models.VAETransformer.noam_opt import NoamOpt as moptim
from models.VAETransformer.label_smoothing import LabelSmoothing
from models.VAETransformer.simpleloss_compute import SimpleLossCompute
from models.VAETransformer.loss import Loss


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

    def initialize_dataloader(self, vocab, data_type):
        data = pd.read_csv(os.path.join(self.opt.data_path, data_type + '.csv'), sep=",")
        dataset = md.Dataset(data=data, vocabulary=vocab, 
                             tokenizer=mv.SMILESTokenizer(), prediction_mode=False)
        dataloader = torch.utils.data.DataLoader(dataset, self.opt.batch_size, 
                                                 shuffle=True, collate_fn=md.Dataset.collate_fn)
        return dataloader


    def get_model(self, vocab, device):
        # build a model from scratch or load a model from a given epoch

        if self.opt.starting_epoch == 1:
            model = transformer.build_transformer(len(vocab.tokens()), len(vocab.tokens()), 
                                                  self.opt.N, self.opt.d_model, self.opt.d_ff, 
                                                  self.opt.H, self.opt.latent_dim, self.opt.dropout, 
                                                  self.opt.nconds, use_cond2dec=True, use_cond2lat=False)
        else:
            file_path = os.path.join(self.save_path, f'checkpoint/model_{self.opt.starting_epoch-1}.pt')
            model = transformer.load_from_file(file_path)
        model.to(device)
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
    

    def train_epoch(self, dataloader, model, loss_compute, device, vocab):
        tokenizer = mv.SMILESTokenizer()
        """
        The following variables records the total values from the dataset
        """
        total_loss = 0 # total loss
        total_rec_loss = 0 # total reconstruction loss
        total_KL_div = 0 # total KL divergence
        total_tokens = 0 # total non-padded tokens
        n_correct = 0 # total number of correct predicted smiles
        total_n_trg = 0 # total number of target smiles
        n_rounds = 0 # the number of batches

        for i, batch in enumerate(ul.progress_bar(dataloader, total=len(dataloader))):
            src, trg, conds, _ = batch

            trg_y = trg[:, 1:] # the expected output
            trg = trg[:, :-1] # the input of the model
            ntokens = float((trg_y != self.pad_idx).data.sum())

            # CPU to GPU
            src = src.to(device)
            trg_y = trg_y.to(device)
            trg = trg.to(device)
            conds = conds.to(device)
            
            # dim of out: (batch_size, max_trg_seq_length-1, d_model)
            _, out, mu, log_var, _, _, _, _ = model.forward(src, trg, conds)

            # compute training loss and do back-propagation
            # rec_loss has not normalized by the number of tokens here
            # KL_div is the average value of a batch
            rec_loss, KL_div = loss_compute(out, trg_y, ntokens, mu, log_var) 

            total_loss += float(rec_loss + KL_div)
            total_rec_loss += float(rec_loss)
            total_KL_div += float(KL_div)

            # compute accuracy of predicted smiles
            smiles = decode.decode(model, src, conds, self.max_len_trg, 
                                   type='greedy', use_cond2dec=True)

            for j in range(trg.size()[0]):
                seq = smiles[j, :]
                target = trg[j]
                target = tokenizer.untokenize(vocab.decode(target.cpu().numpy()))
                seq = tokenizer.untokenize(vocab.decode(seq.cpu().numpy()))
                if seq == target:
                    n_correct += 1            
    
            n_rounds += 1
            total_tokens += ntokens
            total_n_trg += trg.size(0)
            
        return (n_correct*1.0 / total_n_trg, # the overall accuracy
                total_rec_loss / total_tokens, # the average reconstruction loss per token
                total_KL_div / n_rounds) # the average KL divergence per prediction (smiles)


    def validation_stat(self, dataloader, model, loss_compute, device, vocab):
        tokenizer = mv.SMILESTokenizer()

        total_loss = 0 # total loss
        total_rec_loss = 0 # total reconstruction loss
        total_KL_div = 0 # total KL divergence
        total_tokens = 0 # total non-padded tokens
        n_correct = 0 # total number of correct predicted smiles
        total_n_trg = 0 # total number of target smiles
        n_rounds = 0 # the number of batches

        for _, batch in enumerate(ul.progress_bar(dataloader, total=len(dataloader))):
            src, trg, conds, _ = batch

            trg_y = trg[:, 1:] # the expected output
            trg = trg[:, :-1] # the input of the model
            ntokens = float((trg_y != self.pad_idx).data.sum()) # num tokens not padding

            # CPU to GPU
            src = src.to(device)
            trg_y = trg.to(device)
            trg = trg.to(device)
            conds = conds.to(device)

            with torch.no_grad():
                # dim of out: (batch_size, max_trg_seq_length-1, d_model)
                _, out, mu, log_var, _, _, _, _ = model.forward(src, trg, conds)
                rec_loss, KL_div = loss_compute(out, trg_y, ntokens, mu, log_var) 

                total_loss += float(rec_loss + KL_div)
                total_rec_loss += float(rec_loss)
                total_KL_div += float(KL_div)

                # Decode
                smiles = decode.decode(model, src, conds, self.max_len_trg, 
                                       type='greedy', use_cond2dec=True)

                # Compute accuracy
                for j in range(trg.size()[0]):
                    seq = smiles[j, :]
                    target = trg[j]
                    target = tokenizer.untokenize(vocab.decode(target.cpu().numpy()))
                    seq = tokenizer.untokenize(vocab.decode(seq.cpu().numpy()))
                    if seq == target:
                        n_correct += 1

            n_rounds += 1
            total_tokens += ntokens
            total_n_trg += trg.size()[0]

        return (n_correct / total_n_trg, # the overall accuracy
                total_rec_loss / total_tokens, # the average reconstruction loss per token
                total_KL_div / n_rounds) # the average KL divergence per prediction (smiles)



    
    # def to_tensorboard(self, train_loss, validation_loss, accuracy, epoch):
    #     self.summary_writer.add_scalars("loss", {"train": train_loss, "validation": validation_loss}, epoch)
    #     self.summary_writer.add_scalar("accuracy/validation", accuracy, epoch)
    #     self.summary_writer.close()


    def _get_model_parameters(self, vocab_size):
        return {
            'N': self.opt.N,
            'H': self.opt.H,
            'd_model': self.opt.d_model,
            'd_ff': self.opt.d_ff,
            'latent_dim': self.opt.latent_dim,
            'dropout': self.opt.dropout,
            'nconds': self.opt.nconds,
            'vocab_size': vocab_size,
        }

    def save(self, model, optim, vocab_size, model_name):
        # save optimizer and hyperparameters
        file_name = os.path.join(self.save_path, "checkpoint", model_name)
        uf.make_directory(file_name, is_dir=False)
        save_dict = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.save_state_dict(),
            'model_parameters': self._get_model_parameters(vocab_size)
        }
        torch.save(save_dict, file_name)

    def train(self):
        device = ut.allocate_gpu()

        # Load vocabulary        
        with open(os.path.join(self.opt.data_path, 'vocab.pkl'), "rb") as input_file:
            vocab = pkl.load(input_file)
        vocab_size = len(vocab.tokens())

        # training/validation data loader
        dataloader_train = self.initialize_dataloader(vocab, 'train')
        dataloader_validation = self.initialize_dataloader(vocab, 'validation')

        model = self.get_model(vocab, device)
        optim = self.get_optimization(model)
        criterion = Loss(size=len(vocab), padding_idx=self.pad_idx, smoothing=self.opt.label_smoothing)

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

        while epoch < n_epochs:
            self.LOG.info("Starting EPOCH #%d", epoch)
            self.LOG.info("Training start")
            model.train()

            acc_train, rec_loss_train, KL_div_train = self.train_epoch(dataloader_train, model,
                                                      SimpleLossCompute(criterion, optim), device, vocab)
            self.LOG.info("Training end")
            self.LOG.info("Validation start")
            
            model.eval()

            acc_val, rec_loss_val, KL_div_val = self.validation_stat(dataloader_validation, model, 
                                                SimpleLossCompute(criterion, None), device, vocab)

            self.LOG.info("Validation end")
            self.LOG.info("Train:Acc/rec_loss/kl_div {:.6},{:.6},{:.6}  Validation:Acc/rec_loss/kl_div {:.6},{:.6},{:.6}".
                          format(acc_train, rec_loss_train, KL_div_train, acc_val, rec_loss_val, KL_div_val))
                
            loss_acc_writer.info("Train:Acc/rec_loss/kl_div {:.6},{:.6},{:.6}  Validation:Acc/rec_loss/kl_div {:.6},{:.6},{:.6}".
                            format(acc_train, rec_loss_train, KL_div_train, acc_val, rec_loss_val, KL_div_val))

            if lowest_loss > rec_loss_val + KL_div_val:
                # store lowest-loss new model every time is safer
                if os.path.exists(os.path.join(self.save_path, "checkpoint", f"best_{epoch_best}.pt")):
                    os.remove(os.path.join(self.save_path, "checkpoint", f"best_{epoch_best}.pt"))
                    
                self.save(model, optim, vocab_size, f"best_{epoch}.pt")
                lowest_loss = rec_loss_val + KL_div_val
                model_best = model
                optim_best = optim
                epoch_best = epoch
                early_stop_cnt = 0
            else:
                early_stop_cnt += 1

            self.save(model, optim, vocab_size, f"model_{epoch}.pt")
            epoch += 1

            if early_stop_cnt > early_stop:
                break

        self.save(model_best, optim_best, vocab_size, f"best_{epoch_best}.pt")
