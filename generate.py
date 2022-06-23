import os
import argparse
import pickle as pkl
import numpy as np
import pandas as pd


import Utils.log as ul
import Model.dataset as md
import Utils.gpu as ut
import Configuration.opts as opts
import Process.vocabulary as mv
import Configuration.config_default as cfgd
from Model.cvae_Transformer import decode, transformer

import torch


class GenerateRunner(object):

    def __init__(self, opt):
        self.opt = opt      
  
        self.save_path = os.path.join('experiments', opt.save_directory, opt.test_file_name,
                                      f'evaluation_{opt.epoch}')
        self.model_path = os.path.join(self.opt.model_path, f'model_{self.opt.epoch}.pt')

        global LOG
        LOG = ul.get_logger(name="generate", log_path=os.path.join(self.save_path, 'generate.log'))
        LOG.info(opt)
        LOG.info("Save directory: {}".format(self.save_path))

        with open(os.path.join(opt.data_path, 'vocab.pkl'), "rb") as input_file:
            vocab = pkl.load(input_file)
        self.vocab = vocab
        self.tokenizer = mv.SMILESTokenizer()

    def initialize_dataloader(self):
        data = pd.read_csv(os.path.join(self.opt.data_path, self.opt.test_file_name + '.csv'), sep=",")
        dataset = md.Dataset(data=data, vocabulary=self.vocab, tokenizer=self.tokenizer, prediction_mode=True)
        dataloader = torch.utils.data.DataLoader(dataset, self.opt.batch_size,
                                                 shuffle=False, collate_fn=md.Dataset.collate_fn)
        return dataloader

    def generate(self):
        # set device
        device = ut.allocate_gpu()
        
        # Data loader
        dataloader_test = self.initialize_dataloader()
        
        # prepare model
        model = transformer.load_from_file(self.model_path)
        model.to(device)
        model.eval()
        
        max_len = cfgd.DATA_DEFAULT['max_sequence_length']
        df_list = []
        sampled_smiles_list = []
          
        for j, batch in enumerate(ul.progress_bar(dataloader_test, total=len(dataloader_test))):
            src, _, conds, df = batch

            # CPU to GPU
            conds = conds.to(device)
            src = src.to(device)

            smiles = self.sample(model, conds, src, max_len=max_len, device=device)
            df_list.append(df)
            sampled_smiles_list.extend(smiles)
        
        # prepare dataframe
        data_sorted = pd.concat(df_list)
        sampled_smiles_list = np.array(sampled_smiles_list)
      
        for i in range(self.opt.num_samples):
            data_sorted[f'Predicted_smi_{i + 1}'] = sampled_smiles_list[:, i].copy()
  
        result_path = os.path.join(self.save_path, "generated_molecules.csv")
        LOG.info("Save to {}".format(result_path))
        data_sorted.to_csv(result_path, index=False)


    def sample(self, model, conds, src, max_len, device=None):
        batch_size = src.shape[0]
        num_valid_batch = np.zeros(batch_size)  # number valid samples out of total sampled
        sequences_all = torch.ones((self.opt.num_samples, batch_size, max_len)).type(torch.LongTensor)

        with torch.no_grad():            
            for i in range(self.opt.num_samples):
                sequences = decode.decode(model, src, conds, max_len, 
                                          self.opt.decode_type, use_cond2dec=self.opt.use_cond2dec)

                # padding: (padding_left, padding_right, padding_top, padding_bottom)
                # ref: https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html#torch.nn.functional.pad
                padding = (0, max_len-sequences.shape[1], 0, 0)
                sequences = torch.nn.functional.pad(sequences, padding) 
                
                for ibatch in range(batch_size):
                    num_valid_batch[ibatch] += 1
                    sequences_all[int(num_valid_batch[ibatch] - 1), ibatch, :] = sequences[ibatch]

        # Convert to SMILES
        smiles_list = []
        seqs = np.asarray(sequences_all.numpy())
        
        for ibatch in range(batch_size):
            topk_list = []
            for k in range(self.opt.num_samples):
                seq = seqs[k, ibatch, :]
                topk_list.extend([self.tokenizer.untokenize(self.vocab.decode(seq))])
            smiles_list.append(topk_list)
        
        return smiles_list


def run_main():
    """Main function."""
    
    parser = argparse.ArgumentParser(
        description='generate.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    opts.generate_opts(parser)
    opt = parser.parse_args()        
    
    runner = GenerateRunner(opt)
    runner.generate()


if __name__ == "__main__":
    run_main()
