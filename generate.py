import numpy as np
import pickle as pkl
import os
import argparse
import pandas as pd

import torch

import utils.chem as uc
import utils.torch_util as ut
import utils.log as ul
import utils.plot as up
import configuration.config_default as cfgd
import models.dataset as md
import preprocess.vocabulary as mv
import configuration.opts as opts
from models.transformer_v1 import decode, transformer


class GenerateRunner(object):

    def __init__(self, opt):
        self.opt = opt        
        self.save_path = os.path.join('experiments', opt.save_directory, opt.test_file_name,
                                      f'evaluation_{opt.epoch}')
        global LOG
        LOG = ul.get_logger(name="generate", log_path=os.path.join(self.save_path, 'generate.log'))
        LOG.info(opt)
        LOG.info("Save directory: {}".format(self.save_path))

        # Load vocabulary
        with open(os.path.join(opt.data_path, 'vocab.pkl'), "rb") as input_file:
            vocab = pkl.load(input_file)
        self.vocab = vocab
        self.tokenizer = mv.SMILESTokenizer()

    def initialize_dataloader(self):
        """
        Initialize dataloader
        :param opt:
        :param vocab: vocabulary
        :param test_file: test_file_name
        :return:
        """

        # Read test
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
        
        # Load model
        file_name = os.path.join(self.opt.model_path, f'model_{self.opt.epoch}.pt')
        
        model = transformer.load_from_file(file_name)
        model.to(device)
        model.eval()
        
        max_len = cfgd.DATA_DEFAULT['max_sequence_length']
        df_list = []
        sampled_smiles_list = []
          
        for j, batch in enumerate(ul.progress_bar(dataloader_test, total=len(dataloader_test))):

            src, source_length, _, src_mask, _, _, df = batch
            # src.shape: torch.Size([batch_size, 70])
            # source_length: torch.Size([batch_size])
            # src_mask: torch.Size([batch_size, 1, 70])
            # Move to GPU
            src = src.to(device)
            src_mask = src_mask.to(device)

            smiles = self.sample(model, src, src_mask, max_len=max_len, device=device)
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


    def sample(self, model, src, src_mask, max_len=cfgd.DATA_DEFAULT['max_sequence_length'], device=None):
        batch_size = src.shape[0]
        num_valid_batch = np.zeros(batch_size)  # current number of unique and valid samples out of total sampled
        unique_set_num_samples = [list()] * batch_size  # for each starting molecule

        sequences_all = torch.ones((self.opt.num_samples, batch_size, max_len)).type(torch.LongTensor)
        max_trials = self.opt.num_samples  # Maximum trials for sampling
        
        if self.opt.decode_type == 'greedy':
            max_trials = 1

        if src is not None:
            for ibatch in range(batch_size):
                source_smi = self.tokenizer.untokenize(self.vocab.decode(src[ibatch].tolist()[1: ])) # remove property change and untokenize
                source_smi = uc.get_canonical_smile(source_smi)
                # unique_set_num_samples[ibatch].add(source_smi)
                unique_set_num_samples[ibatch].append(source_smi)

        with torch.no_grad():            
            for i in range(max_trials):
                sequences = decode.decode(model, src, src_mask, max_len, self.opt.decode_type) # shape: (batch_size, max_length_source)
                # https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html#torch.nn.functional.pad
                # padding = (padding_left, padding_right, padding_top, padding_bottom)
                padding = (0, max_len-sequences.shape[1], 0, 0) # max_len: self-defined (128)
                sequences = torch.nn.functional.pad(sequences, padding) 
                
                for ibatch in range(batch_size):
                    num_valid_batch[ibatch] += 1
                    sequences_all[int(num_valid_batch[ibatch] - 1), ibatch, :] = sequences[ibatch]

        
        # Convert to SMILES
        smiles_list = [] # [batch, topk]
        seqs = np.asarray(sequences_all.numpy())
        # [num_sample, batch_size, max_len]
        batch_size = len(seqs[0])
        
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
