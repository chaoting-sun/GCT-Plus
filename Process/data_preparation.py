import os
import numpy as np
import pandas as pd
import dill as pickle
import argparse
from typing import Dict, Tuple, List

from torchtext.legacy import data
from sklearn.model_selection import train_test_split

from .tokenizer import moltokenize

import utils.file as uf
import configuration.config_default as cfgd
# import Process.property_change_encoder as pce
import Process.batch as bt


import torch

SEED = 42
SPLIT_RATIO = 0.8


"""
get data from different dataset (moses, GuacaMol, ...)
return np.array each (sequence, conditions)
"""

def read_data_from_path(data_path: Dict[str, str]
                        )-> Dict[str, np.array]:
    """
    Access the dataset specified in the dictionary.
    Ex: 
    data_path = { "source train": "/source/path", "target train": "/target/path" }
    read_data_from_path(data_path)

    Args:
        data_path: a dictionary mapping the dataset name to its path
    """

    dataset = {}

    for key, value in data_path.items():
        try:
            dataset[key] = open(value, 'rt', encoding='UTF8').read().strip().split('\n')
        except:
            print("error: '" + key + "' file not found")
            quit()

    return dataset


def create_seq_fields(data_type="train",
                      lang_format="SMILES",
                      weights_path=None
                      )-> Tuple[data.field.Field, data.field.Field]:
    """
    Create fields for source and target.

    Args:
        data_type: dataset, such as 'train', 'validation', and 'test'
        lang_format: the supported language, which are SMILES and SELFIES.
        weights_path: the folder in which the fields of source and target are in. 
    """

    lang_supported = ['SMILES', 'SELFIES']

    if lang_format not in lang_supported:
        print('invalid src language: {0} supported languages: {1}'.format(lang_format, lang_supported))

    print("loading molecule tokenizers...")

    t_src = moltokenize()
    t_trg = moltokenize()

    SRC = data.Field(tokenize=t_src.tokenizer, batch_first=True)
    TRG = data.Field(tokenize=t_trg.tokenizer, batch_first=True, init_token='<sos>', eos_token='<eos>')

    if data_type == 'train' and weights_path is not None:
        try:
            print("loading presaved fields...")
            SRC = pickle.load(open(f'{weights_path}/SRC.pkl', 'rb'))
            TRG = pickle.load(open(f'{weights_path}/TRG.pkl', 'rb'))
        except:
            print("error opening SRC.pkl and TRG.pkl field files, please ensure they are in " + weights_path + "/")
            quit()

#     return (SRC, TRG)


# class PreprocessWrapper(object):
#     def __init__(self, 
#                  opt: argparse.Namespace, 
#                  source: List[str], 
#                  target: List[str], 
#                  SRC: data.field.Field, 
#                  TRG: data.field.Field):

#         self.opt = opt
#         self.SRC = SRC
#         self.TRG = TRG
#         self.source = source
#         self.target = target
        
#         self.field_type = ["src", "trg"]

#     def extend_fields(self):
#         if self.opt.cond_dim > 0:
#             self.fields.extend(self.opt.cond_list)
    
#     def create_dataset(self):
#         pass

    # def create_dataloader(self):
    #     pass


def mask_invalid_len_data(df, cond_dim, max_strlen):
    if cond_dim


def create_dataset(opt: argparse.Namespace,
                   SRC: data.field.Field, 
                   TRG: data.field.Field, 
                   tr_te: str,
                   lang_format="SMILES"
                   )-> bt.MyIterator:

    print("-------------------->", type(opt))

    raw_data = {'src': [line for line in opt.source], 'trg': [line for line in opt.target]}
    df = pd.DataFrame(raw_data, columns=["src", "trg"])
    if opt.cond_dim > 0:
        df = pd.concat([df, opt.conds], axis=1)

    # masking data longer than max_strlen

    if lang_format == 'SMILES':
        mask = (df['src'].str.len() + opt.cond_dim < opt.max_strlen) \
             & (df['trg'].str.len() + opt.cond_dim < opt.max_strlen)
    elif lang_format == 'SELFIES':
        mask = (df['src'].str.count('][') + opt.cond_dim < opt.max_strlen) \
             & (df['trg'].str.count('][') + opt.cond_dim < opt.max_strlen)
    df = df.loc[mask]

    data_path = os.path.join(opt.data_folder, './DB_temp.csv')

    if tr_te == "train":
        print("     - # of training samples:", len(df.index))
        df.to_csv(data_path, index=False)
    elif tr_te == "test":
        print("     - # of test samples:", len(df.index))
        df.to_csv(data_path, index=False)
    elif tr_te == "attn_test":
        print("     - # of attn_test samples:", len(df.index))
        df = pd.concat([df]*80, ignore_index=True)
        df.to_csv(data_path, index=False)

    data_fields = [('src', SRC), ('trg', TRG)]
    if opt.cond_dim > 0:
        for c in opt.cond_list:
            c_field = data.Field(use_vocab=False, sequential=False, batch_first=True, dtype=torch.float)
            data_fields.append((c, c_field))

    if tr_te == "train":
        toklenList = [] 
        dataset = data.TabularDataset(data_path, format='csv', fields=data_fields, skip_header=True)

        for i in range(len(dataset)):
            toklenList.append(len(vars(dataset[i])['src']))
        df_toklenList = pd.DataFrame(toklenList, columns=["toklen"])
        df_toklenList.to_csv(os.path.join(opt.data_folder, "toklen_list.csv"), index=False)
        if opt.verbose == True:
            print(" - tokenized training sample 0:", vars(dataset[0]))
    elif tr_te == "test":
        dataset = data.TabularDataset(data_path, format='csv', fields=data_fields, skip_header=True)
        if opt.verbose == True:
            print(" - tokenized testing sample 0:", vars(dataset[0]))
    elif tr_te == "attn_test":
        dataset = data.TabularDataset(data_path, format='csv', fields=data_fields, skip_header=True)
        if opt.verbose == True:
            print(" - tokenized attention testing sample 0:", vars(dataset[0]))

    if tr_te == "train":
        if opt.load_weights is False:
            print(" - building vocab from train data...")
            SRC.build_vocab(dataset)

            if True:
                print("The first sample in the dataset:")
                print(" - dict-key:", dataset[0].__dict__.keys())
                print(" - source:", dataset[0].src)

            if opt.verbose == True:
                print(' - vocab size of SRC: {}\n -> {}'.format(len(SRC.vocab), SRC.vocab.stoi))
            TRG.build_vocab(dataset)
            if opt.verbose == True:
                print(' - vocab size of TRG: {}\n -> {}'.format(len(TRG.vocab), TRG.vocab.stoi))
            if opt.checkpoint > 0:
                try:
                    os.mkdir("weights")
                except:
                    print("weights folder already exists, run program with -load_weights weights to load them")
                    quit()
                pickle.dump(SRC, open('weights/SRC.pkl', 'wb'))
                pickle.dump(TRG, open('weights/TRG.pkl', 'wb'))

        opt.src_pad = SRC.vocab.stoi['<pad>']
        opt.trg_pad = TRG.vocab.stoi['<pad>']
        assert opt.src_pad == opt.trg_pad

        data_iter = data.Iterator(dataset, batch_size=opt.batch_size, sort_key=lambda x: (len(x.src),len(x.trg)), 
                                  device=opt.device, train=True, repeat=False, shuffle=True)

        opt.train_len = sum(1 for _ in data_iter)

    elif tr_te == "test":
        opt.test_len = sum(1 for _ in data_iter)

    elif tr_te == "attn_test":
        opt.src_pad = SRC.vocab.stoi['<pad>']
        opt.trg_pad = TRG.vocab.stoi['<pad>']
        opt.test_len = 1

    return data_iter


# def split_data(input_transformations_path, LOG=None):
#     """
#     Split data into training, validation and test set, write to files
#     :param input_transformations_path:L
#     :return: dataframe
#     """
#     data = pd.read_csv(input_transformations_path, sep=",")
#     if LOG:
#         LOG.info("Read %s file" % input_transformations_path)

#     train, test = train_test_split(
#         data, test_size=0.1, random_state=SEED)
#     train, validation = train_test_split(train, test_size=0.1, random_state=SEED)
#     if LOG:
#         LOG.info("Train, Validation, Test: %d, %d, %d" % (len(train), len(validation), len(test)))

#     parent = uf.get_parent_dir(input_transformations_path)
#     train.to_csv(os.path.join(parent, "train.csv"), index=False)
#     validation.to_csv(os.path.join(parent, "validation.csv"), index=False)
#     test.to_csv(os.path.join(parent, "test.csv"), index=False)

#     return train, validation, test