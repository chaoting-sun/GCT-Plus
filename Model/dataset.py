# coding=utf-8

"""
Implementation of a SMILES dataset.
for "mlp" model
"""
import pandas as pd

import torch
from torch.utils.data import Dataset

import Configuration.config_default as cfgd


class mlpDataset(Dataset):
    def __init__(self, data, vocabulary, tokenizer, scaler=None, prediction_mode=False):
        """
        :param data: dataframe read from training, validation or test file
        :param vocabulary: used to encode source/target tokens
        :param tokenizer: used to tokenize source/target smiles
        :param prediction_mode: if use target smiles or not (training or test)
        """
        self._vocabulary = vocabulary
        self._tokenizer = tokenizer
        self._data = data
        self._scaler = scaler


    def __getitem__(self, i):
        row = self._data.iloc[i]

        # conditions (real numbers)
        conds = []
        for cond_name in cfgd.PROPERTIES:
            conds.append(row['Delta_{}'.format(cond_name)])
        if self._scaler is not None: # re-scale the values
            conds = self._scaler.transform(conds)
        
        # encode source SMILES
        source_encoded = self._vocabulary.encode(self._tokenizer.tokenize(row['Source_Mol']))
        source_encoded = torch.tensor(source_encoded, dtype=torch.long)

        if self._prediction_mode: # for testing
            return source_encoded, conds, row
        else: # for training
            # encode target SMILES
            target_encoded = self._vocabulary.encode(self._tokenizer.tokenize(row['Target_Mol']))
            target_encoded = torch.tensor(target_encoded, dtype=torch.long)
            return source_encoded, target_encoded, conds, row

    def __len__(self):
        return len(self._data)