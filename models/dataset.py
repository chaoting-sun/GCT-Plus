# coding=utf-8

"""
Implementation of a SMILES dataset.
"""
import pandas as pd

import torch
import torch.utils.data as tud
from torch.autograd import Variable

import configuration.config_default as cfgd
from models.VAETransformer.mask import subsequent_mask


class Dataset(tud.Dataset):
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
        self._prediction_mode = prediction_mode

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

    @classmethod
    def collate_fn(cls, data_all):
        """
        (a) Input
        :data_all: [(source, conds, row), ...] if in prediction mode else [(source, target, conds, row), ...]   
        - source: a batch of encoded source
        - target: a batch of encoded target
        - conds: a batch of conditions
        - row: a row of DataFrame
        (b) Output
        :collated_source: a batch of padded encoded source
        :collated_target: a batch of padded encoded target
        :conds: a batch of a number of conditions
        :data: a DataFrame with a batch of data
        """
        # check if in prediction mode
        is_prediction_mode = True if len(data_all[0]) == 3 else False
 
        if is_prediction_mode:
            source_encoded, conds, data = zip(*data_all)
            data = pd.DataFrame(data)
        else:
            source_encoded, target_encoded, conds, data = zip(*data_all)
            data = pd.DataFrame(data)

        conds = torch.tensor(conds, dtype=torch.float32)

        # create padded source
        max_length_source = max([seq.size(0) for seq in source_encoded]) # max. length of source
        collated_source = torch.zeros(len(source_encoded), max_length_source, dtype=torch.long)
        for i, seq in enumerate(source_encoded):
            collated_source[i, :seq.size(0)] = seq # encoded source SMILES

        # create padded target
        if not is_prediction_mode:
            max_length_target = max([seq.size(0) for seq in target_encoded])
            collated_target = torch.zeros(len(target_encoded), max_length_target, dtype=torch.long)
            for i, seq in enumerate(target_encoded):
                collated_target[i, :seq.size(0)] = seq
        else:
            collated_target = None
        
        return collated_source, collated_target, conds, data