import os
import sys
import time
import pandas as pd
from multiprocessing import Pool
import torch
from torchtext import data
from torch.utils.data import Dataset

from Utils.smiles import smi_to_mol, randomize_smiles
from Utils.properties import property_fn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from Model.collate_fn import get_collate_fn
import random


def measure_time(f):
    def timed(*args, **kw):
        global io_elapsed_time
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        io_elapsed_time += te-ts
        return result
    return timed


def get_condition(dataset, condition_list, n_jobs=1) -> pd.DataFrame:
    """
    Compute the properties of the source and target in dataset
    Return a DataFrame of given properties

    Arguments:
        dataset: a list of SMILES
        condition_list: the list of property names
        condition_path: the conditions to download
        n_jobs: number of processes
    """
    condition_dict = {}

    pool = Pool(n_jobs)
    mol = list(pool.map(smi_to_mol, dataset))

    for prop in condition_list:
        results = pool.map(property_fn[prop], mol)
        condition_dict[prop] = list(results)
    return pd.DataFrame.from_dict(condition_dict)


def to_tokenlist(dataset, data_path):
    toklenList = []
    for i in range(len(dataset)):
        toklenList.append(len(vars(dataset[i])['src']))
    df_toklenList = pd.DataFrame(toklenList, columns=["toklen"])
    df_toklenList.to_csv(os.path.join(
        data_path, "toklen_list.csv"), index=False)


class Batch:
    def __init__(self, src, trg_en, trg, pad_idx, device,
                 src_cond=None, trg_cond=None, dif_cond=None):
        # input of encoder
        self.src = src.to(device)
        if trg_en is not None:
            self.trg_en = trg_en.to(device)
        # output of the decoder
        if trg is not None:
            self.trg_y = trg[:, 1:].to(device)
            self.trg = trg[:, :-1].to(device)

        if None not in (src_cond, dif_cond, trg_cond):
            self.econds = src_cond.to(device)
            self.mconds = torch.cat([src_cond, dif_cond], dim=1).to(device)
            self.dconds = trg_cond.to(device)


def padding(obj, max_strlen, cond_len, pad_id):
    obj_pad = torch.ones(obj.size(0), abs(max_strlen - obj.size(1)
                       - cond_len), dtype=torch.long) * pad_id
    return torch.cat([obj, obj_pad], dim=1)


def rebatch(batch, conds, pad_idx, max_strlen, device):
    src = padding(batch.src.transpose(0,1), max_strlen, len(conds), pad_idx)
    trg = trg_en = None
    if hasattr(batch, 'trg'):
        trg = padding(batch.trg.transpose(0,1), max_strlen, len(conds), pad_idx)
    if hasattr(batch, 'trg_en'):
        trg_en = padding(batch.trg_en.transpose(0,1), max_strlen, len(conds), pad_idx)

    if len(conds) > 0:
        # TODO: initialize by a tensor
        # src_conds = torch.zeros((len(batch),))
        # for i, c in enumerate(conds):
        src_conds, trg_conds = [], []
        for c in conds:
            src_conds.append(getattr(batch, f"src_{c}").view(-1, 1))
            trg_conds.append(getattr(batch, f"trg_{c}").view(-1, 1))
        src_conds = torch.cat(src_conds, dim=1)
        trg_conds = torch.cat(trg_conds, dim=1)
        del_conds = torch.sub(trg_conds, src_conds)
        return Batch(src, trg_en, trg, pad_idx, device,
                     src_conds, trg_conds, del_conds)
    else:
        return Batch(src, trg_en, trg, pad_idx, device)


def to_dataloader(data_iter, conditions, pad_idx, max_strlen, device):
    return (rebatch(batch, conditions, pad_idx, max_strlen, device)
            for batch in data_iter)


def rebatch2(batch, conds, pad_idx, device):
    src = batch.src.transpose(0, 1)
    trg = trg_en = None
    if getattr(batch, 'trg', None) is not None:
        trg = batch.trg.transpose(0, 1)
    if getattr(batch, 'trg_en', None) is not None:
        trg_en = batch.trg_en.transpose(0, 1)
    
    if len(conds) > 0:
        src_conds, trg_conds = [], []
        for c in conds:
            src_conds.append(getattr(batch, f"src_{c}").view(-1, 1))
            trg_conds.append(getattr(batch, f"trg_{c}").view(-1, 1))
        src_conds = torch.cat(src_conds, dim=1)
        trg_conds = torch.cat(trg_conds, dim=1)
        del_conds = torch.sub(trg_conds, src_conds)
        return Batch(src, trg_en, trg, pad_idx, device,
                     src_conds, trg_conds, del_conds)
    else:
        return Batch(src, trg_en, trg, pad_idx, device)


def get_dataloader(data_iter, conds, pad_idx, device):
    return (rebatch2(batch, conds, pad_idx, device)
            for batch in data_iter) 


# updated version

class BatchData:
    def __init__(self, src, trg=None, device=None,
                 econds=None, dconds=None, mconds=None):
        # input of encoder
        self.src = src.to(device)
        if trg is not None:
            self.trg_y = trg[:, 1:].to(device)
            self.trg = trg[:, :-1].to(device)
        if econds is not None:
            self.econds = econds.to(device)
        if dconds is not None:
            self.dconds = dconds.to(device)
        if mconds is not None:
            self.mconds = mconds.to(device)


# def rebatch_data(src, econds=None, trg=None, dconds=None, pad_id=None,
#                  max_strlen=None, pad_to_same_len=False, device=None,
#                  include_delconds=True):
#     def padding(obj, cond_len):
#         obj_pad = torch.ones(obj.size(0), abs(max_strlen - obj.size(1)
#                         - cond_len), dtype=torch.long) * pad_id
#         return torch.cat([obj, obj_pad], dim=1)

#     batch_data = {}
#     batch_data['device'] = device
#     batch_data['src'] = padding(src, econds.size(-1)) if pad_to_same_len else src

#     if econds is not None:
#         batch_data['econds'] = econds
#     if trg is not None:
#         batch_data['trg'] = padding(trg, dconds.size(-1)) if pad_to_same_len else trg
#     if dconds is not None:
#         batch_data['dconds'] = dconds
#     if include_delconds:
#         batch_data['mconds'] = torch.sub(dconds, econds)
    
#     return BatchData(**batch_data)


def get_tensor_size(t):
    return t.element_size() * t.nelement()


def rebatch_data(src, econds=None, trg=None, dconds=None, pad_id=None,
                 max_strlen=None, pad_to_same_len=False, device=None,
                 include_mconds=True):
    
    def padding(obj, cond_len):
        obj_pad = torch.ones(obj.size(0), abs(max_strlen - obj.size(1)
                        - cond_len), dtype=torch.long) * pad_id
        return torch.cat([obj, obj_pad], dim=1)

    if src is not None and pad_to_same_len:
        src = padding(src, econds.size(-1))
    src = src.transpose(0, 1)

    if trg is not None and pad_to_same_len:
        trg = padding(trg, econds.size(-1)).transpose(0, 1)
    trg = trg.transpose(0, 1)

    mconds = torch.sub(dconds, econds) if include_mconds else None

    return BatchData(src=src, trg=trg, econds=econds, dconds=dconds,
                     mconds=mconds, device=device)


def get_dataset(data_folder, fields, file_name_list):
    return data.TabularDataset.splits(
        path=data_folder,
        train=f'{file_name_list[0]}.csv' if file_name_list[0] else None,
        validation=f'{file_name_list[1]}.csv' if file_name_list[1] else None,
        test=f'{file_name_list[2]}.csv' if file_name_list[2] else None,
        format='csv',
        fields=fields,
        skip_header=True
    )


def get_iterator(train, valid, batch_size):
    train_iter, valid_iter = data.BucketIterator.splits(
        (train, valid), batch_sizes=(batch_size, batch_size),
        sort_key=lambda x: (len(x.src), len(x.trg))
    )
    return train_iter, valid_iter


def get_loader(data_iter, property_list, pad_id, max_strlen,
               pad_to_same_len=False, device=None, include_mconds=False):
    def extract_conds(batch, data_type):
        prop_vals = []
        for p in property_list:
            prop_vals.append(getattr(batch, f"{data_type}_{p}").view(-1, 1))
        return torch.cat(prop_vals, dim=1)
    
    return (rebatch_data(getattr(batch, 'src', None), 
                         extract_conds(batch, 'src'),
                         getattr(batch, 'trg', None),
                         extract_conds(batch, 'trg'),
                         pad_id,
                         max_strlen,
                         pad_to_same_len,
                         device,
                         include_mconds
                         )
            for batch in data_iter)


class SmilesDataset(Dataset):
    def __init__(self, input_data, property_list, SRC=None, TRG=None,
                 include_mconds=False, use_scaffold=False,
                 randomize=False):
        self.SRC = SRC
        self.TRG = TRG
        self.data = input_data
        self.property_list = property_list
        self.include_mconds = include_mconds
        self.use_scaffold = use_scaffold
        self.randomize = randomize
    
    def tokenize_smiles(self, smi, field):
        if self.randomize and random.random() > 0.7:
            smi = randomize_smiles(smi)
        return field.tokenize(smi)
        
    def __getitem__(self, rid):
        item = {}
        row = self.data.iloc[rid]
        
        if 'src' in row:
            item['src'] = self.tokenize_smiles(row['src'], self.SRC)            
            if len(self.property_list) > 0:
                item['econds'] = [row[f'src_{p}'] for p in self.property_list]
            
        if 'trg' in row:
            item['trg'] = self.tokenize_smiles(row['trg'], self.TRG)
            if len(self.property_list) > 0:
                item['dconds'] = [row[f'trg_{p}'] for p in self.property_list]

        if self.use_scaffold:
            if 'src_scaffold' in row:
                item['src_scaffold'] = self.tokenize_smiles(row['src_scaffold'], self.SRC)
            if 'trg_scaffold' in row:
                item['trg_scaffold'] = self.tokenize_smiles(row['trg_scaffold'], self.TRG)

        if self.include_mconds:
            item['mconds'] = [item['dconds'][i] - item['econds'][i]
                            for i in range(len(self.property_list))]
        return item

    def __len__(self):
        return len(self.data)


# class DataloaderPreparation:
#     def __init__(self, rank, SRC, TRG, batch_size, model_type,
#                  property_list, world_size=1, randomize=False,
#                  use_scaffold=False):
#         self.SRC = SRC
#         self.TRG = TRG
#         self.rank = rank
#         self.world_size = world_size
#         self.batch_size = batch_size
#         self.property_list = property_list
#         self.use_scaffold = use_scaffold
#         self.randomize = randomize
#         self.collate_fn = get_collate_fn(model_type, SRC, TRG, rank)
    
#     def _get_sampler(self, dataset, shuffle=False,
#                      drop_last=False):
#         return DistributedSampler(dataset, self.world_size, self.rank,
#                                   shuffle, drop_last)

#     def _get_dataset(self, dataframe, include_mconds=True):
#         return SmilesDataset(dataframe, self.property_list, self.SRC,
#                              self.TRG, include_mconds, self.use_scaffold,
#                              self.randomize)

#     def get_dataloader(self, dataframe, is_train,
#                        include_mconds=False, shuffle=False,
#                        sampler=None):
#         dataset = self._get_dataset(dataframe, include_mconds)
    
#         if self.world_size > 1:
#             sampler = self._get_sampler(dataset,
#                 shuffle=True if is_train else False)
#         if self.world_size == 1 and is_train:
#             # no need to shuffle if sampler is used
#             shuffle = True

#         return DataLoader(dataset, batch_size=self.batch_size,
#                           drop_last=False, sampler=sampler,
#                           shuffle=shuffle, collate_fn=self.collate_fn,
#                           num_workers=0, pin_memory=False)
    

class DataloaderPreparation:
    def __init__(self, rank, SRC, TRG, model_type, property_list,
                 world_size=1, randomize=False, use_scaffold=False):
        self.SRC = SRC
        self.TRG = TRG
        self.rank = rank
        self.world_size = world_size
        self.property_list = property_list
        self.use_scaffold = use_scaffold
        self.randomize = randomize
        self.collate_fn = get_collate_fn(model_type, SRC, TRG, rank)
    
    def _get_sampler(self, dataset, shuffle=False,
                     drop_last=False):
        return DistributedSampler(dataset, self.world_size, self.rank,
                                  shuffle, drop_last)

    def _get_dataset(self, dataframe, include_mconds=True):
        return SmilesDataset(dataframe, self.property_list, self.SRC,
                             self.TRG, include_mconds, self.use_scaffold,
                             self.randomize)

    def get_dataloader(self, dataframe, batch_size, is_train,
                       include_mconds=False, shuffle=False,
                       sampler=None):
        dataset = self._get_dataset(dataframe, include_mconds)
    
        if self.world_size > 1:
            sampler = self._get_sampler(dataset,
                shuffle=True if is_train else False)
        if self.world_size == 1 and is_train:
            # no need to shuffle if sampler is used
            shuffle = True

        return DataLoader(dataset, batch_size=batch_size, drop_last=False,
                          sampler=sampler, shuffle=shuffle,
                          collate_fn=self.collate_fn, num_workers=0,
                          pin_memory=False)