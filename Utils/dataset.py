# general modules
import io
import os
import time
import gc
import joblib
import sqlite3
import pandas as pd
import dill as pickle
from multiprocessing import Pool
from collections import OrderedDict
import numpy as np

# ml modules
import moses
import torch
from torchtext import data
# from torchtext.legacy import data
from torch.utils.data import Dataset

# other packages
from .property import to_mol, property_prediction
from Utils.chrono import Chrono


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
    mol = list(pool.map(to_mol, dataset))

    for prop in condition_list:
        results = pool.map(property_prediction[prop], mol)
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


def rebatch_data(src, econds=None, trg=None, dconds=None, pad_id=None,
                 max_strlen=None, pad_to_same_len=False, device=None,
                 include_delconds=True):
    def padding(obj, cond_len):
        obj_pad = torch.ones(obj.size(0), abs(max_strlen - obj.size(1)
                        - cond_len), dtype=torch.long) * pad_id
        return torch.cat([obj, obj_pad], dim=1)

    batch_data = {}
    batch_data['device'] = device

    src = src.transpose(0, 1)
    batch_data['src'] = padding(src, econds.size(-1)) if pad_to_same_len else src
    if econds is not None:
        batch_data['econds'] = econds
    if trg is not None:
        trg = trg.transpose(0, 1)
        batch_data['trg'] = padding(trg, dconds.size(-1)) if pad_to_same_len else trg
    if dconds is not None:
        batch_data['dconds'] = dconds
    if include_delconds:
        batch_data['mconds'] = torch.sub(dconds, econds)
    
    return BatchData(**batch_data)


def get_loader(data_iter, conditions, pad_id, max_strlen,
               pad_to_same_len=False, device=None):
    def extract_conds(batch, data_type):
        cond_vals = []
        for c in conditions:
            cond_vals.append(getattr(batch, f"{data_type}_{c}").view(-1, 1))
        return torch.cat(cond_vals, dim=1)

    return (rebatch_data(batch.src, extract_conds(batch, 'src'),
                         batch.trg, extract_conds(batch, 'trg'),
                         pad_id, max_strlen, pad_to_same_len, device)
            for batch in data_iter)