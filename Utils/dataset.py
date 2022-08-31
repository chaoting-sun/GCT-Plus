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


# def get_benchmarking_dataset(data_name, data_type='train') -> pd.DataFrame:
#     """
#     Get the benchmarking dataset represented as strings
#     Returns a DataFrame with columns ['src_no', 'trg_no', 'src', 'trg']

#     Arguments:
#         data_name: 'moses' or 'guacamol'
#         data_type: 'train' or 'test'
#     """
#     if data_name == 'moses':
#         dataset = moses.get_dataset(data_type)
#     elif data_name == 'guacamol':
#         pass
#     dataset = pd.DataFrame.from_dict({
#         'no': [i+1 for i in range(len(dataset))],
#         'molecule': dataset,
#     })
#     return dataset


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
                 src_cond=None, dif_cond=None, trg_cond=None):
        # input of encoder
        self.src = src.to(device)
        self.trg_en = trg_en.to(device)
        # output of the decoder
        self.trg_y = trg[:, 1:].to(device)
        self.trg = trg[:, :-1].to(device)

        if src_cond is not None:
            self.econds = src_cond.to(device)
            self.mconds = torch.cat([src_cond, dif_cond], dim=1).to(device)
            self.dconds = trg_cond.to(device)


def rebatch(batch, conditions, pad_idx, max_strlen, device):
    src = batch.src.transpose(0, 1)
    trg_en = batch.trg_en.transpose(0, 1)
    trg = batch.trg.transpose(0, 1)

    src_pad = torch.ones((src.size(0), abs(max_strlen - src.size(1))),
                         dtype=torch.long) * pad_idx
    trg_en_pad = torch.ones((trg_en.size(0), abs(max_strlen - trg_en.size(1))),
                            dtype=torch.long) * pad_idx
    trg_pad = torch.ones((trg.size(0), abs(max_strlen - trg.size(1))),
                     dtype=torch.long) * pad_idx
    
    src = torch.cat([src, src_pad], dim=1)
    trg_en = torch.cat([trg_en, trg_en_pad], dim=1)
    trg = torch.cat([trg, trg_pad], dim=1)

    # src, trg = batch.src, batch.trg
    if len(conditions) > 0:
        src_conds, trg_conds = [], []
        for c in conditions:
            src_conds.append(getattr(batch, f"src_{c}").view(-1, 1))
            trg_conds.append(getattr(batch, f"trg_{c}").view(-1, 1))
        src_cond_t = torch.cat(src_conds, dim=1)
        trg_cond_t = torch.cat(trg_conds, dim=1)
        dif_cond_t = torch.sub(trg_cond_t, src_cond_t)
    else:
        src_cond_t = trg_cond_t = dif_cond_t = None
    return Batch(src, trg_en, trg, pad_idx, device,
                 src_cond_t, trg_cond_t, dif_cond_t)


def to_dataloader(data_iter, conditions, pad_idx, max_strlen, device):
    return (rebatch(batch, conditions, pad_idx, max_strlen, device) 
            for batch in data_iter)


def adapter(python_type):
    # adapt the Python type into an SQLite type
    out = io.BytesIO()
    np.save(out, python_type)
    return sqlite3.Binary(out.getvalue())


def converter(sqlite_object):
    # convert SQLite objects into a Python object
    return np.load(io.BytesIO(sqlite_object))


def sqlite_initialize(db_filepath):
    sqlite3.register_adapter(np.ndarray, adapter)
    sqlite3.register_converter("array", converter)
    con = sqlite3.connect(db_filepath, detect_types=sqlite3.PARSE_DECLTYPES)
    cur = con.cursor()
    # cur.execute(f"DROP TABLE IF EXISTS {table_name}")
    # cur.execute(f"CREATE TABLE IF NOT EXISTS {table_name} (idx integer, arr array)")
    cur.execute("PRAGMA cache_size = -163840") # kb
    # cur.execute("PRAGMA journal_mode = OFF")
    # cur.execute("PRAGMA synchronous = 0")
    # cur.execute("PRAGMA locking_mode = EXCLUSIVE")
    return con, cur


@Chrono
def torch_load(file_path):
    tensor = torch.load(file_path)
    return tensor


@Chrono
def pickle_load(file_path):
    mat = pickle.load(open(file_path, 'rb'))
    return mat
    # return torch.from_numpy(mat).to(device)


@Chrono
def np_load(file_path, device):
    # mat = np.memmap(file_path, dtype='float32', mode='c', shape=(80,512))
    mat = np.load(file_path, mmap_mode="c", allow_pickle=True)
    return mat


@Chrono
def memmap_tp_torch(memmap):
    return torch.from_numpy(memmap)


@Chrono
def to_device(tensor, device):
    return tensor.to(device)


@Chrono
def sqlite_select(cursor, no1, no2):
    table_name = "nparray"
    cursor.execute(f"SELECT arr FROM {table_name} "
                   f"WHERE idx IN (?, ?)", (no1, no2))
    records = cursor.fetchall()
    if no1 == no2:
        return records[0][0], records[0][0]
    return records[0][0], records[1][0]


# class mlpDataset(Dataset):
#     def __init__(self, conditions, encoder_outputs, pair_path,
#                  prop_path, device, transform=None):
#         self.conditions = conditions
#         self.encoder_outputs = encoder_outputs
#         self.transform = transform
#         self.device = device

#         self.pair_data = pd.read_csv(pair_path)
#         self.prop_data = pd.read_csv(prop_path)
#         # self.prop_data = self.prop_data.set_index('no')
#         self.prop_data = self.prop_data.set_index('no').T.to_dict('list')


#     def __len__(self):
#         return len(self.pair_data)

#     def __getitem__(self, idx):
#         row = self.pair_data.iloc[idx]
#         no1, no2 = row['no1'], row['no2']

#         src = memmap_tp_torch(self.encoder_outputs[no1-1])
#         trg = memmap_tp_torch(self.encoder_outputs[no2-1])
#         src = to_device(src, self.device)
#         trg = to_device(trg, self.device)

#         src_conds = torch.as_tensor(self.prop_data[no1],
#                                     dtype=torch.float32,
#                                     device=self.device)
#         trg_conds = torch.as_tensor(self.prop_data[no2],
#                                     dtype=torch.float32,
#                                     device=self.device)
#         dif_conds = torch.sub(trg_conds, src_conds)

#         mconds = torch.cat([src_conds, dif_conds]).clone().detach()

#         sample = { 'src': src, 'trg': trg, 'mconds': mconds }
#         return sample



class mlpDataset(Dataset):
    def __init__(self, conditions, mat_folder, pair_path, 
                 prop_path, device, batch_size, last_batch=1):
        self.conditions = conditions
        self.mat_folder = mat_folder
        self.device = device
        self.dtype = torch.float32

        self.pair_data = pd.read_csv(pair_path)
        self.prop_data = pickle.load(open(prop_path, "rb"))
        self.tensor_dict = OrderedDict()
        # self.prop_data = self.prop_data.set_index('no')
        # self.prop_data = self.prop_data.set_index('no').T.to_dict('list')

        self.batch_size = batch_size
        self.last_batch = last_batch
        self.num_samples = 0

    def __len__(self):
        return len(self.pair_data)

    def __getitem__(self, idx):
        if self.num_samples // self.batch_size < self.last_batch:
            self.num_samples += 1
            return { 'src': torch.empty(1), 'trg': torch.empty(1), 'mconds': torch.empty(1) }

        row = self.pair_data.iloc[idx]
        no1, no2 = row['no1'], row['no2']
        
        """
        method1
        1. preprocess: 將個別資料存成 torch array
        2. train: 讀檔案 -> 給模型使用
        """

        src = torch_load(os.path.join(self.mat_folder, f'{no1}.pt'))
        trg = torch_load(os.path.join(self.mat_folder, f'{no2}.pt'))

        # if no1 in self.tensor_dict:
        #     src = self.tensor_dict[no1]
        # else:
        #     src = torch_load(os.path.join(self.mat_folder, f'{no1}.pt'))

        # if no2 in self.tensor_dict:
        #     trg = self.tensor_dict[no2]
        # else:
        #     trg = torch_load(os.path.join(self.mat_folder, f'{no2}.pt'))

        # if len(self.tensor_dict) < 1000:
        #     self.tensor_dict[no1] = src
        #     self.tensor_dict[no2] = trg
            # del self.tensor_dict
            # self.tensor_dict = OrderedDict()
            # self.tensor_dict.popitem(last=False)
            # self.tensor_dict.popitem(last=False)

        """
        method2
        1. preprocess: 將個別資料存成 numpy array
        2. train: 讀檔案 -> 轉成 tensor -> 給模型使用
        comment: 一開始比 method1 快很多，後來平均會比 method1 慢一點
        """

        # if no1 in self.tensor_dict:
        #     src = self.tensor_dict[no1]
        # else:
        #     src = pickle_load(os.path.join(self.mat_folder, f'{no1}.pt'))
        #     # self.tensor_dict[no1] = src

        # if no2 in self.tensor_dict:
        #     trg = self.tensor_dict[no2]
        # else:
        #     trg = pickle_load(os.path.join(self.mat_folder, f'{no2}.pt'))
        #     self.tensor_dict[no2] = trg

        # if len(self.tensor_dict) > 20000:
        #     del self.tensor_dict
        #     self.tensor_dict = OrderedDict()

        # src = pickle_load(os.path.join(self.mat_folder, f'{no1}.pt'))
        # trg = pickle_load(os.path.join(self.mat_folder, f'{no2}.pt'))
        # src = torch.from_numpy(src).to(self.device, torch.float32)
        # trg = torch.from_numpy(trg).to(self.device, torch.float32)

        """
        method3
        1. preprocess: 將個別資料存成 numpy array
        2. train: 用 memory-mapped file 方式讀檔案 -> 轉成 tensor -> 給模型使用
        comment: 聽說比較快（省去很多system call），但沒比較快，還變慢
        """
        # src = np_load(os.path.join(self.mat_folder, f'{no1}.pt'), self.device)
        # trg = np_load(os.path.join(self.mat_folder, f'{no2}.pt'), self.device)
        # src = memmap_tp_torch(src).to(self.device)
        # trg = memmap_tp_torch(trg).to(self.device)
        
        """
        method4
        1. preprocess: 將個別資料用 numpy array 的型別存到 sqlite3
        2. train: 找檔案 -> 轉成 tensor -> 給模型使用
        comment: 後來發現找檔案要 O(lg n)，結果更慢...
        """
        # src, trg = sqlite_select(self.sqlite_cursor, int(no1), int(no2))
        # src = torch.from_numpy(src).to(self.device)
        # trg = torch.from_numpy(trg).to(self.device)

        src_conds = torch.as_tensor(self.prop_data[no1],
                                    dtype=torch.float32)
        trg_conds = torch.as_tensor(self.prop_data[no2],
                                    dtype=torch.float32)
        dif_conds = torch.sub(trg_conds, src_conds)
        mconds = torch.cat([src_conds, dif_conds]).detach()
        # mconds = torch.cat([src_conds, dif_conds]).clone().detach()

        sample = { 'src': src, 'trg': trg, 'mconds': mconds }
        # del src, trg, mconds, src_conds, trg_conds, dif_conds

        return sample


# class mlpDataset(Dataset):
#     def __init__(self, conditions, tensor_folder, pair_path,
#                  prop_path, device, transform=None):
#         self.conditions = conditions
#         self.tensor_folder = tensor_folder
#         self.transform = transform
#         self.device = device

#         self.pair_data = pd.read_csv(pair_path)
#         self.prop_data = pd.read_csv(prop_path)
#         # self.prop_data = self.prop_data.set_index('no')
#         self.prop_data = self.prop_data.set_index('no').T.to_dict('list')

#         self.cache = OrderedDict()
#         self.cacacity = 50000

#     def get_tensor(self, no):
#         if no in self.cache:
#             return self.cache[no]
#         else:
#             value = torch_load(os.path.join(self.tensor_folder, f'{no}.pt'))
#             self.put_tensor(no, value)
#             return value

#     def put_tensor(self, key, value):
#         if len(self.cache) > self.cacacity:
#             self.cache = OrderedDict()
#         self.cache[key] = value

#     def __len__(self):
#         return len(self.pair_data)

#     def __getitem__(self, idx):
#         row = self.pair_data.iloc[idx]
#         no1, no2 = row['no1'], row['no2']

#         src_t = self.get_tensor(no1)
#         trg_t = self.get_tensor(no2)

#         src_conds = torch.FloatTensor(self.prop_data[no1])
#         trg_conds = torch.FloatTensor(self.prop_data[no2])
#         dif_conds = torch.sub(trg_conds, src_conds)

#         mconds = torch.cat([src_conds, dif_conds]).clone().detach()

#         sample = {
#             'src': src_t.to(device=self.device),
#             'trg': trg_t.to(device=self.device),
#             'mconds': mconds.to(self.device)
#         }
#         return sample
