# general modules
import os
import joblib
import pandas as pd
from multiprocessing import Pool

# ml modules
import moses
import torch
from torchtext import data
# from torchtext.legacy import data
from torch.utils.data import Dataset

# other packages
from .property import to_mol, property_prediction


def get_benchmarking_dataset(data_name, data_type='train') -> pd.DataFrame:
    """
    Get the benchmarking dataset represented as strings
    Returns a DataFrame with columns ['src_no', 'trg_no', 'src', 'trg']

    Arguments:
        data_name: 'moses' or 'guacamol'
        data_type: 'train' or 'test'
    """
    if data_name == 'moses':
        dataset = moses.get_dataset(data_type)
    elif data_name == 'guacamol':
        pass
    dataset = pd.DataFrame.from_dict({
        'no': [i+1 for i in range(len(dataset))],
        'molecule': dataset,
    })
    return dataset


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
    "Object for holding a batch of data with mask during training."

    def __init__(self, src, trg_en, trg, pad_idx, device,
                 src_cond=None, dif_cond=None, trg_cond=None):
        self.src = src.to(device)
        self.trg_en = trg_en.to(device)    # the encoder input
        # the expected output of the decoder
        self.trg_y = trg[:, 1:].to(device)
        self.trg = trg[:, :-1].to(device)  # the input of the decoder

        # torch.set_printoptions(threshold=10_000)

        # print('src:', self.src.size(), self.src[0])
        # print('trg_y:', self.trg_y.size(), self.trg_y[0])
        # print('trg:', self.trg.size(), self.trg_de[0])
        # print('trg_en:', self.trg_en.size(), self.trg_en[0])

        if src_cond is not None:
            self.econds = src_cond.to(device)
            self.mconds = torch.cat([src_cond, dif_cond], dim=1).to(device)
            self.dconds = trg_cond.to(device)


def rebatch(batch, conditions, pad_idx, device):
    src = batch.src.transpose(0, 1)
    trg_en = batch.trg_en.transpose(0, 1)
    trg = batch.trg.transpose(0, 1)

    src_len, trg_len = src.size(1), trg_en.size(1)
    pad = torch.ones((src.size(0), abs(src_len - trg_len)),
                     dtype=torch.long) * pad_idx

    if src_len > trg_len:
        trg_en = torch.cat([trg_en, pad], dim=1)
    elif src_len < trg_len:
        src = torch.cat([src, pad], dim=1)

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


def to_dataloader(data_iter, conditions, pad_idx, device):
    return (rebatch(batch, conditions, pad_idx, device) for batch in data_iter)


class mlpDataset(Dataset):
    def __init__(self, conditions, tensor_folder, pair_path,
                 prop_path, device, transform=None):
        self.conditions = conditions
        self.tensor_folder = tensor_folder
        self.pair_data = pd.read_csv(pair_path)
        self.prop_data = pd.read_csv(prop_path)
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.pair_data)

    def __getitem__(self, idx):
        row = self.pair_data.iloc[idx]
        src_t = torch.load(os.path.join(
            self.tensor_folder, f'{row["no1"]}.pt'))
        trg_t = torch.load(os.path.join(
            self.tensor_folder, f'{row["no2"]}.pt'))

        src_conds = self.prop_data[self.conditions].loc[self.prop_data['no']
                                                        == row["no1"]].to_numpy()
        trg_conds = self.prop_data[self.conditions].loc[self.prop_data['no']
                                                        == row["no2"]].to_numpy()

        if self.transform is not None:
            src_conds = self.transform(src_conds.reshape(1, len(self.conditions)))
            trg_conds = self.transform(trg_conds.reshape(1, len(self.conditions)))

        # print('src_conds:', src_conds.size(), src_conds)
        # print('trg_conds:', trg_conds.size(), trg_conds)

        src_conds = torch.from_numpy(src_conds).view(-1)
        trg_conds = torch.from_numpy(trg_conds).view(-1)
        dif_conds = torch.sub(src_conds, trg_conds)
        mconds = torch.cat([src_conds, dif_conds]).clone().detach()

        sample = {
            'src': src_t.to(device=self.device),
            'trg': trg_t.to(device=self.device),
            'mconds': mconds.to(self.device, dtype=torch.float32)
        }
        return sample
