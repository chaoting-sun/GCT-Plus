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

# other packages
from .property import to_mol, property_prediction, tanimoto_similarity
from .field import get_fields, save_fields


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



def get_dataset(data_path, fields, train=None, validation=None, test=None):
    # fields = get_fields(conditions, field_path)
    train_data, valid_data = data.TabularDataset.splits(path=data_path,
                                                        train=train,
                                                        validation=validation,
                                                        test=test,
                                                        format='csv', 
                                                        fields=fields, 
                                                        skip_header=True)
    return train_data, valid_data


def get_iterator(dataset, data_type, batch_size, device):
    if data_type == 'train':
        train, shuffle = True, True
    else:
        train, shuffle = False, False

    return data.BucketIterator(dataset=dataset,
                               batch_size=batch_size,
                               sort_key=lambda x: (len(x.src), len(x.trg)),
                               device=device,
                               train=train,
                               repeat=False,
                               shuffle=shuffle)


def to_tokenlist(dataset, data_path):
    toklenList = []
    for i in range(len(dataset)):
        toklenList.append(len(vars(dataset[i])['src']))
    df_toklenList = pd.DataFrame(toklenList, columns=["toklen"])
    df_toklenList.to_csv(os.path.join(data_path, "toklen_list.csv"), index=False)


class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg, src_cond=None, dif_cond=None, trg_cond=None):
        self.src = src
        self.trg = trg[:, :-1]  # the input of the model
        self.trg_y = trg[:, 1:] # the expected output
    
        if src_cond is not None:
            self.econds = src_cond
            self.mconds = torch.cat([src_cond, dif_cond], dim=1)
            self.dconds = trg_cond
        

def rebatch(batch, conditions, device):
    "Fix order in torchtext to match ours"
    src = batch.src.transpose(0, 1).to(device)
    trg = batch.trg.transpose(0, 1).to(device)
    # src, trg = batch.src, batch.trg
    if len(conditions) > 0:
        src_conds, trg_conds = [], []
        for c in conditions:
            src_conds.append(getattr(batch, f"src_{c}").view(-1, 1))
            trg_conds.append(getattr(batch, f"trg_{c}").view(-1, 1))
        src_cond_t = torch.cat(src_conds, dim=1).to(device)
        trg_cond_t = torch.cat(trg_conds, dim=1).to(device)
        dif_cond_t = torch.sub(trg_cond_t, src_cond_t).to(device)
    else:
        src_cond_t = trg_cond_t = dif_cond_t = None    
    return Batch(src, trg, src_cond_t, trg_cond_t, dif_cond_t)


def to_dataloader(data_iter, conditions, device):
    return (rebatch(batch, conditions, device) for batch in data_iter)