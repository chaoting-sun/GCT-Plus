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
from sklearn.preprocessing import RobustScaler, StandardScaler

# other packages
from .property import to_mol, property_prediction
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
        # 補
        pass
    dataset = pd.DataFrame.from_dict({
        'no': [i+1 for i in range(len(dataset))],
        'molecule': dataset,
    })
    return dataset


def get_condition(dataset, condition_list,
                  condition_path=None, n_jobs=1) -> pd.DataFrame:
    """
    Compute the properties of the source and target in dataset
    Return a DataFrame of given properties

    Arguments:
        dataset: a DataFrame of source and target strings
        condition_list: the list of property names
        condition_path: the conditions to download
        n_jobs: number of processes
    """
    if condition_path is not None:
        try:
            condition_library = pd.read_csv(condition_path)
        except:
            print("File not Found:", condition_path)
            exit(1)
        # 補
        return condition_library

    condition_dict = {}

    pool = Pool(n_jobs)
    mol = list(pool.map(to_mol, dataset['molecule']))
    
    for prop in condition_list:
        results = pool.map(property_prediction[prop], mol)
        condition_dict[prop] = list(results)
    return pd.DataFrame.from_dict(condition_dict)


def data_augmentation(dataset, similarity):
    assert 0 <= similarity and similarity <= 1
    # 補
    return dataset


def get_scaler(condition, scaler_path=None):
    if scaler_path is not None:
        try:
            scaler = joblib.load(scaler_path)
            print("- load scaler from", scaler_path)
        except:
            exit(f"error: {scaler_path} file not found")
    else:
        if os.path.exists(scaler_path):
            exit(f"Scaler already existed: {scaler_path}")
        scaler = RobustScaler(quantile_range=(0.1, 0.9))
        scaler.fit(condition.copy(), len(condition.columns))
        joblib.dump(scaler, open(scaler_path), 'wb')

    return scaler


def scaler_transform(condition, scaler):
    """ 
    scaler transformation
    return a DataFrame of rescaled properties
    
    Parameters:
        condition: DataFrame, the properties
        scaler: a property transformer
    """
    return pd.DataFrame(scaler.transform(condition), 
                        columns=condition.columns)


def get_raw_dataset(data_name, data_type, scaler_path, condition_list,
                    condition_path, max_strlen, n_jobs):
    dataset = get_benchmarking_dataset(data_name, data_type)
    dataset = dataset.loc[(dataset['molecule'].str.len() + len(condition_list) < max_strlen)]
    condition = get_condition(dataset, condition_list, condition_path, n_jobs)

    scaler = get_scaler(condition[condition_list], scaler_path)
    condition = scaler_transform(condition, scaler)

    src = dataset.rename(columns={'no':'src_no', 'molecule': 'src'})
    trg = dataset.rename(columns={'no':'trg_no', 'molecule': 'trg'})
    src_c = condition.rename(columns={c: f'src_{c}' for c in condition.columns})
    trg_c = condition.rename(columns={c: f'trg_{c}' for c in condition.columns})
    
    return pd.concat([src[['src_no']], trg[['trg_no']],
                      src[['src']], trg[['trg']], src_c, trg_c], axis=1)


def get_dataset(data_path, conditions, field_path, load_field=False,
                train=None, validation=None, test=None):
    fields = get_fields(conditions, field_path)
    train_data, valid_data = data.TabularDataset.splits(path=data_path,
                                                        train=train,
                                                        validation=validation,
                                                        test=test,
                                                        format='csv', 
                                                        fields=fields, 
                                                        skip_header=True)
    field_dict = {p: f for p, f in fields}
    if load_field is False:
        field_dict['src'].build_vocab(train_data)
        field_dict['trg'].build_vocab(valid_data)
        save_fields(field_dict['src'], field_dict['trg'], field_path)

    return (train_data, valid_data), (field_dict['src'], field_dict['trg'])


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


def rebatch(batch, cond_list):
    "Fix order in torchtext to match ours"
    src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
    if len(cond_list) > 0:
        src_conds, trg_conds = [], []
        for c in cond_list:
            src_conds.append(getattr(batch, f"src_{c}").view(-1, 1))
            trg_conds.append(getattr(batch, f"trg_{c}").view(-1, 1))
        src_cond_t = torch.cat(src_conds, dim=1)
        trg_cond_t = torch.cat(trg_conds, dim=1)
        dif_cond_t = torch.sub(trg_cond_t, src_cond_t)
    else:
        src_cond_t = None
        trg_cond_t = None
        dif_cond_t = None
    return Batch(src, trg, src_cond_t, trg_cond_t, dif_cond_t)


def to_dataloader(data_iter, condition_list):
    return (rebatch(batch, condition_list) for batch in data_iter)