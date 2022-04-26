import os
import argparse
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import dill as pickle
from multiprocessing import Pool
import torch
from torchtext.legacy import data


from sklearn.model_selection import train_test_split

import utils.file as uf
from .tokenizer import moltokenize
import configuration.config_default as cfgd
import Process.batch as bt

SEED = 42
SPLIT_RATIO = 0.8

import moses
from utils import compute_property as cp

def get_dataset(dataset_name: str, 
                data_type='train')->np.array:
    if dataset_name == 'moses':
        dataset = moses.get_dataset(data_type)
    elif dataset_name == 'guacamol':
        pass
    return dataset


def get_property(dataset: List,
                 lang_format='SMILES',
                 propertylist=['logP', 'QED', 'tPSA', 'SA', 'SC', 'NP'],
                 n_jobs=1
                 )->pd.DataFrame:
    """ Obtain a DataFrame of properties from a list of dataset. """

    pool = Pool(n_jobs)
    

    if lang_format is 'SMILES':
        dataset = list(pool.map(cp.MolFromSmiles, dataset))

    data_dict = {}

    for prop in propertylist:
        data_dict[prop] = list(pool.map(getattr(cp, prop), dataset))
    
    return pd.DataFrame.from_dict(data_dict)


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


def create_fields(lang_format="SMILES",
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

    print(" - loading molecule tokenizers...")

    t_src = moltokenize()
    t_trg = moltokenize()

    SRC = data.Field(tokenize=t_src.tokenizer, batch_first=True)
    TRG = data.Field(tokenize=t_trg.tokenizer, batch_first=True, init_token='<sos>', eos_token='<eos>')

    if weights_path is not None:
        try:
            print("loading presaved fields...")
            SRC = pickle.load(open(f'{weights_path}/SRC.pkl', 'rb'))
            TRG = pickle.load(open(f'{weights_path}/TRG.pkl', 'rb'))
        except:
            print("error opening SRC.pkl and TRG.pkl field files, please ensure they are in " + weights_path + "/")
            quit()
    
    return (SRC, TRG)


def _extend_fields(data_fields, cond_list):
    if len(cond_list) > 0:
        for c in cond_list:
            c_field = data.Field(use_vocab=False, sequential=False, 
                                 batch_first=True, dtype=torch.float)
            data_fields.append((c, c_field))
    return data_fields


def _get_iterator(dataset, batch_size, device, data_type):
    if data_type == 'train':
        return data.Iterator(dataset, batch_size=batch_size, 
                             sort_key=lambda x: (len(x.src),len(x.trg)), 
                             device=device, train=True, repeat=False, shuffle=True)
    elif data_type in ('test', 'attn_test'):
        return data.Iterator(dataset, batch_size=batch_size, 
                             sort_key=lambda x: (len(x.src),len(x.trg)), 
                             device=device, train=False, repeat=False, shuffle=True)


def _mask_invalid_len_data(df, nconds, lang_format, max_strlen):
    if lang_format == 'SMILES':
        mask = (df['src'].str.len() + nconds < max_strlen) \
             & (df['trg'].str.len() + nconds < max_strlen)
    elif lang_format == 'SELFIES':
        mask = (df['src'].str.count('][') + nconds < max_strlen) \
             & (df['trg'].str.count('][') + nconds < max_strlen)
    df = df.loc[mask]
    return df



import joblib
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn_pandas import DataFrameMapper


def _transform_conditions(df_conds, scaler_path, new_scaler=False):
    """
    The following website provides a scaling methods on a DataFrame with several columns of features
    ref: https://stackoverflow.com/questions/35723472/how-to-use-sklearn-fit-transform-with-pandas-and-return-dataframe-instead-of-num
    """
    if not new_scaler:
        try:
            print(" - load map_scaler in", scaler_path)
            map_scaler = joblib.load(os.path.join(scaler_path, 'map_scaler.pkl'))
        except:
            exit("error:", os.path.join(scaler_path, 'map_scaler.pkl'), " file not found")
    else:
        print(" - create map scaler")
        scaler = RobustScaler(quantile_range = (0.1,0.9))
        # scaler = StandardScaler()
        map_scaler = DataFrameMapper([(df_conds.columns, scaler)])
        map_scaler.fit(df_conds.copy(), len(df_conds.columns))

    scaled_features = map_scaler.transform(df_conds.copy())
    df_scaled_features = pd.DataFrame(scaled_features, 
                                      index=df_conds.index, columns=df_conds.columns)
    
    print(f"\n - cond original\n{df_conds.describe()}:\n")
    print(f"\n - cond transformed\n{df_scaled_features.describe()}:\n")

    if new_scaler:
        print(" - dump map scaler in", scaler_path)
        joblib.dump(map_scaler, open(os.path.join(scaler_path, 'map_scaler.pkl'), 'wb'))
    return df_scaled_features


def create_dataset(opt: argparse.Namespace,
                   tr_te: str,
                   source: List[str],
                   target: List[str],
                   SRC: data.field.Field,
                   TRG: data.field.Field,
                   conds: pd.DataFrame,
                   )-> bt.MyIterator:

    raw_data = {'src': [line for line in source], 'trg': [line for line in target]}
    df = pd.DataFrame(raw_data, columns=["src", "trg"])

    if opt.nconds > 0:
        if tr_te == 'train' and not opt.load_scalar:
            conds = _transform_conditions(conds, opt.data_path, new_scaler=True)
        else:
            conds = _transform_conditions(conds, opt.data_path, new_scaler=False)
        df = pd.concat([df, conds], axis=1)
    df = _mask_invalid_len_data(df, opt.nconds, opt.lang_format, opt.max_strlen)

    data_fields = [('src', SRC), ('trg', TRG)]
    data_fields = _extend_fields(data_fields, opt.cond_list)
    data_path = os.path.join(opt.data_path, 'DB_temp.csv')
    df.to_csv(data_path, index=False)

    dataset = data.TabularDataset(data_path, format='csv', fields=data_fields, skip_header=True)

    # if opt.verbose:
    #     print(f' - tokenized {tr_te} sample 0:', vars(dataset[0]))

    if tr_te == "train":
        toklenList = []
        for i in range(len(dataset)):
            toklenList.append(len(vars(dataset[i])['src']))
        df_toklenList = pd.DataFrame(toklenList, columns=["toklen"])
        df_toklenList.to_csv(os.path.join(opt.data_path, "toklen_list.csv"), index=False)
    
    data_iter = _get_iterator(dataset, opt.batch_size, opt.device, data_type=tr_te)

    # print(" - dict-key:", dataset[0].__dict__.keys())
    # print(" - source:", dataset[0].src)

    if tr_te == "train":
        if opt.load_field is False:
            print(" - building vocab from train data...")
            SRC.build_vocab(dataset)
            TRG.build_vocab(dataset)

            field_folder = os.path.join(opt.data_path, opt.field_path)
            os.makedirs(field_folder, exist_ok=True)
            pickle.dump(SRC, open(os.path.join(field_folder, 'SRC.pkl'), 'wb'))
            pickle.dump(TRG, open(os.path.join(field_folder, 'TRG.pkl'), 'wb'))

        opt.src_pad = SRC.vocab.stoi['<pad>']
        opt.trg_pad = TRG.vocab.stoi['<pad>']
        assert opt.src_pad == opt.trg_pad

        opt.train_len = sum(1 for _ in data_iter)

    elif tr_te == "test":
        opt.test_len = sum(1 for _ in data_iter)

    return data_iter

