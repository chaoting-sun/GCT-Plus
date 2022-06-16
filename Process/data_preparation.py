import os
import argparse
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
import dill as pickle
from multiprocessing import Pool

import moses
import torch
from torchtext import data
# from torchtext.legacy import data

from Tokenize import moltokenize
import Process.batch as bt
from utils import compute_property as cp

SEED = 42
SPLIT_RATIO = 0.8


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
    
    if lang_format == 'SMILES':
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
        print(f'- invalid src language: {lang_format} supported languages: {lang_supported}')

    print("- loading molecule tokenizers...")

    t_src = moltokenize()
    t_trg = moltokenize()

    SRC = data.Field(tokenize=t_src.tokenizer, batch_first=True)
    TRG = data.Field(tokenize=t_trg.tokenizer, batch_first=True, init_token='<sos>', eos_token='<eos>')

    print(pickle.load(open(os.path.join('..', 'molGCT', 'saved_model', 'SRC.pkl'), 'rb')))
    print('>', os.path.exists(os.path.join('..', 'molGCT', 'saved_model', 'SRC.pkl')))
    if weights_path is not None:
        try:
            print("- loading presaved fields...") 
            SRC = pickle.load(open(os.path.join('..', 'molGCT', 'saved_model', 'SRC.pkl'), 'rb'))
            TRG = pickle.load(open(os.path.join('..', 'molGCT', 'saved_model', 'TRG.pkl'), 'rb'))
            # SRC = pickle.load(open(os.path.join(weights_path, 'SRC.pkl'), 'rb'))
            # TRG = pickle.load(open(os.path.join(weights_path, 'TRG.pkl'), 'rb'))
        except:
            print("- error opening SRC.pkl and TRG.pkl field files, please ensure they are in " + weights_path + "/")
            quit()
    
    return (SRC, TRG)


def _extend_fields(data_fields, cond_list):
    if len(cond_list) > 0:
        for c in cond_list:
            c_field = data.Field(use_vocab=False, sequential=False, 
                                 batch_first=True, dtype=torch.float)
            data_fields.append((c, c_field))
    return data_fields


def _get_iterator(dataset, batch_size, device, data_type, debug=False):
    if data_type == 'train':
        shuffle = True if debug == False else False
        return data.Iterator(dataset, batch_size=batch_size, 
                             sort_key=lambda x: (len(x.src),len(x.trg)), 
                             device=device, train=True, repeat=False, shuffle=shuffle)
    elif data_type in ('test', 'attn_test'):
        return data.Iterator(dataset, batch_size=batch_size, 
                             sort_key=lambda x: (len(x.src),len(x.trg)), 
                             device=device, train=False, repeat=False, shuffle=False)


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


def _transform_conditions(df_conds, scaler_path, 
                          property_order, new_scaler=False):
    if not new_scaler:
        try:
            # scaler order: [[logP, tPSA, QED]]
            scaler = joblib.load(os.path.join(scaler_path, 'scaler.pkl')) 
            print("- load scaler from", os.path.join(scaler_path, 'scaler.pkl'))
        except:
            exit("error:", os.path.join(scaler_path, 'scaler.pkl'), " file not found")
    else:
        print("- create map scaler")
        scaler = RobustScaler(quantile_range = (0.1,0.9))
        scaler.fit(df_conds.copy(), len(df_conds.columns))

    # check if the property orders of the data and the scaler are match
    if list(df_conds.columns) != property_order:
        df_conds = df_conds.reindex(columns=property_order)

    conds = df_conds.to_numpy()
    conds = scaler.transform(conds)
    df_conds_new = pd.DataFrame(conds, columns=property_order)

    print(f"\n - cond original\n{df_conds.describe()}:\n")
    print(f"\n - cond transformed\n{df_conds_new.describe()}:\n")

    if new_scaler:
        print(" - dump map scaler in", scaler_path)
        joblib.dump(scaler, open(os.path.join(scaler_path, 'new_scaler.pkl'), 'wb'))
    return df_conds_new


def create_dataset(opt: argparse.Namespace,
                   tr_te: str,
                   source: List[str],
                   target: List[str],
                   SRC: data.field.Field,
                   TRG: data.field.Field,
                   conds: pd.DataFrame,
                   debug: bool
                   )-> bt.MyIterator:

    raw_data = {'src': [line for line in source], 'trg': [line for line in target]}
    df = pd.DataFrame(raw_data, columns=["src", "trg"])

    if opt.nconds > 0:
        # if tr_te == 'train' and not opt.load_scaler:
        if opt.load_scaler is not True:
            conds = _transform_conditions(conds, opt.scaler_path,
                                          None, new_scaler=True)
        else:
            # use the scaler from molGCT
            property_order = ['logP', 'tPSA', 'QED']
            conds = _transform_conditions(conds, opt.scaler_path, 
                                          property_order, new_scaler=False)
        df = pd.concat([df, conds], axis=1)
    df = _mask_invalid_len_data(df, opt.nconds, 
                                opt.lang_format, opt.max_strlen)

    data_fields = [('src', SRC), ('trg', TRG)]
    data_fields = _extend_fields(data_fields, opt.cond_list)
    data_path = os.path.join(opt.data_path, 'DB_temp.csv')
    df.to_csv(data_path, index=False)

    dataset = data.TabularDataset(data_path, format='csv', 
                                  fields=data_fields, skip_header=True)

    if tr_te == "train":
        toklenList = []
        for i in range(len(dataset)):
            toklenList.append(len(vars(dataset[i])['src']))
        df_toklenList = pd.DataFrame(toklenList, columns=["toklen"])
        df_toklenList.to_csv(os.path.join(opt.data_path, "toklen_list.csv"), index=False)
    
    data_iter = _get_iterator(dataset, opt.batch_size, opt.device, data_type=tr_te, debug=debug)

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

