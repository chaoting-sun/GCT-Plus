import os
import time
import joblib
import argparse
import pandas as pd
from functools import partial
from datetime import timedelta
from multiprocessing import Pool

import moses
import torch
from torchtext import data
from sklearn.preprocessing import RobustScaler, StandardScaler

from Utils.field import smiles_fields, condition_fields
from Utils import allocate_gpu
from Utils.property import tanimoto_similarity
from Utils.dataset import get_condition
from Model import create_source_mask
from Model.build_model import build_model
from fp2sim2 import get_similar_molecular_pairs


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


def scaler_transform(condition, scaler_path=None):
    """ 
    scaler transformation
    return a DataFrame of rescaled properties
    
    Parameters:
        condition: DataFrame, the properties
        scaler: a property transformer
    """
    scaler = get_scaler(condition, scaler_path)
    return pd.DataFrame(scaler.transform(condition),
                        columns=condition.columns)


def data_augmentation(dataset, save_path, similarity, n_jobs):
    assert 0 <= similarity and similarity <= 1
    
    n_data = len(dataset)
    buffer = 100

    if similarity == 1:
        df = pd.DataFrame({ 'no1': dataset['no'].tolist(), 'no2': dataset['no'].tolist()})
        df.to_csv(save_path, index=False)
        return
    
    data_smi, data_no = dataset['smiles'].tolist(), dataset['no'].tolist()
    dataset = list(dataset.to_records(index=False)) # DataFrame to a list of tuples (smiles, no)
    convert_dict = { i: data_no[i] for i in range(len(data_no)) }

    start_time = time.time()
    pairs = []

    for begin, smiles in enumerate(data_smi):
        right_smi = data_smi[begin:n_data]

        with Pool(n_jobs) as p:
            similarities = list(p.map(partial(tanimoto_similarity, smiles), right_smi))
            similar_no = [begin + i for i in range(n_data - begin)
                          if similarities[i] >= similarity]
            for no in similar_no:
                pairs.append([convert_dict[begin], convert_dict[no]])
    
        if (begin + 1) % buffer == 0 or begin == n_data - 1:
            print('>>> PROCESSED {:.2f}% - SIMILAR PAIRS: {}\tSIMILARITY: {}\tBUFFER: {}'.format(
                (begin + 1)/len(data_smi)*100, len(pairs), similarity, buffer
            ))
            df = pd.DataFrame(pairs, columns=["no1", "no2"])
            if not os.path.exists(save_path):
                df.to_csv(save_path, index=False)
            else:
                df.to_csv(save_path, index=False, mode='a', header=False)
            pairs = []

    elipsed_time = time.time() - start_time
    print(">>> ELAPSED TIME:", str(timedelta(seconds=elipsed_time)))


def dataset_preparation(datatype, conditions, max_strlen):
    datadict = {
        'train': 'train',
        'validation': 'test',
        'test': 'test_scaffolds'
    }
    dataset = moses.get_dataset(datadict[datatype])
    dataset = pd.DataFrame({'smiles': dataset,
                            'no': [i+1 for i in range(len(dataset))]})
    dataset = dataset.loc[(dataset['smiles'].str.len()
                          + len(conditions) < max_strlen)]
    return dataset


def condition_preparation(data_path, conditions, scaler_path):
    cond = pd.read_csv(data_path)
    cond = scaler_transform(cond[conditions], scaler_path)
    return pd.concat([cond[['no']], cond], axis=1)



def data_preparation(dataset, conditions, condition_path, 
                     serial_path, save_path, scaler_path=None, n_samples=None):
    print('Transform conditions...')
    cond_values = condition_preparation(condition_path, conditions, scaler_path)


    print('>>> OBTAIN PAIRPATH')
    pair = pd.read_csv(serial_path)
    if n_samples is not None:
        pair = pair[:n_samples]
    print(pair.describe())

    src_no = pair['no1'].tolist()
    trg_no = pair['no2'].tolist()

    print('>>> GET SOURCE SMILES/CONDITIONS')
    src_smi = dataset.set_index('no').loc[src_no].reset_index(inplace=False)
    src_smi = src_smi[['smiles']].rename(columns={'smiles': 'src'})
    src_prop = cond_values.set_index('no').loc[src_no].reset_index(inplace=False)
    src_prop = src_prop.rename(columns={ k:f'src_{k}' for k in src_prop.columns })
    print(src_smi.describe())
    print(src_prop.describe())

    print('>>> GET TARGET SMILES/CONDITIONS')
    trg_smi = dataset.set_index('no').loc[trg_no].reset_index(inplace=False)
    trg_smi = trg_smi[['smiles']].rename(columns={'smiles': 'trg'})
    trg_en_smi = trg_smi[['trg']].rename(columns={'trg': 'trg_en'})
    trg_prop = cond_values.set_index('no').loc[trg_no].reset_index(inplace=False)
    trg_prop = trg_prop.rename(columns={ k:f'trg_{k}' for k in trg_prop.columns })
    print(trg_smi.describe())
    print(trg_prop.describe())

    print('>>> GET RESULTS')
    results = pd.concat([src_smi, trg_en_smi, trg_smi, src_prop, trg_prop], axis=1)
    # results = pd.concat([src_smi, trg_smi, src_prop, trg_prop], axis=1)
    results.to_csv(save_path, index=False)

    print(results.head())
    print(">>> SAVE TO: " + save_path)


def obtain_dataset_conditions(dataset, conditions, save_path, n_jobs):
    df_cond = get_condition(dataset, conditions, n_jobs)
    df_cond['no'] = [i+1 for i in range(len(df_cond))]
    df_cond.to_csv(save_path, index=False)


class Batch:
    def __init__(self, smiles, conds, device):
        self.smiles = smiles.to(device)
        if conds is not None:
            self.conds = conds.to(device)


def rebatch(batch, conditions, pad_idx, max_strlen, device):
    smiles = batch.smiles.transpose(0, 1)
    pad = torch.ones((smiles.size(0), max_strlen-smiles.size(1)),
                     dtype=torch.long) * pad_idx
    smiles = torch.cat([smiles, pad], dim=1)
    if len(conditions) > 0:
        conds = []
        for c in conditions:
            conds.append(getattr(batch, f"{c}").view(-1, 1))
        cond_t = torch.cat(conds, dim=1)
    else:
        cond_t = None
    return Batch(smiles, cond_t, device)


def to_dataloader(dataiter, conditions, pad_idx, device):
    return (rebatch(batch, conditions, pad_idx, device) for batch in dataiter)


def mlp_preprocess(args, datatype, n_samples=None):
    # ex: mlp_preprocess('validation', n_samples=1000)
    """
    - raw
        - train
        - validation
    - aug
        - train - pair_serial_{similarity}.csv
        - validation - pair_serial_{similarity}.csv
        - data: train.csv, validation.csv
    """

    raw_folder = os.path.join(args.data_path, 'raw', datatype)
    aug_folder = os.path.join(args.data_path, 'aug', datatype)
    os.makedirs(aug_folder, exist_ok=True)

    print('Obtain transformed conditions...')
    df_conds = condition_preparation(args.conditions,
                                     args.data_name,
                                     args.data_path,
                                     args.scaler_path)

    print('Obtain SMILES from dataset...')
    if os.path.exists(os.path.join(raw_folder, 'smiles_serial.csv')):
        df_smiles = pd.read_csv(os.path.join(raw_folder, 'smiles_serial.csv'))
    else:
        df_smiles = dataset_preparation(datatype, args.conditions, args.max_strlen)

    print('Obtain encoder inputs...')
    exist_no = df_smiles['no'].tolist() # the no. that meets the length constraint
    df_conds = df_conds.set_index('no').loc[exist_no].reset_index(inplace=False)
    results = pd.concat([df_smiles['smiles'], df_conds], dim=1)
    results.to_csv(os.path.join(raw_folder, 'predata.csv'))

    print('Obtain encoder outputs...')
    device = allocate_gpu()
    COND = condition_fields(args.conditions)
    SRC, TRG = smiles_fields(args.field_path)
    fields = [('smiles', SRC)].extend([(f'{args.conditions[i]}', COND[i]) 
                                        for i in range(len(args.conditions))])
    model = build_model(args, len(SRC.vocab), len(TRG.vocab), 
                        args.molgct_path).to(device)

    dataset = data.TabularDataset(path=os.path.join(raw_folder, 'predata.csv'),
                                  format='csv', fields=fields, skip_header=True)
    dataiter = data.BucketIterator(dataset, batch_sizes=128)
    dataloader = to_dataloader(dataiter, args.conditions, 
                               SRC.vocab.stoi['<pad>'], device)

    os.makedirs(os.path.join(raw_folder, 'tensor'), exist_ok=True)
    for i, batch in enumerate(dataloader):
        src_mask = create_source_mask(batch.src, args.conditions)
        x = model.encode(batch.src.to(device), args.conditions, src_mask)[0]
        for b in range(x.size(0)):
            torch.save(x, os.path.join(raw_folder, 'tensor', f'{batch.src_no[b]}.pt'))


    print('Get similar molecular pairs...')
    if not os.path.exists(os.path.join(aug_folder, f'pair_serial_{args.similarity:.2f}.csv')):
        get_similar_molecular_pairs(data_folder=raw_folder,
                                    data_name='smiles_serial',
                                    pair_path=os.path.join(aug_folder, f'pair_serial_{args.similarity:.2f}.csv'),
                                    similarity=args.similarity,
                                    n_workers=args.n_jobs)

    # print('Process training/validation/testing set...')
    # pair = pd.read_csv(os.path.join(aug_folder, f'pair_serial_{args.similarity:.2f}'))

    # src_no = pair['no1'].tolist()
    # trg_no = pair['no2'].tolist()

    # print('Get source SMILES/conditions')
    # src_smi = df_smiles.set_index('no').loc[src_no].reset_index(inplace=False)
    # src_smi = src_smi[['smiles']].rename(columns={'smiles': 'src'})
    # src_prop = df_conds.set_index('no').loc[src_no].reset_index(inplace=False)
    # src_prop = src_prop.rename(columns={ k:f'src_{k}' for k in src_prop.columns })
    # print(src_smi.describe())
    # print(src_prop.describe())

    # print('Get target SMILES/conditions')
    # trg_smi = df_smiles.set_index('no').loc[trg_no].reset_index(inplace=False)
    # trg_smi = trg_smi[['smiles']].rename(columns={'smiles': 'trg'})
    # trg_en_smi = trg_smi[['trg']].rename(columns={'trg': 'trg_en'})
    # trg_prop = df_conds.set_index('no').loc[trg_no].reset_index(inplace=False)
    # trg_prop = trg_prop.rename(columns={ k:f'trg_{k}' for k in trg_prop.columns })
    # print(trg_smi.describe())
    # print(trg_prop.describe())

    # results = pd.concat([src_smi, trg_en_smi, trg_smi, src_prop, trg_prop], axis=1)
    # results.to_csv(os.path.join(aug_folder, f'data_sim{args.similarity:.2f}', datatype), index=False)

    # print("Result saved to: " + os.path.join(aug_folder, f'data_sim{args.similarity:.2f}', datatype))
    # print(results.head())