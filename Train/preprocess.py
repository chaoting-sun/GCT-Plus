import os
import time
import joblib
import argparse
from datetime import timedelta
import pandas as pd
from functools import partial
from multiprocessing import Pool

import moses
from sklearn.preprocessing import RobustScaler, StandardScaler

from Utils.property import tanimoto_similarity
from Utils.dataset import get_condition


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


def data_preparation(dataset, conditions, condition_path, serial_path, 
                     save_path, scaler_path=None, n_samples=None):

    prop = pd.read_csv(condition_path)
    prop = pd.concat([prop[['no']], scaler_transform(prop[conditions], scaler_path)], axis=1)
    pair = pd.read_csv(serial_path)

    src_no = pair['no1'].tolist()[:n_samples]
    trg_no = pair['no2'].tolist()[:n_samples]

    if n_samples is not None:
        src_no = src_no[:n_samples]
        trg_no = trg_no[:n_samples]

    src_smi = dataset.set_index('no').loc[src_no].reset_index(inplace=False)
    
    src_smi = src_smi[['smiles']].rename(columns={'smiles': 'src'})
    src_prop = prop.set_index('no').loc[src_no].reset_index(inplace=False)
    src_prop = src_prop.rename(columns={ k:f'src_{k}' for k in src_prop.columns })

    trg_smi = dataset.set_index('no').loc[trg_no].reset_index(inplace=False)
    trg_smi = trg_smi[['smiles']].rename(columns={'smiles': 'trg'})
    trg_prop = prop.set_index('no').loc[trg_no].reset_index(inplace=False)
    trg_prop = trg_prop.rename(columns={ k:f'trg_{k}' for k in trg_prop.columns })

    results = pd.concat([src_smi, trg_smi, src_prop, trg_prop], axis=1)
    results.to_csv(save_path, index=False)

    print(results.head())
    print(">>> SAVE TO: " + save_path)


def preprocess(args, debug=False):
    n_samples_train = 500
    n_samples_valid = 100
    
    """
    train
    """

    """ OBTAIN TRAINING DATASET """
    train_dataset = moses.get_dataset('train')
    if n_samples_train is not None:
        train_dataset = train_dataset[:n_samples_train]

    train_dataset = pd.DataFrame({'smiles': train_dataset,
                               'no': [i+1 for i in range(len(train_dataset))]})
    train_dataset = train_dataset.loc[(train_dataset['smiles'].str.len() 
                                     + len(args.conditions) < args.max_strlen)]

    """ TRAINING DATASET PATH """
    train_condition_path = os.path.join(args.condition_path, 'train', 'prop_serial.csv')
    train_pair_serial_path = os.path.join(os.path.join(args.serial_path, 'train'), 
                             'pair_serial_{:.2f}.csv'.format(args.similarity))
    train_processed_path = os.path.join(args.processed_path, 'data', 'train.csv')

    if not os.path.exists(train_condition_path):
        print('>>> TRAINING SET - COMPUTE PROPERTIES: SAVE TO ' + train_condition_path)
        df_cond = get_condition(train_dataset, args.conditions, None, args.n_jobs)
        df_cond['no'] = [i+1 for i in range(len(df_cond))]
        df_cond.to_csv(train_condition_path, index=False)
    else:
        print('>>> TRAINING SET - COMPUTE PROPERTIES: FILE EXISTS ' + train_condition_path)

    if not os.path.exists(train_pair_serial_path):
        print('>>> TRAINING SET - DATA AUGMENTATION: SAVE TO ' + train_pair_serial_path)
        data_augmentation(train_dataset, train_pair_serial_path, args.similarity, args.n_jobs)
    else:
        print('>>> TRAINING SET - DATA AUGMENTATION: FILE EXISTS ' + train_pair_serial_path)

    if not os.path.exists(train_processed_path):
        print('>>> TRAINING SET - DATA PREPARATION: SAVE TO ' + train_processed_path)
        data_preparation(dataset=train_dataset,
                        conditions=args.conditions,
                        condition_path=train_condition_path,
                        serial_path=train_pair_serial_path, 
                        save_path=train_processed_path,
                        scaler_path=args.scaler_path,
                        n_samples=n_samples_train)
    else:
        print('>>> TRAINING SET - DATA AUGMENTATION: FILE EXISTS ' + train_processed_path)

    """
    validation
    """

    """ OBTAIN VALIDATION DATASET """
    valid_dataset = moses.get_dataset('test')[:n_samples_valid]
    if n_samples_valid is not None:
        valid_dataset = valid_dataset[:n_samples_valid]

    valid_dataset = pd.DataFrame({'smiles': valid_dataset,
                               'no': [i+1 for i in range(len(valid_dataset))]})
    valid_dataset = valid_dataset.loc[(valid_dataset['smiles'].str.len() + len(args.conditions) < args.max_strlen)]

    """ VALIDATION DATASET PATH """
    valid_condition_path = os.path.join(args.condition_path, 'validation', 'prop_serial.csv')
    valid_pair_serial_path = os.path.join(os.path.join(args.serial_path, 'validation'), 
                             'pair_serial_{:.2f}.csv'.format(args.similarity))
    valid_processed_path = os.path.join(args.processed_path, 'data', 'validation.csv')

    if not os.path.exists(valid_condition_path):
        print('>>> VALIDATOIN SET - COMPUTE PROPERTIES: SAVE TO ' + valid_condition_path)
        df_cond = get_condition(valid_dataset, args.conditions, None, args.n_jobs)
        df_cond['no'] = [i+1 for i in range(len(df_cond))]
        df_cond.to_csv(valid_condition_path, index=False)
    else:
        print('>>> VALIDATION SET - COMPUTE PROPERTIES: FILE EXISTS ' + valid_condition_path)
    

    if not os.path.exists(valid_pair_serial_path):
        print('>>> VALIDATOIN SET - DATA AUGMENTATION: SAVE TO ' + valid_pair_serial_path)
        data_augmentation(valid_dataset, valid_pair_serial_path, args.similarity, args.n_jobs)
    else:
        print('>>> VALIDATION SET - DATA AUGMENTATION: FILE EXISTS ' + valid_pair_serial_path)

    if not os.path.exists(valid_processed_path):
        print('>>> VALIDATION SET - DATA PREPARATION: SAVE TO ' + valid_processed_path)
        data_preparation(dataset=valid_dataset,
                        conditions=args.conditions,
                        condition_path=valid_condition_path,
                        serial_path=valid_pair_serial_path, 
                        save_path=valid_processed_path,
                        scaler_path=args.scaler_path,
                        n_samples=n_samples_valid)
    else:
        print('>>> VALIDATION SET - DATA AUGMENTATION: FILE EXISTS ' + valid_processed_path)