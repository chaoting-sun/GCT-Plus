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


def data_preparation(dataset, conditions, condition_path, 
                     serial_path, save_path, scaler_path=None, n_samples=None):
    print('>>> OBTAIN CONDITIONS')
    prop = pd.read_csv(condition_path)
    prop = pd.concat([prop[['no']], scaler_transform(prop[conditions], scaler_path)], axis=1)
    print(prop.describe())

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
    src_prop = prop.set_index('no').loc[src_no].reset_index(inplace=False)
    src_prop = src_prop.rename(columns={ k:f'src_{k}' for k in src_prop.columns })
    print(src_smi.describe())
    print(src_prop.describe())

    print('>>> GET TARGET SMILES/CONDITIONS')
    trg_smi = dataset.set_index('no').loc[trg_no].reset_index(inplace=False)
    trg_smi = trg_smi[['smiles']].rename(columns={'smiles': 'trg'})
    trg_en_smi = trg_smi[['trg']].rename(columns={'trg': 'trg_en'})
    trg_prop = prop.set_index('no').loc[trg_no].reset_index(inplace=False)
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
    
    
def allworks(data_name, conditions, similarity, data_path, 
             scaler_path, max_strlen, n_jobs, n_samples):
    condition_path = os.path.join(data_path, f'raw/{data_name}/prop_serial.csv')
    pair_serial_path = os.path.join(data_path, f'aug/{data_name}', 
                             'pair_serial_{:.2f}.csv'.format(similarity))
    processed_path = os.path.join(data_path, f'data/{data_name}.csv')
    
    if data_name == 'train':
        dataset = moses.get_dataset('train')
    elif data_name == 'validation':
        dataset = moses.get_dataset('test')
    elif data_name == 'test':
        dataset = moses.get_dataset('test_scaffolds')
        
    if not os.path.exists(condition_path):
        print('>>> COMPUTE PROPERTIES: SAVE TO ' + condition_path)
        obtain_dataset_conditions(dataset, conditions, condition_path, n_jobs)

    return

    if n_samples is not None:
        dataset = dataset[:n_samples]

    dataset = pd.DataFrame({'smiles': dataset,
                            'no': [i+1 for i in range(len(dataset))]})
    dataset = dataset.loc[(dataset['smiles'].str.len() 
                          + len(conditions) < max_strlen)]
    
    if not os.path.exists(pair_serial_path):
        print('>>> DATA AUGMENTATION: SAVE TO ' + pair_serial_path)
        data_augmentation(dataset, pair_serial_path, similarity, n_jobs)

    if not os.path.exists(processed_path):
        print('>>> DATA PREPARATION: SAVE TO ' + processed_path)
        data_preparation(dataset=dataset,
                         conditions=conditions,
                         condition_path=condition_path,
                         serial_path=pair_serial_path, 
                         save_path=processed_path,
                         scaler_path=scaler_path)


def preprocess(args, debug=False):    
    """
    - raw
        - train
        - validation
    - aug
        - train - pair_serial_{similarity}.csv
        - validation - pair_serial_{similarity}.csv
        - data: train.csv, validation.csv
    """

    # moses, moses_aug_{similarity}

    """
    Dataset path 
    """
    # datatype = 'train'
    # n_samples = 10000

    datatype = 'validation'
    n_samples = 1000
    
    datadict = {
        'train': 'train',
        'validation': 'test',
        'test': 'test_scaffolds'
    }

    for data in ('train', 'validation', 'test'):
        os.makedirs(os.path.join(args.data_path, 'raw', data), exist_ok=True)
        os.makedirs(os.path.join(args.data_path, 'aug', data), exist_ok=True)
    
    os.makedirs(os.path.join(args.data_path, 'aug', 'data'), exist_ok=True)

    # allworks('train', args.conditions, args.similarity, args.data_path,
    #           args.scaler_path, args.max_strlen, args.n_jobs, n_samples_train)
    # allworks('validation', args.conditions, args.similarity, args.data_path, 
    #          args.scaler_path, args.max_strlen, args.n_jobs, n_samples_valid)
    # allworks('test', args.conditions, args.similarity, args.data_path, 
    #          args.scaler_path, args.max_strlen, args.n_jobs, n_samples_test)

    def get_path(data_path, data_name, similarity):
        cond_path = os.path.join(data_path, f'raw/{data_name}/prop_serial.csv')
        pair_path = os.path.join(data_path, f'aug/{data_name}', 
                                'pair_serial_{:.2f}.csv'.format(similarity))
        proc_path = os.path.join(data_path, 'aug/data_sim{:.2f}/{}.csv'.format(similarity, data_name))
        return cond_path, pair_path, proc_path

    print('>>> CREATE PATHS')
    c_path, pair_path, proc_path = get_path(args.data_path, datatype, args.similarity)

    print('>>> OBTAIN DATAPATH')
    dataset = moses.get_dataset(datadict[datatype])
    dataset = pd.DataFrame({'smiles': dataset,
                            'no': [i+1 for i in range(len(dataset))]})
    dataset = dataset.loc[(dataset['smiles'].str.len() 
                          + len(args.conditions) < args.max_strlen)]

    if not os.path.exists(pair_path):
        print('>>> DATA AUGMENTATION: SAVE TO ' + pair_path)
        data_augmentation(dataset, pair_path, args.similarity, args.n_jobs)

    print('>>> PREPARE DATA')
    data_preparation(dataset=dataset,
                     conditions=args.conditions,
                     condition_path=c_path,
                     serial_path=pair_path, 
                     save_path=proc_path,
                     scaler_path=args.scaler_path,
                     n_samples=n_samples)
