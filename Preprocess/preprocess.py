# from email.contentmanager import raw_data_manager
import os
import numpy as np
import time
import joblib
import argparse
from datetime import timedelta
import pandas as pd
from functools import partial
from multiprocessing import Pool

import moses
from sklearn.preprocessing import RobustScaler, StandardScaler

from Utils.property import to_mol, property_prediction
from Preprocess.augmentation import get_similar_molecular_pairs


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
                        columns=condition.columns,
                        index=condition.index)


def get_properties_from_smiles(dataset, properties,
                               property_prediction, n_jobs=1) -> pd.DataFrame:
    """
    Compute the properties of the source and target in dataset
    Return a DataFrame of given properties

    Arguments:
        dataset: a list of SMILES
        properties: the list of property names
        properties_path: the properties to download
        n_jobs: number of processes
    """
    properties_dict = {}

    pool = Pool(n_jobs)
    mol = list(pool.map(to_mol, dataset))

    for prop in properties:
        results = pool.map(property_prediction[prop], mol)
        properties_dict[prop] = list(results)
    return pd.DataFrame.from_dict(properties_dict)


def prepare_processed_data(src_info, trg_info, conditions, processed_path):
    for r in ['no', 'smiles'] + [c for c in conditions]:
        assert r in src_info.columns.values.tolist()
        assert r in trg_info.columns.values.tolist()

    print('Rename src/trg SMILES/properties')
    src_dict = dict([('smiles', 'src'), ('no', 'src_no')]
                    + [(c, f'src_{c}') for c in conditions])
    trg_dict = dict([('smiles', 'trg'), ('no', 'trg_no')]
                    + [(c, f'trg_{c}') for c in conditions])

    src_info = src_info.rename(columns=src_dict)
    trg_info = trg_info.rename(columns=trg_dict)
    trg_info['trg_en'] = trg_info['trg'].copy()

    print('Concatenate src/trg SMILES/properties')
    results = pd.concat([src_info, trg_info], axis=1)

    print("Save preprocessed data in", processed_path)
    results.to_csv(processed_path, index=False)


# def prepare_processed_data(smiles_path, property_path,
#                            processed_path, pair_path, scaler_path=None):
#     print('Transform properties')

#     prop = pd.read_csv(property_path, index_col='no')
#     prop = scaler_transform(prop, scaler_path)

#     print("Reading file:", pair_path)
#     pair = pd.read_csv(pair_path)

#     src_no = pair['no1'].tolist()
#     trg_no = pair['no2'].tolist()

#     print("Reading file:", smiles_path)
#     dataset = pd.read_csv(smiles_path, index_col='no')

#     print('Getting source smiles/properties...')
#     src_smi = dataset.loc[src_no].reset_index(inplace=False)
#     src_smi = src_smi[['smiles']].rename(columns={'smiles': 'src'})
#     src_prop = prop.loc[src_no].reset_index(inplace=False)
#     src_prop = src_prop.rename(
#         columns={k: f'src_{k}' for k in src_prop.columns})

#     print('Getting target smiles/properties...')
#     trg_smi = dataset.loc[trg_no].reset_index(inplace=False)
#     trg_smi = trg_smi[['smiles']].rename(columns={'smiles': 'trg'})
#     trg_en_smi = trg_smi[['trg']].rename(columns={'trg': 'trg_en'})
#     trg_prop = prop.loc[trg_no].reset_index(inplace=False)
#     trg_prop = trg_prop.rename(
#         columns={k: f'trg_{k}' for k in trg_prop.columns})

#     print('Concatenating source/target smiles/properties...')
#     results = pd.concat([src_smi, trg_en_smi, trg_smi,
#                         src_prop, trg_prop], axis=1)
#     # results = pd.concat([src_smi, trg_smi, src_prop, trg_prop], axis=1)

#     print("Saving file:", processed_path)
#     results.to_csv(processed_path, index=False)
#     print(results.head())


def preprocess(args, data_type):
    print("Create folders...")

    raw_folder = os.path.join(args.data_path, 'raw', data_type)
    aug_folder = os.path.join(args.data_path, 'aug', data_type)
    processed_folder = os.path.join(
        args.data_path, 'aug', f'data_sim{args.similarity:.2f}')

    for f in (raw_folder, aug_folder, processed_folder):
        os.makedirs(f, exist_ok=True)

    smiles_path = os.path.join(raw_folder, 'smiles_serial.csv')
    property_path = os.path.join(raw_folder, 'prop_serial.csv')
    pair_path = os.path.join(
        aug_folder, f'pair_serial_{args.similarity:.2f}.csv')
    processed_path = os.path.join(processed_folder, f'{data_type}1.csv')

    if not os.path.exists(smiles_path):
        print("Create smiles path:", smiles_path)

        if data_type == 'train':
            dataset = moses.get_dataset('train')
        elif data_type == 'validation':
            dataset = moses.get_dataset('test')
        elif data_type == 'test':
            dataset = moses.get_dataset('test_scaffolds')

        dataset = pd.DataFrame({'smiles': dataset,
                                'no': [i+1 for i in range(len(dataset))]})

        # filter out those longer than max string length
        dataset = dataset.loc[(dataset['smiles'].str.len()
                               + len(args.conditions) <= args.max_strlen)]
        dataset.to_csv(smiles_path, index=False)

    if not os.path.exists(property_path):
        print("Prepare property path:", pair_path)
        df_properties = get_properties_from_smiles(
            dataset, args.conditions, args.n_jobs)
        df_properties['no'] = [i+1 for i in range(len(df_properties))]
        df_properties.to_csv(property_path, index=False)

    if not os.path.exists(pair_path):
        print("Prepare pair path:", pair_path)
        get_similar_molecular_pairs(smiles_path=smiles_path,
                                    pair_path=pair_path,
                                    similarity=args.similarity,
                                    n_workers=args.n_jobs)

    if not os.path.exists(processed_path):
        print("Prepare processed path:", processed_path)

        pair_serial = pd.read_csv(pair_path)
        src_no, trg_no = pair_serial.loc[:, 'no1'], pair_serial.loc[:, 'no2']

        smiles_serial = pd.read_csv(smiles_path, index_col='no')
        prop_serial = pd.read_csv(property_path, index_col='no')
        prop_serial = scaler_transform(prop_serial, args.scaler_path)

        src_info = pd.concat([smiles_serial.loc[src_no],
                             prop_serial.loc[src_no]], axis=1).reset_index()
        trg_info = pd.concat([smiles_serial.loc[trg_no],
                             prop_serial.loc[trg_no]], axis=1).reset_index()

        prepare_processed_data(
            src_info, trg_info, args.conditions, processed_path)
