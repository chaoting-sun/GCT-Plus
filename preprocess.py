import os
import moses
import argparse
# import faulthandler
import dill as pkl
import numpy as np
import pandas as pd
from time import time
from random import sample
from datetime import timedelta
from collections import OrderedDict
from pathos.multiprocessing import ProcessingPool as Pool
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
# from rdkit.Chem import MurckoDecompose
# from rdkit import Chem
    
from FPSim2.io import create_db_file
from FPSim2 import FPSim2Engine

from Utils.seed import set_seed
from Utils.log import get_logger
from Utils.scaler import get_scaler
from Utils.dataset import get_dataset
from Utils.field import smiles_field
# from Utils.properties import predict_properties#, MurckoScaffoldSimilarity as similarity_fcn
from Utils.properties import property_fn
# from Utils.properties import tanimoto_similarity as similarity_fcn

from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from Configuration.config import preprocessing_opts, device_opts
# from Preprocess.data_augmentation import augment_data
# import seaborn as sns
# import matplotlib.pyplot as plt
from Utils.smiles import get_mol


def get_benchmark_datatype(benchmark):
    if benchmark == 'moses':
        return ['train', 'test', 'test_scaffolds']
    elif benchmark == 'chembl_02':
        return ['train', 'validation']
    

def get_benchmark_smiles(benchmark, data_type):
    if benchmark == 'moses':
        return moses.get_dataset(data_type)
    elif benchmark == 'chembl_02':
        raw_data = pd.read_csv(f'/fileserver-gamma/chaoting/ML/dataset/chembl_02/raw/{data_type}/smiles_serial.csv')
        return raw_data['smiles'].tolist()


def get_conditions(smiles, condition_fn, n_jobs=1):
    with Pool(n_jobs) as pool:
        mol = pool.map(get_mol, smiles)
    results = []
    for fn in condition_fn:
        with Pool(n_jobs) as pool:
            res = pool.map(fn, mol)
        results.append(res)
    return results


def save_raw_data(save_folder, data_type, property_list,
                  n_jobs=1, benchmark='moses'):
    smiles = get_benchmark_smiles(benchmark, data_type)
    calculated_properties = predict_properties(smiles, property_list, n_jobs)
    raw_data = pd.DataFrame({ 'smiles': smiles })
    raw_data = pd.concat([raw_data, calculated_properties], axis=1)
    raw_data.to_csv(os.path.join(save_folder, f'{data_type}.csv'))
    return raw_data


def get_vocab(data_folder, file_name, add_sep=False):
    SRC, TRG = smiles_field(add_sep=add_sep)
    
    train = get_dataset(data_folder=data_folder,
                        fields=[('smiles', SRC)],
                        file_name_list=[file_name, None, None])[0]
    if add_sep:
        SRC.build_vocab(train, specials=['<sep>'], min_freq=0)
    else:
        SRC.build_vocab(train)
        
    train = get_dataset(data_folder=data_folder,
                        fields=[('smiles', TRG)],
                        file_name_list=[file_name, None, None])[0]
    if add_sep:
        TRG.build_vocab(train, specials=['<sep>'], min_freq=0)
    else:
        TRG.build_vocab(train)
    pkl.dump(SRC, open(os.path.join(util_folder, 
        f'SRC{"_sep" if add_sep else ""}.pkl'), 'wb'))
    pkl.dump(TRG, open(os.path.join(util_folder,
        f'TRG{"_sep" if add_sep else ""}.pkl'), 'wb'))
    return SRC, TRG


def create_fp_file(smiFilePath, fpFilePath, fp_type='Morgan',
                   radius=2, nBits=1024):
    """ create a fingerprint file for all smiles
    the settinggs (fp_type, radius, nBits) are the same
    as those in MOSES when it compute internal diversity
    """
    create_db_file(smiFilePath, fpFilePath, fp_type,
                   {'radius': radius, 'nBits': nBits})


def search_similar_pairs(save_folder, smiles_file_name, pair_file_name,
                         similarity, n_workers, LOG):
    smiles_file_path = os.path.join(save_folder, f'{smiles_file_name}.smi')
    fp_file_path = os.path.join(save_folder, f'{smiles_file_name}.h5')
    pair_file_path = os.path.join(save_folder, f'{pair_file_name}.csv')
    
    if not os.path.exists(fp_file_path):
        LOG.info(f'Create fingerprint file: {fp_file_path}')
        create_fp_file(smiles_file_path, fp_file_path)

    fpe = FPSim2Engine(fp_file_path)

    dataset = pd.read_csv(smiles_file_path, sep='\t', header=None)
    dataset = list(dataset.to_records(index=False))

    LOG.info(f'# of dataset: {len(dataset)}')

    mol_pairs = []
    buffer = min(2000, len(dataset))

    start_time = time()
    
    if os.path.exists(pair_file_path):
        pair_no = pd.read_csv(pair_file_path)
        last_end_no = pair_no['no1'].iloc[len(pair_no)-1]
    else:
        last_end_no = -1

    sample_ratio = 0.005

    for (smi, no1) in dataset:
        if not isinstance(smi, str):
            continue
        
        if int(no1) <= int(last_end_no):
            continue

        results = fpe.similarity(smi, similarity, n_workers=n_workers)
        mol_pairs.extend([[no1, no2] for no2, _ in results])

        if no1 == 0:
            df = pd.DataFrame(mol_pairs, columns=["no1", "no2"])
            df.to_csv(pair_file_path, index=False)
            mol_pairs = []

        elif (no1+1)%buffer == 0 or no1 == len(dataset)-1:            
            n_samples = int(sample_ratio*len(mol_pairs))
            mol_pairs = sample(mol_pairs, n_samples)

            print(f'No: {no1+1} - Process {buffer} smiles -> {len(mol_pairs)} pairs,'
                  f'time: {str(timedelta(seconds=time() - start_time))}')
            
            df = pd.DataFrame(mol_pairs, columns=["no1", "no2"])

            if no1+1 == buffer:
                df.to_csv(pair_file_path, index=False)
            else:
                df.to_csv(pair_file_path, index=False, mode='a', header=False)
            
            start_time = time()
            mol_pairs = []


# def save_input_data(raw_data, property_list, prepared_folder,
#                     similarity, scaler, n_jobs):
#     if similarity == 1:
#         src = raw_data.loc[:, ['smiles']].rename(columns={ 'smiles': 'src' })
#         trg = raw_data.loc[:, ['smiles']].rename(columns={ 'smiles': 'trg' })
        
#         prop = raw_data.loc[:, property_list]
#         src_prop = prop.rename(columns={ p: f'src_{p}' for p in property_list })
#         trg_prop = prop.rename(columns={ p: f'trg_{p}' for p in property_list })        

#         prepared_data = pd.concat([src, src_prop, trg, trg_prop], axis=1)
#     else:
#         prepared_data = augment_data(raw_data,
#                                      property_list,
#                                      similarity,
#                                      similarity_fcn,
#                                      n_jobs)
                    
#     src_prop_name = [f'src_{p}' for p in property_list]
#     trg_prop_name = [f'trg_{p}' for p in property_list]

#     prepared_data[src_prop_name] = scaler.transform(prepared_data[src_prop_name])
#     prepared_data[trg_prop_name] = scaler.transform(prepared_data[trg_prop_name])

#     prepared_data.to_csv(prepared_folder, index=False)


def get_raw_data(smiles, property_list, n_jobs):
    prop_fn = [property_fn[p] for p in property_list]
    sca_fn = lambda mol: MurckoScaffoldSmiles(mol=mol)
    with Pool(n_jobs) as pool:
        mol = pool.map(get_mol, smiles)

    prop = []
    for fn in prop_fn:
        with Pool(n_jobs) as pool:
            res = pool.map(fn, mol)
        prop.append(res)
    prop = pd.DataFrame(np.array(prop).T, columns=property_list)
    
    with Pool(n_jobs) as pool:
        sca = pool.map(sca_fn, mol)
            
    smi_scaffold = pd.DataFrame({ 'smiles': smiles, 'scaffold': sca })
    raw_data = pd.concat([smi_scaffold, prop], axis=1)
    return raw_data


def get_prepared_data(raw_data, property_list):
    src_dict = OrderedDict([('smiles', 'src')])
    trg_dict = OrderedDict([('smiles', 'trg')])
    src_dict.update([('scaffold', 'src_scaffold')])
    trg_dict.update([('scaffold', 'trg_scaffold')])
    src_dict.update([(p, f'src_{p}') for p in property_list])
    trg_dict.update([(p, f'trg_{p}') for p in property_list])
    src = raw_data.loc[:, src_dict.keys()].rename(columns=src_dict)
    trg = raw_data.loc[:, trg_dict.keys()].rename(columns=trg_dict)
    prepared_data = pd.concat([src, trg], axis=1)
    return prepared_data


from Utils.scaler import get_scaler


if __name__ == "__main__":    
    set_seed(0)

    parser = argparse.ArgumentParser()
    preprocessing_opts(parser)
    device_opts(parser)
    parser.add_argument('-similarity', type=float, default=0.50)
    parser.add_argument('-scaffold_similarity', action='store_true')
    parser.add_argument('-property_list', nargs='+', default=['logP', 'tPSA', 'QED', 'SAS'])

    # parser.add_argument('-raw_properties', nargs='+', default=['logP', 'tPSA', 'QED', 'SAS'])
    # parser.add_argument('-prepared_properties', nargs='+', default=['logP', 'tPSA', 'QED', 'SAS'])
    parser.add_argument('-debug', action='store_true')

    args = parser.parse_args()

    logger = get_logger()

    benchmark_folder = os.path.join(args.data_folder, args.benchmark)

    raw_folder = os.path.join(benchmark_folder, 'raw')
    util_folder = os.path.join(benchmark_folder, 'utils')
    prepared_folder = os.path.join(benchmark_folder, 'prepared')
    
    os.makedirs(raw_folder, exist_ok=True)
    os.makedirs(util_folder, exist_ok=True)
    os.makedirs(prepared_folder, exist_ok=True)

    data_types = get_benchmark_datatype(args.benchmark)

    LOG = logger(name='preprocess', log_path=os.path.join(benchmark_folder, 'preprocess.log'))
    LOG.info(args)
    LOG.info('save raw data...')
    
    for data_type in data_types:
        LOG.info(f'save raw data: {data_type}')

        save_path = os.path.join(raw_folder, 
            f'{data_type}{"_debug" if args.debug else ""}.csv')
        
        smiles = get_benchmark_smiles(args.benchmark, data_type)        
        if not os.path.exists(save_path):
            # get smiles list from the benchmark
            if args.debug:
                smiles = smiles[:10000]
            
            LOG.info('compute raw data properties')
            raw_data = get_raw_data(smiles, args.property_list, args.n_jobs)
            raw_data.to_csv(save_path)

        # build vocabulary
        if data_type == 'train':
            LOG.info('build vocabulary from training set...')

            df_smiles = pd.DataFrame({ 'smiles': smiles })
            df_smiles.to_csv(os.path.join(raw_folder, 'train-tmp.csv'), index=False)
            SRC, TRG = get_vocab(raw_folder, 'train-tmp', add_sep=True)
            os.remove(os.path.join(raw_folder, 'train-tmp.csv'))

            LOG.info('SRC vocabulary: %s', SRC.vocab.stoi)
            LOG.info('TRG vocabulary: %s', TRG.vocab.stoi)

    exit(0)

    # for data_type in data_types:
    #     LOG.info(f'save prepared data: {data_type}')

    #     # get raw data
    #     raw_data = pd.read_csv(os.path.join(raw_folder,
    #         f'{data_type}{"_debug" if args.debug else ""}.csv'))

    #     # get what the prepared dataset you want
    #     raw_data = raw_data[['smiles', 'scaffold']+args.property_list]
        
    #     # build scaler and transform the properties
    #     if data_type == 'train':
    #         scaler = get_scaler(util_folder, raw_data[args.property_list],
    #                             rebuild=True)
    #     raw_data[args.property_list] = scaler.transform(raw_data[args.property_list])

    #     prepared_data = get_prepared_data(raw_data, args.property_list)
    #     prepared_data.to_csv(os.path.join(prepared_folder,
    #                                       f"{data_type}_"
    #                                       f"{'-'.join(args.property_list)}"
    #                                       f"{'_debug' if args.debug else ''}.csv"), index=False)


    def concatenate_str(x):
        if isinstance(x.scaffold, str) and len(x.scaffold) > 0:
            return x['smiles']+'<sep>'+x['scaffold']
        return np.nan

    for data_type in data_types:
        LOG.info(f'save prepared data: {data_type}')

        raw_data = pd.read_csv(os.path.join(raw_folder,
            f'{data_type}{"_debug" if args.debug else ""}.csv'))

        raw_data['smiles'] = raw_data.apply(lambda x: concatenate_str(x), axis=1)
        raw_data = raw_data.dropna(subset=['smiles'])
        raw_data = raw_data[['smiles']+args.property_list]
        
        # build scaler and transform the properties
        if data_type == 'train':
            scaler = get_scaler(util_folder, raw_data[args.property_list],
                                rebuild=True)
        raw_data[args.property_list] = scaler.transform(raw_data[args.property_list])

        prepared_data = OrderedDict()
        prepared_data['src'] = raw_data['smiles']
        for p in args.property_list:
            prepared_data[f'src_{p}'] = raw_data[p]
        prepared_data['trg'] = raw_data['smiles']
        for p in args.property_list:
            prepared_data[f'trg_{p}'] = raw_data[p]
        prepared_data = pd.DataFrame(prepared_data)

        prepared_data.to_csv(os.path.join(prepared_folder,
                                          f"{data_type}_"
                                          f"{'-'.join(args.property_list)}"
                                          f"{'_debug' if args.debug else ''}.csv"), index=False)

    LOG.info('Finished preprocessing')
    
    
    # for data_type in data_types:
    #     LOG.info(f'process dataset: {data_type}...')

    #     raw_path = os.path.join(raw_folder, f'{data_type}.csv')
    #     raw_sample_path = os.path.join(raw_folder, f'{data_type}_debug.csv')
        
    #     if not os.path.exists(raw_path):
    #         LOG.info(f'save raw data: {data_type}')
    #         raw_data = save_raw_data(raw_folder,
    #             data_type, args.all_property_list,
    #             args.n_jobs, args.benchmark)
    #     else:
    #         raw_data = pd.read_csv(raw_path, index_col=[0])

    #     if args.debug:
    #         raw_data = raw_data.loc[:1000]
    #         raw_data.to_csv(raw_sample_path)
        
    #     if data_type == 'train':
    #         # here we use the scaler from molgct
    #         scaler = get_scaler(args.property_list,
    #             util_folder, raw_data[args.property_list],
    #             rebuild=False)

    #     raw_data[args.property_list] = scaler.transform(raw_data[args.property_list])

    #     src_dict = dict({'smiles': 'src'},**{ p: f'src_{p}' for p in args.property_list })
    #     trg_dict = dict({'smiles': 'trg'},**{ p: f'trg_{p}' for p in args.property_list })

    #     if args.similarity < 1:
    #         if args.scaffold_similarity:
    #             LOG.info('Get Murcko scaffold smiles...')
    #             smisca_raw_path = os.path.join(raw_folder,
    #                                            f'{data_type}_sca.smi')
    #             if not os.path.exists(smisca_raw_path):                
    #                 with Pool(args.n_jobs) as pool:
    #                     scaffold_smi = list(pool.map(MurckoScaffoldSmiles,
    #                                         raw_data.loc[:, 'smiles']))
    #                 smi_data = pd.DataFrame({
    #                     'smiles': scaffold_smi,
    #                     'no': [i for i in range(len(scaffold_smi))]
    #                 })
    #                 smi_data.to_csv(smisca_raw_path, sep='\t',
    #                                 header=False, index=False)

    #             smiles_file_name = f'{data_type}_sca'
    #             pair_file_name = f'{data_type}_sca_pair-s{args.similarity:.2f}'

    #         else:
    #             smi_data = raw_data.loc[:, ['smiles']]
    #             smi_data['no'] = [i for i in range(len(smi_data))]
    #             smi_data.to_csv(os.path.join(raw_folder, f'{data_type}.smi'),
    #                             sep='\t', header=False, index=False)

    #             smiles_file_name = f'{data_type}'
    #             pair_file_name = f'{data_type}_pair-s{args.similarity:.2f}'

    #         LOG.info("Search similar pairs...")
            
    #         search_similar_pairs(raw_folder, smiles_file_name,
    #             pair_file_name, args.similarity, args.n_jobs, LOG)
                        
    #         pairs = pd.read_csv(os.path.join(raw_folder, f'{pair_file_name}.csv'))
            
    #         LOG.info(f'Get pairs: # = {len(pairs)}')

    #     src = raw_data[src_dict.keys()].rename(columns=src_dict)
    #     trg = raw_data[trg_dict.keys()].rename(columns=trg_dict)
    #     # get smiles and prop cols and rename them

    #     print(src.head())

    #     if args.similarity < 1:
    #         src = src.iloc[pairs['no1']].reset_index(drop=True)
    #         trg = trg.iloc[pairs['no2']].reset_index(drop=True)

    #     LOG.info('save data...')

    #     prepared_data = pd.concat([src, trg], axis=1)

    #     print(prepared_data.head())

    #     prepared_data.to_csv(os.path.join(prepared_folder,
    #         f'{data_type}-s{args.similarity:.2f}.csv'), index=False)
    #     prepared_data[:1000].to_csv(os.path.join(prepared_folder,
    #         f'{data_type}-s{args.similarity:.2f}_debug.csv'), index=False)


