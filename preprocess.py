import os
import moses
import argparse
import dill as pkl
import pandas as pd
from time import time
from datetime import timedelta
from pathos.multiprocessing import ProcessingPool as Pool
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

from Utils.seed import set_seed
from Utils.log import get_logger
from Utils.scaler import get_scaler
from Utils.dataset import get_dataset
from Utils.field import smiles_field
from Utils.properties import predict_properties, MurckoScaffoldSimilarity as similarity_fcn
# from Utils.properties import tanimoto_similarity

from Configuration.config import preprocessing_opts, device_opts
from Preprocess.data_augmentation import augment_data


def get_moses_data(data_type):
    dataset = moses.get_dataset(data_type)
    return dataset                 


def get_chembl02_data(data_type):
    raw_data = pd.read_csv(f'/fileserver-gamma/chaoting/ML/dataset/chembl_02/raw/{data_type}/smiles_serial.csv')
    return raw_data['smiles'].tolist()


def save_raw_data(save_folder, data_type, property_list,
                  n_jobs=1, benchmark='moses'):
    if benchmark == 'moses':
        raw_data = get_moses_data(data_type)
    
    elif benchmark == 'chembl_02':
        raw_data = get_chembl02_data(data_type)
    
    calculated_properties = predict_properties(raw_data,
                                                property_list,
                                                n_jobs)
    raw_data = pd.DataFrame({ 'smiles': raw_data })
    raw_data = pd.concat([raw_data, calculated_properties], axis=1)
    raw_data.to_csv(os.path.join(save_folder, f'{data_type}.csv'))
    return raw_data


def build_vocab(data_folder, util_folder, debug=False):
    SRC, TRG = smiles_field()

    if debug:
        train_name = 'train_sample' if debug else 'train'
    else:
        train_name = 'train'

    train = get_dataset(data_folder=data_folder,
                        fields=[('no', None), ('smiles', SRC)],
                        file_name_list=[train_name, None, None])
    SRC.build_vocab(train[0])

    train = get_dataset(data_folder=data_folder,
                        fields=[('no', None), ('smiles', TRG)],
                        file_name_list=[train_name, None, None])
    TRG.build_vocab(train[0])

    pkl.dump(SRC, open(os.path.join(util_folder, 'SRC.pkl'), 'wb'))
    pkl.dump(TRG, open(os.path.join(util_folder, 'TRG.pkl'), 'wb'))
    return SRC, TRG


from FPSim2.io import create_db_file
from FPSim2 import FPSim2Engine


def create_fp_file(smiFilePath, fpFilePath, fp_type='Morgan',
                   radius=2, nBits=1024):
    """ create a fingerprint file for all smiles
    the settinggs (fp_type, radius, nBits) are the same
    as those in MOSES when it compute internal diversity
    """
    create_db_file(smiFilePath, fpFilePath, fp_type,
                   {'radius': radius, 'nBits': nBits})


def search_similar_pairs(save_folder, smiles_file_name, pair_file_name,
                         similarity_threshold, n_workers, LOG):
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
    buffer = min(1000, len(dataset))

    start_time = time()

    for i, (smi, idx) in enumerate(dataset):
        results = fpe.similarity(smi, similarity_threshold, n_workers=n_workers)
        df_data = pd.DataFrame(results)

        ids = pd.to_numeric(df_data['mol_id']).tolist()
        mol_pairs.extend([[idx, i] for i in ids])

        if (i+1) % buffer == 0 or i == len(dataset)-1:
            elipsed_time = time() - start_time
            print(f'Process {buffer} smiles -> {len(mol_pairs)} pairs,'
                  f'time: {str(timedelta(seconds=elipsed_time))}')
            
            df = pd.DataFrame(mol_pairs, columns=["no1", "no2"])
            df.to_csv(pair_file_path, index=False, mode='a', header=False)
        
            start_time = time()
            mol_pairs = []


def save_input_data(raw_data, property_list, prepared_folder,
                    similarity_threshold, scaler, n_jobs):
    if similarity_threshold == 1:
        src = raw_data.loc[:, ['smiles']].rename(columns={ 'smiles': 'src' })
        trg = raw_data.loc[:, ['smiles']].rename(columns={ 'smiles': 'trg' })
        
        prop = raw_data.loc[:, property_list]
        src_prop = prop.rename(columns={ p: f'src_{p}' for p in property_list })
        trg_prop = prop.rename(columns={ p: f'trg_{p}' for p in property_list })        

        prepared_data = pd.concat([src, src_prop, trg, trg_prop], axis=1)
    else:
        prepared_data = augment_data(raw_data,
                                     property_list,
                                     similarity_threshold,
                                     similarity_fcn,
                                     n_jobs)
                    
    src_prop_name = [f'src_{p}' for p in property_list]
    trg_prop_name = [f'trg_{p}' for p in property_list]

    prepared_data[src_prop_name] = scaler.transform(prepared_data[src_prop_name])
    prepared_data[trg_prop_name] = scaler.transform(prepared_data[trg_prop_name])

    prepared_data.to_csv(prepared_folder, index=False)


if __name__ == "__main__":
    set_seed(0)

    parser = argparse.ArgumentParser()
    preprocessing_opts(parser)
    device_opts(parser)
    parser.add_argument('-similarity_threshold', type=float, default=0.50)
    parser.add_argument('-scaffold_similarity', action='store_true')
    parser.add_argument('-property_list', nargs='+', default=['logP', 'tPSA', 'QED'])
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

    LOG = logger(name='preprocess', log_path=os.path.join(benchmark_folder, 'preprocess.log'))
    LOG.info(args)

    if args.benchmark == 'moses':
        data_type_list = ('train', 'test', 'test_scaffolds')
    elif args.benchmark == 'chembl_02':
        data_type_list = ('train', 'validation')

    for data_type in data_type_list:
        if not os.path.exists(os.path.join(raw_folder, f'{data_type}.csv')):
            LOG.info(f'save raw data: {data_type}')
            raw_data = save_raw_data(raw_folder,
                                     data_type,
                                     args.all_property_list,
                                     args.n_jobs,
                                     args.benchmark
                                     )
        else:
            raw_data = pd.read_csv(os.path.join(raw_folder, f'{data_type}.csv'),
                                   index_col=[0])

        if args.debug:
            raw_data = raw_data.loc[:1000]
            raw_data.to_csv(os.path.join(raw_folder, f'{data_type}_sample.csv'))
        
        if data_type == 'train':
            scaler = get_scaler(args.property_list, util_folder,
                                raw_data.loc[:, args.property_list], rebuild=True)

        raw_data[args.property_list] = scaler.transform(raw_data[args.property_list])
        raw_data[args.property_list] = scaler.transform(raw_data[args.property_list])

        src_dict = dict({'smiles': 'src'},**{ p: f'src_{p}' for p in args.property_list })
        trg_dict = dict({'smiles': 'trg'},**{ p: f'trg_{p}' for p in args.property_list })

        if args.similarity_threshold == 1:
            src = raw_data.rename(columns=src_dict).reset_index(drop=True)
            trg = raw_data.rename(columns=trg_dict).reset_index(drop=True)
            prepared_data = pd.concat([src, trg], axis=1)

            # src = raw_data

            # src = raw_data.loc[:, ['smiles']].rename(columns={ 'smiles': 'src' })
            # trg = raw_data.loc[:, ['smiles']].rename(columns={ 'smiles': 'trg' })
            
            # prop = raw_data.loc[:, args.property_list]
            # src_prop = prop.rename(columns={ p: f'src_{p}' for p in args.property_list })
            # trg_prop = prop.rename(columns={ p: f'trg_{p}' for p in args.property_list })        

            # prepared_data = pd.concat([src, src_prop, trg, trg_prop], axis=1)

        else: # need augmentation

            if args.scaffold_similarity:
                LOG.info('Get Murcko scaffold smiles...')
                with Pool(args.n_jobs) as pool:
                    scaffold_smi = list(pool.map(MurckoScaffoldSmiles, raw_data.loc[:, 'smiles']))
                smi_data = pd.DataFrame({
                    'smiles': scaffold_smi,
                    'no': [i for i in range(len(scaffold_smi))]
                })
                smi_data.to_csv(os.path.join(raw_folder, f'{data_type}_sca.smi'),
                                sep='\t', header=False, index=False)

                smiles_file_name = f'{data_type}_sca'
                pair_file_name = f'{data_type}_sca_pair'

            else:
                smi_data = raw_data.loc[:, ['smiles']]
                smi_data['no'] = [i for i in range(len(smi_data))]
                smi_data.to_csv(os.path.join(raw_folder, f'{data_type}.smi'), 
                                sep='\t', header=False, index=False)

                smiles_file_name = f'{data_type}'
                pair_file_name = f'{data_type}_pair-s{args.similarity_threshold:.2f}'

            if not os.path.exists(os.path.join(raw_folder, f'{pair_file_name}.csv')):
                LOG.info("Search similar pairs...")
                search_similar_pairs(raw_folder, smiles_file_name, pair_file_name,
                                     args.similarity_threshold, args.n_jobs, LOG)
            
            LOG.info('Get pairs...')
            
            pairs = pd.read_csv(os.path.join(raw_folder, f'{pair_file_name}.csv'),
                                names=['no1', 'no2'])

            src = raw_data.iloc[pairs['no1']].rename(columns=src_dict).reset_index(drop=True)
            trg = raw_data.iloc[pairs['no2']].rename(columns=trg_dict).reset_index(drop=True)
            prepared_data = pd.concat([src, trg], axis=1)
            
        LOG.info('save data...')
        print(prepared_data.head())
        
        prepared_data.to_csv(os.path.join(prepared_folder,
            f'{data_type}-s{args.similarity_threshold}.csv'), index=False)


    if not os.path.exists(os.path.join(util_folder, 'SRC.pkl')):
        LOG.info(f'save utils: SRC.pkl & TRG.pkl')
        SRC, TRG = build_vocab(raw_folder, util_folder, args.debug)

        LOG.info('SRC vocabulary: %s', SRC.vocab.stoi)
        LOG.info('TRG vocabulary: %s', TRG.vocab.stoi)
