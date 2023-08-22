import os
import moses
import argparse
import dill as pkl
import pandas as pd
from collections import OrderedDict
from pathos.multiprocessing import ProcessingPool as Pool

from Configuration.config import preprocess_opts
from Utils import set_seed, get_logger, get_scaler, \
    get_dataset, smiles_field, get_property_fn, \
    get_mol, murcko_scaffold, mols_to_props


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


def compute_descriptors(smiles, property_list, n_jobs=1,
                        compute_scaffold=True):
    property_fn = get_property_fn(property_list)
    with Pool(n_jobs) as pool:
        mol = pool.map(get_mol, smiles)
    descriptors = mols_to_props(mol, property_fn, n_jobs)
    
    if compute_scaffold:
        with Pool(n_jobs) as pool:
            scaffold = pool.map(murcko_scaffold, smiles)
        descriptors.insert(0, 'scaffold', scaffold)
    descriptors.insert(0, 'smiles', smiles)
    return descriptors


if __name__ == "__main__":
    set_seed(0)
    
    parser = argparse.ArgumentParser()
    preprocess_opts(parser)
    args = parser.parse_args()

    os.makedirs(args.save_folder, exist_ok=True)

    # define save path

    logger = get_logger()
    LOG = logger(name='preprocess', log_path=os.path.join(args.save_folder, 'record.log'))

    raw_folder = os.path.join(args.save_folder, 'raw')
    util_folder = os.path.join(args.save_folder, 'utils')
    prepared_folder = os.path.join(args.save_folder, 'prepared')

    os.makedirs(raw_folder, exist_ok=True)
    os.makedirs(util_folder, exist_ok=True)
    os.makedirs(prepared_folder, exist_ok=True)

    # save raw data
    
    LOG.info('Get moses dataset: train, test, test_scaffolds')
    
    train = moses.get_dataset('train')
    test = moses.get_dataset('test')
    testsca = moses.get_dataset('test_scaffolds')

    if args.debug:
        train = train[:100]
        test = test[:100]
        testsca = testsca[:100]
    
    LOG.info('Compute descriptors')
    
    LOG.info(f'# train / test / testsca: {len(train)} / {len(test)} / {len(testsca)}')
        
    train_des = compute_descriptors(train, args.property_list, n_jobs=args.n_jobs)
    test_des = compute_descriptors(test, args.property_list, n_jobs=args.n_jobs)
    testsca_des = compute_descriptors(testsca, args.property_list, n_jobs=args.n_jobs)

    train_des.to_csv(os.path.join(raw_folder, 'train.csv'))
    test_des.to_csv(os.path.join(raw_folder, 'test.csv'))
    testsca_des.to_csv(os.path.join(raw_folder, 'test_scaffolds.csv'))
    
    # build vocabulary
    
    LOG.info('Build vocabulary')
    
    if args.build_vocab:
        # Create SRC and TRG
        
        SRC, TRG = smiles_field(add_sep=False)
        
        processed_data = [SRC.preprocess(smile) for smile in train_des['smiles']]
        SRC.build_vocab(processed_data)

        processed_data = [TRG.preprocess(smile) for smile in train_des['smiles']]
        TRG.build_vocab(processed_data)

        LOG.info('Vocabulary of SRC: %s', SRC.vocab.stoi)
        LOG.info('Vocabulary of TRG: %s', TRG.vocab.stoi)

        # Create SRC and TRG with separator
        
        SRC_sep, TRG_sep = smiles_field(add_sep=True)

        processed_data = [SRC_sep.preprocess(smile) for smile in train_des['smiles']]
        SRC_sep.build_vocab(processed_data, specials=['<sep>'], min_freq=0)

        processed_data = [TRG_sep.preprocess(smile) for smile in train_des['smiles']]
        TRG_sep.build_vocab(processed_data, specials=['<sep>'], min_freq=0)

        LOG.info('Vocabulary of SRC_sep: %s', SRC_sep.vocab.stoi)
        LOG.info('Vocabulary of TRG_sep: %s', TRG_sep.vocab.stoi)

    # get prepared data

    for data_type in ('train', 'test'):
        LOG.info(f'save prepared data: {data_type}')
        
        raw_path = os.path.join(raw_folder, f'{data_type}.csv')
        prepared_path = os.path.join(prepared_folder, f'{data_type}.csv')
        prepared_scaffold_path = os.path.join(prepared_folder, f'{data_type}_sca.csv')

        raw_data = pd.read_csv(raw_path, index_col=[0])

        if data_type == 'train':
            scaler = get_scaler(util_folder, raw_data[args.scaled_properties], rebuild=True)

        raw_data[args.scaled_properties] = scaler.transform(raw_data[args.scaled_properties])

        prepared_data = OrderedDict()
        
        prepared_data['src'] = raw_data['smiles']
        prepared_data['src_scaffold'] = raw_data['scaffold']
        for p in args.scaled_properties:
            prepared_data[f'src_{p}'] = raw_data[p]
        prepared_data['trg'] = raw_data['smiles']
        prepared_data['trg_scaffold'] = raw_data['scaffold']
        for p in args.scaled_properties:
            prepared_data[f'trg_{p}'] = raw_data[p]
            
        prepared_data = pd.DataFrame(prepared_data)
        prepared_data.to_csv(prepared_scaffold_path, index=False)

        del prepared_data['src_scaffold']
        del prepared_data['trg_scaffold']
        
        prepared_data.to_csv(prepared_path, index=False)

    LOG.info('Finished preprocessing')