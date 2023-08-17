import os
import joblib
import argparse
import pandas as pd

from Configuration.config_default import max_strlen
from Configuration.config import hard_constraints_opts

from Utils import allocate_gpu, smiles_field, \
    set_seed, get_logger, get_scaler

from Inference.uc_sampling import uc_sampling
from Inference.p_sampling import p_sampling
from Inference.sca_sampling import sca_sampling
from Inference.psca_sampling import psca_sampling
from Inference.mol_interpolation import mol_interpolation
from Inference.model_selection import model_selection
from Inference.visualize_attention import visualize_attention


def add_args(parser):
    hard_constraints_opts(parser)
    parser.add_argument('-use_scaffold', action='store_true')
    
    subparsers = parser.add_subparsers(help='select tasks')
    
    parent_parser = argparse.ArgumentParser(add_help=False)

    parent_parser.add_argument('-property_list', nargs='+', default=[])

    parent_parser.add_argument('-model_type', type=str, required=True)
    parent_parser.add_argument('-model_name', type=str, required=True)
    parent_parser.add_argument('-model_folder', type=str, required=True)
        
    parent_parser.add_argument('-encode_type', type=str, default='encode')
    parent_parser.add_argument('-decode_type', type=str, default='decode')
    parent_parser.add_argument('-decode_algo', type=str, default='greedy')
    parent_parser.add_argument('-top_k', type=int) # top k selection in multinomial

    parent_parser.add_argument('-benchmark', type=str, default='moses')
    parent_parser.add_argument('-data_folder', type=str, default='./Data/')
    parent_parser.add_argument('-save_folder', type=str, required=True)
    parent_parser.add_argument('-scaffold_folder', type=str, default='./Data/scaffold-condition/')

    parent_parser.add_argument('-n_jobs', type=int, default=8)
    parent_parser.add_argument('-debug', action='store_true')

    # unconditioned sampling
    uc_parser = subparsers.add_parser('uc-sampling', parents=[parent_parser])
    uc_parser.add_argument('-n_samples', type=int, default=30000)
    uc_parser.add_argument('-batch_size', type=int, default=512)
    uc_parser.add_argument('-descriptor', type=str, nargs='+',
                           default=['logP', 'tPSA', 'QED', 'MW', 'SAS', 'NP',
                                    'HAC', 'HBA', 'HBD', 'RBN', 'AIRN', 'ARRN'])
    uc_parser.set_defaults(func=uc_sampling)

    # property conditioned sampling
    pc_parser = subparsers.add_parser('p-sampling', parents=[parent_parser])
    pc_parser.add_argument('-n_samples', type=int, default=10000)
    pc_parser.add_argument('-batch_size', type=int, default=512)
    pc_parser.set_defaults(func=p_sampling)

    # scaffold conditioned sampling
    sca_parser = subparsers.add_parser('sca-sampling', parents=[parent_parser])
    sca_parser.add_argument('-n_scaffolds', type=int, default=100)
    sca_parser.add_argument('-n_samples', type=int, default=10000)
    sca_parser.add_argument('-scaffold_source', type=str, default='train')
    sca_parser.add_argument('-use_molgpt', action='store_true')
    sca_parser.add_argument('-batch_size', type=int, default=512)
    sca_parser.add_argument('-substructure', action='store_true')
    sca_parser.set_defaults(func=sca_sampling)

    # property and scaffold sampling
    psca_parser = subparsers.add_parser('psca-sampling', parents=[parent_parser])
    psca_parser.add_argument('-n_scaffolds', type=int, default=100)
    psca_parser.add_argument('-n_samples', type=int, default=1000)
    psca_parser.add_argument('-scaffold_source', type=str, default='train')
    psca_parser.add_argument('-batch_size', type=int, default=512)
    psca_parser.set_defaults(func=psca_sampling)

    # molecular interpolation
    mi_parser = subparsers.add_parser('mol-interpolation', parents=[parent_parser])
    mi_parser.add_argument('-n_pairs', type=int, default=100)
    mi_parser.add_argument('-n_interpolations', type=int, default=8)
    mi_parser.add_argument('-pair_folder', type=str, required=True)
    mi_parser.add_argument('-pair_source', type=str, default='test_scaffolds')
    mi_parser.set_defaults(func=mol_interpolation)
    
    # model selection        
    ms_parser = subparsers.add_parser('model-selection', parents=[parent_parser])
    ms_parser.add_argument('-n_samples', type=int, default=10000)
    ms_parser.add_argument('-batch_size', type=int, default=512)
    ms_parser.add_argument('-epoch_list', type=int, nargs='+', default=[21,22,23,24,25])
    # ms_parser.set_defaults(func=)    

    # encoder test
    et_parser = subparsers.add_parser('encoder-test', parents=[parent_parser])
    et_parser.add_argument('-encoder_test', action='store_true')
    # et_parser.add_argument('-model_type', type=str, required=True)
    
    # decoder test
    dt_parser = subparsers.add_parser('decoder-test', parents=[parent_parser])
    dt_parser.add_argument('-decoder_test', action='store_true')
    # dt_parser.add_argument('-model_type', type=str, required=True)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    print(args)

    set_seed(0)

    print("Add the logger...")
    logger = get_logger()

    print("Looking for gpu...")
    device = allocate_gpu()
    print("device:", device)
    
    util_path = os.path.join(args.data_folder, 'utils')
    data_path = os.path.join(args.data_folder, 'raw')
    
    args.max_strlen = max_strlen
    
    # get fields    
    
    if args.use_scaffold:
        SRC, TRG = smiles_field(util_path, add_sep=True)
    else:
        SRC, TRG = smiles_field(util_path)

    if len(args.property_list) > 0:
        scaler = joblib.load(os.path.join(util_path, f'scaler_{"-".join(args.property_list)}.pkl'))
    else:
        scaler = None

    toklen_data = pd.read_csv(os.path.join(data_path, 'toklen_list.csv'))

    # get dataset: train / test / scaffold test

    train = pd.read_csv(os.path.join(data_path, 'train.csv'))
    test = pd.read_csv(os.path.join(data_path, 'test.csv'))
    test_scaffolds = pd.read_csv(os.path.join(data_path, 'test_scaffolds.csv'))

    if args.func == uc_sampling:
        args.func(args, train, test, test_scaffolds, toklen_data,
                  scaler, SRC, TRG, device, logger)

    elif args.func == p_sampling:
        args.func(args, train, toklen_data, scaler,
                  SRC, TRG, device, logger)

    elif args.func == sca_sampling:
        args.func(args, toklen_data, train, test_scaffolds,
                  scaler, SRC, TRG, device, logger)

    elif args.func == psca_sampling:
        args.func(args, toklen_data, train, test_scaffolds,
                  scaler, SRC, TRG, device, logger)
        
    elif args.func == mol_interpolation:
        args.func(args, toklen_data, train, test_scaffolds,
                  scaler, SRC, TRG, device, logger)

    # Not finished

    elif args.func == visualize_attention:
        args.func(args, toklen_data, train, test,
                  test_scaffolds, scaler, SRC, TRG,
                  device, logger)

    # Not finished

    elif args.func == model_selection: # only for vaetf
        args.func(args, train, test, toklen_data,
                  scaler, SRC, TRG, device)
