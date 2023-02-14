import os
import joblib
import argparse
import pandas as pd
from Utils.seed import set_seed
from Utils import allocate_gpu
import Utils.log as ul
from Configuration.config import hard_constraints_opts

from Test.reconstruction import reconstruction
from Utils.field import smiles_fields, condition_fields


def get_logger(args):
    def logger(name, log_path):
        logger = ul.get_logger(name=name, log_path=log_path)
        logger.info(args)
        return logger
    return logger


def add_args(parser):
    hard_constraints_opts(parser)
    subparsers = parser.add_subparsers(help='choose test methods') 
    
    parent_parser = argparse.ArgumentParser(add_help=False)

    parent_parser.add_argument('-encode_type', type=str, default='encode')
    parent_parser.add_argument('-decode_type', type=str, default='decode')
    parent_parser.add_argument('-decode_algo', type=str, default='greedy')

    parent_parser.add_argument('-inference_path', type=str, default='/fileserver-gamma/chaoting/ML/cvae-transformer/Inference-Results')
    parent_parser.add_argument('-n_jobs', type=int, default=4)
    parent_parser.add_argument('-pad_to_same_len', action='store_true')
    parent_parser.add_argument('-model_name', type=str, default='ctf')
    parent_parser.add_argument('-epoch_list', type=int, nargs='+', default=[21,22,23,24,25])

    """
    reconstruction
    """
    rc_parser = subparsers.add_parser('reconstruction', parents=[parent_parser])
    rc_parser.add_argument('-reconstruction', action='store_true')
    rc_parser.add_argument('-similarity', type=float)
    rc_parser.add_argument('-tolerance', type=float)


def sample_from_dataset():
    """same + not_same"""
    from sklearn.utils import shuffle
    data_folder = '/fileserver-gamma/chaoting/ML/dataset/moses/aug/data_sim0.50_tol0.20/'

    def sample_data(data_type):
        df = pd.read_csv(os.path.join(data_folder, f'{data_type}.csv'))
        df_same = df.loc[df.src == df.trg]
        df_not_same = df.loc[df.src != df.trg]
        df_same = df_same.sample(len(df_not_same),
                                 random_state=1,
                                 ignore_index=True)
        df = pd.concat([df_same, df_not_same], axis=0)
        return shuffle(df)
    
    train = sample_data('train')
    valid = sample_data('validation')
    
    train.to_csv(os.path.join(data_folder, 'train_half.csv'), index=False)
    valid.to_csv(os.path.join(data_folder, 'valid_half.csv'), index=False)
    

if __name__ == "__main__":
    set_seed(0)
    device = allocate_gpu()

    print("Parse the arguments...")
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    logger = get_logger(args)

    SRC, TRG = smiles_fields(args.molgct_path)
    COND = condition_fields(args.conditions)
    toklen_data = pd.read_csv(os.path.join(args.data_path, 'raw', 'train', 'toklen_list.csv'))
    scaler = joblib.load(os.path.join(args.molgct_path, 'scaler.pkl'))

    if hasattr(args, 'reconstruction'):
        reconstruction(args, toklen_data, scaler, device, logger)