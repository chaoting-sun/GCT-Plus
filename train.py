# import configuration.opts_mlpcvae as opts
# # import configuration.opts as opts

# from trainer.mlptransformer_trainer import TransformerTrainer
# # from trainer.transformer_trainer import TransformerTrainer

import os
import argparse

# from Train.train import train
# from Train.mlp_train import mlp_train
from Train.attencvaetf_train import attencvaetf_train
from Train.mlpcvaetf_encoder_train import mlpcvaetf_encoder_train
from Train.cvaetfcut_train import cvaetfcut_train
from Train.mlpcvaetf_train import mlpcvaetf_train
from Train.sepcvaetf_train import sepcvaetf_train
from Train.sepcvaetf2_train import sepcvaetf2_train
from Train.cvaetf_train import cvaetf_train
from Train.ctf_train import ctf_train
from Train.attenctf_train import attenctf_train
from Configuration.config import options, hard_constraints_opts
from Utils.log import get_logger as gl
from Utils.seed import set_seed
from Train.plot_results import plot_results

DEBUG = True

def get_logger(args):
    def logger(name, log_path):
        logger = gl(name=name, log_path=log_path)
        logger.info(args)
        return logger
    return logger


def add_args(parser):
    hard_constraints_opts(parser)

    parser.add_argument('-similarity', type=float, default=1)
    parser.add_argument('-tolerance', type=float, default=0.01)
    parser.add_argument('-start_epoch', type=int)
    parser.add_argument('-num_epoch', type=int, default=30)
    parser.add_argument('-use_cvaetf_path', type=str)
    parser.add_argument('-use_model_path', type=str)
    parser.add_argument('-save_directory', type=str, required=True)
    parser.add_argument('-train_params', type=str, nargs='+')
    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-uninit_optimizer', action='store_true')
    parser.add_argument('-optimizer_choice', default='original', choices=['sgd', 'rmsprop', 'adagrad', 'adam', 'original'])
    parser.add_argument('-pad_to_same_len', action='store_true')

    # KL Annealing
    parser.add_argument('-use_KLA', type=bool, default=True)
    parser.add_argument('-KLA_ini_beta', type=float, default=0.02)
    parser.add_argument('-KLA_inc_beta', type=float, default=0.02)
    parser.add_argument('-KLA_max_beta', type=float, default=1.0)
    parser.add_argument('-KLA_beg_epoch', type=int, default=1) # KL annealing begin

    # Optimization Tasks
    parser.add_argument('-lr_scheduler', type=str, default="WarmUpDefault", help="WarmUpDefault, SGDR")
    parser.add_argument('-lr_WarmUpSteps', type=int, default=8000, help="only for WarmUpDefault")
    parser.add_argument('-lr', type=float, default=0.0001)
    parser.add_argument('-lr_beta1', type=float, default=0.9)
    parser.add_argument('-lr_beta2', type=float, default=0.98)
    parser.add_argument('-lr_eps', type=float, default=1e-9)

    parser.add_argument('-debug', action='store_true')


if __name__ == "__main__":
    # 0, 100, 200, 400
    set_seed(0)

    # import pandas as pd
    # data_folder = '/fileserver-gamma/chaoting/ML/dataset/moses/aug/data_sim0.70_tol0.20/'
    # data_type = 'train'
    # df = pd.read_csv(os.path.join(data_folder, f'{data_type}.csv'))
    # df.insert(len(df.columns), "trg_en", df['trg'])
    
    # df.to_csv(os.path.join(data_folder, f'{data_type}_en.csv'), index=False)
    # exit()

    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    logger = get_logger(args)

    print('model type:', args.model_type)

    if args.model_type == 'cvaetf':
        cvaetf_train(args, logger=logger)

    elif args.model_type == 'ctf':
        ctf_train(args, logger=logger)

    elif args.model_type == 'attenctf':
        attenctf_train(args, logger=logger)

    elif args.model_type == 'cvaetfcut':
        cvaetfcut_train(args, logger=logger)

    elif args.model_type == 'mlpcvaetf_encoder':
        mlpcvaetf_encoder_train(args, logger=logger)

    elif args.model_type == 'mlpcvaetf':
        mlpcvaetf_train(args, logger=logger)

    elif args.model_type == 'sepcvaetf':
        sepcvaetf_train(args, logger=logger)

    elif args.model_type == 'sepcvaetf2':
        sepcvaetf2_train(args, logger=logger)

    elif args.model_type == 'attencvaetf':
        attencvaetf_train(args, logger=logger)

    print('training finished.')