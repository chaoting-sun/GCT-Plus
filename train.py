# import configuration.opts_mlpcvae as opts
# # import configuration.opts as opts

# from trainer.mlptransformer_trainer import TransformerTrainer
# # from trainer.transformer_trainer import TransformerTrainer

import os
import argparse

from Train.train import train
from Train.mlp_train import mlp_train
from Train.tf_train import tf_train
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
    parser.add_argument('-use_model_path', type=str)
    parser.add_argument('-save_directory', type=str, required=True)
    parser.add_argument('-train_params', type=str, nargs='+')
    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-uninit_optimizer', action='store_true')
    parser.add_argument('-optimizer_choice', default='original', choices=['sgd', 'rmsprop', 'adagrad', 'adam', 'original'])

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


if __name__ == "__main__":
    # 0, 100, 200, 400
    set_seed(0)

    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()    

    logger = get_logger(args)

    print(args)

    if args.model_type == 'mlp':
        mlp_train(args, debug=DEBUG)
        
    elif args.model_type == 'transformer':
        tf_train(args, logger=logger)

    elif args.model_type == 'mlp_encoder':
        print('Start to train MLP_Encoder')
        train(args, debug=DEBUG)
    
    elif args.model_type == 'att_encoder':
        print('Start to train ATT_Encoder')
        train(args, debug=DEBUG)