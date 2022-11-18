# import configuration.opts_mlpcvae as opts
# # import configuration.opts as opts

# from trainer.mlptransformer_trainer import TransformerTrainer
# # from trainer.transformer_trainer import TransformerTrainer

import os
import argparse

from Train.train import train
from Train.mlp_train import mlp_train
from Train.tf_train import tf_train
from Configuration.config import options
from Utils.log import get_logger as gl
from Utils.seed import set_seed

DEBUG = True

def get_logger(args):
    def logger(name, log_path):
        logger = gl(name=name, log_path=log_path)
        logger.info(args)
        return logger
    return logger


if __name__ == "__main__":
    set_seed(100)

    parser = argparse.ArgumentParser()
    options(parser)
    args = parser.parse_args()    
    logger = get_logger(args)

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