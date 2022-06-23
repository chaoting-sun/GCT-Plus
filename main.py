# import configuration.opts_mlpcvae as opts
# # import configuration.opts as opts

# from trainer.mlptransformer_trainer import TransformerTrainer
# # from trainer.transformer_trainer import TransformerTrainer

import os
import argparse

from Train.preprocess import preprocess
from Train.train import train
from Configuration.config import options


DEBUG = True


def main_preprocess(args, debug=False):
    preprocess(args, debug)


def main_train(args, debug=False):
    if not os.path.exists(os.path.join(args.data_path, 'train.csv')):
        exit('File not found: train.csv')

    if not os.path.exists(os.path.join(args.data_path, 'validation.csv')):
        exit('File not found: validation.csv')

    train(args, debug)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    options(parser)
    args = parser.parse_args()
    print(' '.join(f'{k}={v}' for k, v in vars(args).items()))
 
    main_preprocess(args, debug=DEBUG)
    # main_train(args, DEBUG)