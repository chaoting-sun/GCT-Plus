# import configuration.opts_mlpcvae as opts
# # import configuration.opts as opts

# from trainer.mlptransformer_trainer import TransformerTrainer
# # from trainer.transformer_trainer import TransformerTrainer

import os
import argparse

from Train.train import train
from Configuration.config import options


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    options(parser)
    args = parser.parse_args()
    # print(' '.join(f'{k}={v}' for k, v in vars(opt).items()))
 
    if not os.path.exists(os.path.join(args.data_path, 'train.csv')):
        exit('File not found: train.csv')

    if not os.path.exists(os.path.join(args.data_path, 'validation.csv')):
        exit('File not found: validation.csv')

    train(args, DEBUG=True)
