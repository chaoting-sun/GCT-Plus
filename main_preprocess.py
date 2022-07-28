# import configuration.opts_mlpcvae as opts
# # import configuration.opts as opts

# from trainer.mlptransformer_trainer import TransformerTrainer
# # from trainer.transformer_trainer import TransformerTrainer

import os
import argparse

from Train.preprocess import preprocess
from Preprocess.mlp_preprocess import mlp_preprocess
from Configuration.config import options


DEBUG = True


def main_preprocess(args, debug=False):
    preprocess(args, debug)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    options(parser)
    args = parser.parse_args()
    print(' '.join(f'{k}={v}' for k, v in vars(args).items()))

    if args.model_type == 'mlp':
        mlp_preprocess(args, 'train')
        mlp_preprocess(args, 'validation')
        mlp_preprocess(args, 'test')
    else:
        preprocess(args, debug=DEBUG)