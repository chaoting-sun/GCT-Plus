# import configuration.opts_mlpcvae as opts
# # import configuration.opts as opts

# from trainer.mlptransformer_trainer import TransformerTrainer
# # from trainer.transformer_trainer import TransformerTrainer

import os
import argparse
import pandas as pd
import joblib

from Preprocess.preprocess import preprocess
from Preprocess.mlp_preprocess import mlp_preprocess
from Preprocess.cond_augmentation import cond_augmentation
from Configuration.config import options
from Utils.scaler import get_scaler
from Utils.log import get_logger as gl
from Utils.seed import set_seed

DEBUG = True


def get_logger(args):
    def logger(name, log_path):
        logger = gl(name=name, log_path=log_path)
        logger.info(args)
        return logger
    return logger


def props_rescaler(scaler_path):
    def rescale(props, inverse=False):
        scaler = get_scaler(props, scaler_path)
        if inverse:
            _props = scaler.inverse_transform(props)
        else:
            _props = scaler.transform(props)
        return pd.DataFrame(_props, index=props.index,
                            columns=props.columns)
    return rescale


if __name__ == "__main__":
    set_seed(100)

    parser = argparse.ArgumentParser()
    options(parser)
    args = parser.parse_args()
    print(' '.join(f'{k}={v}' for k, v in vars(args).items()))

    print("Get logger...")
    logger = get_logger(args)

    raw_path = os.path.join(args.data_path, 'raw')
    aug_path = os.path.join(args.data_path, 'aug')

    rescaler = props_rescaler(os.path.join(args.molgct_path, 'scaler.pkl'))
    cond_augmentation(args, raw_path, aug_path, rescaler, logger=logger)


    # if args.model_type == 'mlp':
    #     mlp_preprocess(args, 'train')
    #     mlp_preprocess(args, 'validation')
    #     mlp_preprocess(args, 'test')
    # elif args.model_type == 'mlp_encoder':
    #     preprocess(args, 'validation')
    #     preprocess(args, 'train')
