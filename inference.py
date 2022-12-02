import os
import re
import joblib
import argparse
import numpy as np
import pandas as pd

import Utils.log as ul
from Utils.seed import set_seed 
from Configuration.config import options
from Utils import allocate_gpu
from Utils.field import get_inference_fields
from Model.build_model import build_model, get_model
from Inference.model_prediction import Predictor
from Inference.decode_algo import MultinomialSearch, BeamSearch, NewBeamSearch
from Inference.uniform_generation import uniform_generation
from Inference.generate_z import generate_z
from Inference.varying_z_generate import varying_z_generate
from Inference.continuity_check import continuity_check_on_z, continuity_check_on_conds
from Inference.atten_generate import atten_generate


def get_smiles_generator(predictor, decode_algo, latent_dim,
                         max_strlen, use_cond2dec):
    if decode_algo in ("greedy", "multinomial"):
        return MultinomialSearch(
            predictor, latent_dim, TRG, toklen_data, scaler,
            max_strlen, use_cond2dec, device, decode_algo
        )
    elif decode_algo == "beam":
        return BeamSearch(
            predictor, latent_dim, TRG, toklen_data,
            scaler, max_strlen, use_cond2dec, device
        )
    elif decode_algo == "newbeam":
        return NewBeamSearch(
            predictor, latent_dim, TRG, toklen_data,
            scaler, max_strlen, use_cond2dec, device
        )        
    else:
        exit(f"No such decoding algorithm: {decode_algo}")


def get_logger(args):
    def logger(name, log_path):
        try:
            os.remove(log_path)
        except FileNotFoundError:
            pass
        logger = ul.get_logger(name=name, log_path=log_path)
        logger.info(args)
        return logger
    return logger


def get_generator(args, SRC, TRG, device):
    if args.model_type == 'transformer':
        print("get model...")
        model = get_model(args, len(SRC.vocab), len(TRG.vocab))
    elif args.model_type == "att_encoder":
        model = build_model(args, len(SRC.vocab),
                            len(TRG.vocab), att_type='ATT_v5')
    else:
        model = build_model(args, len(SRC.vocab), len(TRG.vocab))

    model = model.to(device)
    model.eval()
    print('Get predictor...')
    predictor = Predictor(args.use_cond2dec,
                          getattr(model, args.decode_type),
                          getattr(model, args.encode_type))
    print('Get generator...')
    generator = get_smiles_generator(predictor,
                                     args.decode_algo,
                                     args.latent_dim,
                                     args.max_strlen,
                                     args.use_cond2dec)
    return generator


if __name__ == "__main__":
    set_seed(0)
    
    print("Parse the arguments...")
    parser = argparse.ArgumentParser()
    parser = options(parser)
    args = parser.parse_args()
    print(args)

    print("Add the logger...")
    logger = get_logger(args)

    os.makedirs(args.storage_path, exist_ok=True)

    print("Looking for gpu...")
    device = allocate_gpu()
    print("device:", device)

    scaler_path = os.path.join(args.molgct_path, 'scaler.pkl')
    toklen_path = os.path.join(args.data_path, 'raw',
                               'train', 'toklen_list.csv')
    train_smiles_path = os.path.join(args.data_path, 'raw',
                                     'train', 'smiles_serial.csv')
    
    scaler = joblib.load(scaler_path)
    fields, SRC, TRG = get_inference_fields(args.conditions, args.molgct_path)
    toklen_data = pd.read_csv(toklen_path)

    print("Get train smiles...")
    train_smiles = pd.read_csv(train_smiles_path)
    train_smiles = train_smiles['smiles'].tolist()

    print("Get smiles generator...")
    generator = get_generator(args, SRC, TRG, device)

    # if args.has_source:
    #     generate_z(args, smiles_generator, fields, device, TRG, logger)
    # else:
    #     greedy_generate(args, smiles_generator, train_smiles, logger)
    #     # varying_z_generate(args, smiles_generator, fields, device, logger, SRC, TRG)
    #     # generate_uniformly(args, smiles_generator, train_smiles, logger)
    
    if hasattr(args, 'continuity_check'):
        # no source smiles. Don't know how to extend "has_source"
        print("[PURPOSE] Check the continuity property...")

        if args.test_for == "z":
            continuity_check_on_z(args, generator, train_smiles, logger=logger)
        elif args.test_for == "conds":
            continuity_check_on_conds(args, generator, train_smiles, logger)

    elif hasattr(args, 'self_attention'):
        print("[PURPOSE] Run Transformer with self-attention...")
        atten_generate(generator, args.smiles, args.target_props,
                       args.storage_path, train_smiles, SRC, TRG,
                       fields, args.toklen, args.conditions,
                       logger=logger)
        
    elif hasattr(args, 'uniform_generation'):
        print("[PURPOSE] Generate uniformly over the valid property bounds")
        uniform_generation(args, generator, train_smiles, logger)
        
        
