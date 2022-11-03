import os
import re
import joblib
import argparse
import numpy as np
import pandas as pd

import Utils.log as ul
from Utils.seed import set_seed 
from Configuration.config import options
from Utils import allocate_gpu, get_fields
from Model.build_model import build_model
from Inference.model_prediction import Predictor
from Inference.decode_algo import MultinomialSearch, MultinomialSearchFromSource, BeamSearch, BeamSearchFromSource
from Inference.generate_uniformly import generate_uniformly
from Inference.generate_z import generate_z
from Inference.varying_z_generate import varying_z_generate
from Inference.continuity_check import continuity_check


def getSmilesGenerator(predictor, decode_algo, has_source,
                       latent_dim, max_strlen, use_cond2dec):
    if decode_algo in ("greedy", "multinomial"):
        if has_source:
            return MultinomialSearchFromSource(
                predictor, latent_dim, TRG, toklen_data, scaler,
                max_strlen, use_cond2dec, device, decode_algo
            )
        else:
            return MultinomialSearch(
                predictor, latent_dim, TRG, toklen_data, scaler,
                max_strlen, use_cond2dec, device, decode_algo
            )
    
    elif decode_algo == "beam":
        if has_source:
            return BeamSearchFromSource(
                predictor, latent_dim, TRG, toklen_data,
                scaler, max_strlen, use_cond2dec, device
            )
        else:
            return BeamSearch(
                predictor, latent_dim, TRG, toklen_data,
                scaler, max_strlen, use_cond2dec, device
            )
    else:
        exit(f"No such decoding algorithm: {decode_algo}")


def generate_smiles_to_test_z_properties():
    pass


def generate_smiles_to_compute_metrics():

    pass


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


if __name__ == "__main__":
    set_seed(0)

    # df = pd.read_csv("/fileserver-gamma/chaoting/ML/dataset/moses/aug/data_sim1.00/train.csv")
    # df = df.sample(n=100000).reset_index()
    # del df['index']
    # df.to_csv("/fileserver-gamma/chaoting/ML/dataset/moses/aug/data_sim1.00/train_tiny.csv", index=False)

    # df = pd.read_csv("/fileserver-gamma/chaoting/ML/dataset/moses/aug/data_sim1.00/validation.csv")
    # df = df.sample(n=10000).reset_index()
    # del df['index']
    # df.to_csv("/fileserver-gamma/chaoting/ML/dataset/moses/aug/data_sim1.00/validation_tiny.csv", index=False)

    # exit()
    
    print("Parse the arguments...")
    parser = argparse.ArgumentParser()
    parser = options(parser)
    args = parser.parse_args()

    print("Add the logger...")
    logger = get_logger(args)

    # if os.path.exists(args.storage_path) and not re.search("test", args.storage_path):
    #     exit("You may be using a formal folder. Please check it out!")
    os.makedirs(args.storage_path, exist_ok=True)

    print("Looking for gpu...")
    device = allocate_gpu()

    scaler = joblib.load(os.path.join(args.molgct_path, 'scaler.pkl'))
    
    fields, SRC, TRG = get_fields(args.conditions, args.molgct_path)
    toklen_data = pd.read_csv(os.path.join(
        args.data_path, 'raw', 'train', 'toklen_list.csv'))

    print("Get train smiles...")
    train_smiles = pd.read_csv(os.path.join(args.data_path, 'raw', 'train', 'smiles_serial.csv'))
    train_smiles = train_smiles['smiles'].tolist()

    print("build the model...")
    model = build_model(args, len(SRC.vocab), len(TRG.vocab)).to(device)
    model.eval()

    print("Get the generator...")
    decoder = getattr(model, args.decode_type)
    predictor = Predictor(args.use_cond2dec, decoder, model.encode)

    generator = getSmilesGenerator(predictor, args.decode_algo,
                                   args.has_source, args.latent_dim,
                                   args.max_strlen, args.use_cond2dec)

    # if args.has_source:
    #     generate_z(args, smiles_generator, fields, device, TRG, logger)
    # else:
    #     greedy_generate(args, smiles_generator, train_smiles, logger)
    #     # varying_z_generate(args, smiles_generator, fields, device, logger, SRC, TRG)
    #     # generate_uniformly(args, smiles_generator, train_smiles, logger)
    
    if args.continuity_check:
        continuity_check(generator, args.latent_dim, args.conditions, args.storage_path,
                         args.properties, args.toklen, args.n_steps, args.n_samples,
                         args.n_jobs, train_smiles, logger)
    

