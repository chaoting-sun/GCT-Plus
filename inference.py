import os
import joblib
import argparse
import pandas as pd

import Utils.log as ul
from Configuration.config import options
from Utils import allocate_gpu, get_fields
from Model.build_model import build_model
from Inference.model_prediction import Predictor
from Inference.decode_algo import MultinomialSearch, MultinomialSearchFromSource, BeamSearch, BeamSearchFromSource
from Inference.generate_uniformly import generate_uniformly


def getSmilesGenerator(predictor, decode_algo, has_source,
                       latent_dim, max_strlen, use_cond2dec):
    if decode_algo == "multinomial":
        if has_source:
            return MultinomialSearchFromSource(
                predictor, latent_dim, TRG, toklen_data,
                scaler, max_strlen, use_cond2dec, device
            )
        else:
            return MultinomialSearch(
                predictor, latent_dim, TRG, toklen_data,
                scaler, max_strlen, use_cond2dec, device
            )
    
    elif decode_algo == "beam_search":
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
        exit(f"No decodeing algorithm: {decode_algo}")


def generate_smiles_to_test_z_properties():
    pass


def generate_smiles_to_compute_metrics():

    pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser = options(parser)
    args = parser.parse_args()

    logger = ul.get_logger(name="inference",
                           log_path=os.path.join(args.storage_path, 'inference.log'))
    logger.info(args)

    device = allocate_gpu()
    logger.info(f"Get Device: {device}")

    scaler = joblib.load(os.path.join(args.molgct_path, 'scaler.pkl'))
    fields, SRC, TRG = get_fields(args.conditions, args.molgct_path)
    toklen_data = pd.read_csv(os.path.join(
        args.data_path, 'raw', 'train', 'toklen_list.csv'))

    train_smiles = pd.read_csv(os.path.join(args.data_path, 'raw', 'train', 'smiles_serial.csv'))
    train_smiles = train_smiles['smiles'].tolist()

    model = build_model(args, len(SRC.vocab), len(TRG.vocab)).to(device)
    model.eval()

    if args.has_source:
        predictor = Predictor(args.use_cond2dec,
                              getattr(model, args.decode_type),
                              model.encode)        
    else:
        predictor = Predictor(args.use_cond2dec,
                              getattr(model, args.decode_type))

    smiles_generator = getSmilesGenerator(predictor,
                                          args.decode_algo,
                                          args.has_source,
                                          args.latent_dim,
                                          args.max_strlen,
                                          args.use_cond2dec)

    generate_uniformly(args, logger, smiles_generator, train_smiles)
    
    

