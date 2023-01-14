import os
import re
import joblib
import argparse
import numpy as np
import pandas as pd
import torch

import Utils.log as ul
from Utils.seed import set_seed 
from Configuration.config import options, hard_constraints_opts
from Utils import allocate_gpu
from Utils.field import get_inference_fields, smiles_fields, condition_fields
from Model.build_model import get_model
from Inference.model_prediction import Predictor
from Inference.decode_algo import MultinomialSearch, BeamSearch, NewBeamSearch
from Inference.uniform_generation import uniform_generation, fast_uniform_generation
# from Inference.generate_z import generate_z
# from Inference.varying_z_generate import varying_z_generate
from Inference.continuity_check import continuity_check
from Inference.fast_continuity_check import fast_continuity_check
from Inference.src_generation import fast_src_generation
from Inference.test_encoder import fast_test_encoder
from Utils.property import tanimoto_similarity as similarity_fcn

# from Inference.atten_generate import atten_generate


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
        # try:
        #     os.remove(log_path)
        # except FileNotFoundError:
        #     pass
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


def calc_distance(args):
    file_folder = "/fileserver-gamma/chaoting/ML/cvae-transformer/Inference/transformer_ep25_aug-all_ep29/check_z/toklen30/greedy/"
    z1 = torch.load(os.path.join(file_folder, 'z1.pt'))
    z2 = torch.load(os.path.join(file_folder, 'z2.pt'))
    
    distance = torch.sqrt(torch.sum((z2 - z1)**2)).item()
    print('z distance     :', distance)
    print('z distance each:', distance/50)

    logp_dist = (args.logp_ub - args.logp_lb) * 0.8
    tpsa_dist = (args.tpsa_ub - args.tpsa_lb) * 0.8
    qed_dist = (args.qed_ub - args.qed_lb) * 0.8
    print('logp distance each:', logp_dist/50)
    print('tpsa distance each:', tpsa_dist/50)
    print('qed  distance each:', qed_dist/50)
    
    print('tolP1:', (args.logp_ub - args.logp_lb)*0.01)
    print('tolP2:', (args.tpsa_ub - args.tpsa_lb)*0.01)
    print('tolP3:', (args.qed_ub - args.qed_lb)*0.01)
    exit()


def main():
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
        continuity_check(args, generator, train_smiles, logger=logger)

    # elif hasattr(args, 'self_attention'):
    #     print("[PURPOSE] Run Transformer with self-attention...")
    #     atten_generate(generator, args.smiles, args.target_props,
    #                    args.storage_path, train_smiles, SRC, TRG,
    #                    fields, args.toklen, args.conditions,
    #                    logger=logger)
        
    elif hasattr(args, 'uniform_generation'):
        print("[PURPOSE] Generate uniformly over the valid property bounds")
        uniform_generation(args, generator, train_smiles, logger)
        
        
def add_args(parser):
    """ Continuity check on the latent space:
    
    Test if the latent space of the model is continuous
    by sampling multiple points on the line between two
    latent points.
    """
    hard_constraints_opts(parser)

    subparsers = parser.add_subparsers(help='choose test methods')

    parent_parser = argparse.ArgumentParser(add_help=False)
    
    parent_parser.add_argument('-encode_type', type=str, default='encode')
    parent_parser.add_argument('-decode_type', type=str, default='decode')
    parent_parser.add_argument('-decode_algo', type=str, default='greedy')

    parent_parser.add_argument('-inference_path', type=str, default='/fileserver-gamma/chaoting/ML/cvae-transformer/Inference-Results')
    parent_parser.add_argument('-n_jobs', type=int, default=4)
    parent_parser.add_argument('-model_name', type=str, required=True)
    parent_parser.add_argument('-epoch_list', type=int, nargs='+', default=[21,22,23,24,25])
    """
    Continuity check    
    """
    cc_parser = subparsers.add_parser('continuity-check', parents=[parent_parser])
    cc_parser.add_argument('-continuity_check', action='store_true')
    cc_parser.add_argument('-properties', type=float, nargs='+', default=[3.075,93.411,0.609])
    cc_parser.add_argument('-toklen', type=int, default=30)
    cc_parser.add_argument('-n_steps', type=int, default=40)
    cc_parser.add_argument('-n_samples', type=int, default=50)
    """
    Uniform generation
    """
    ug_parser = subparsers.add_parser('uniform-generation', parents=[parent_parser])
    ug_parser.add_argument('-uniform_generation', action='store_true')
    ug_parser.add_argument('-n_each_sampling', type=int, default=100)
    ug_parser.add_argument('-n_each_prop', type=int, default=5)
    """
    Source generation
    """
    sg_parser = subparsers.add_parser('src-generation', parents=[parent_parser])
    sg_parser.add_argument('-src_generation', action='store_true')
    sg_parser.add_argument('-n_steps', type=int, nargs='+', default=[1])
    sg_parser.add_argument('-n_samples', type=int, default=10)
    sg_parser.add_argument('-n_selections', type=int, default=5)
    sg_parser.add_argument('-src_smiles', type=str)
    sg_parser.add_argument('-trg_props', type=float, nargs='+')
    """
    encoder test
    """
    et_parser = subparsers.add_parser('encoder-test', parents=[parent_parser])
    et_parser.add_argument('-encoder_test', action='store_true')


if __name__ == "__main__":
    set_seed(0)

    print("Parse the arguments...")
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    print("Add the logger...")
    logger = get_logger(args)

    print("Looking for gpu...")
    device = allocate_gpu()
    print("device:", device)
    
    scaler = joblib.load(os.path.join(args.molgct_path, 'scaler.pkl'))
    fields, SRC, TRG = get_inference_fields(args.conditions, args.molgct_path)
    COND = condition_fields(args.conditions)
        
    toklen_data = pd.read_csv(os.path.join(args.data_path, 'raw', 'train', 'toklen_list.csv'))

    print("Get train smiles...")
    train_smiles = pd.read_csv(os.path.join(args.data_path, 'raw', 'train', 'smiles_serial.csv'))
    train_smiles = train_smiles['smiles'].tolist()

    if hasattr(args, 'continuity_check'):
        fast_continuity_check(args, toklen_data, train_smiles, 
                              scaler, SRC, TRG, device, logger)
    
    if hasattr(args, 'uniform_generation'):
        fast_uniform_generation(args, train_smiles, SRC, TRG, 
                                toklen_data, scaler, device, logger)
        
    if hasattr(args, 'src_generation'):
        fast_src_generation(args, toklen_data, train_smiles, 
                            scaler, SRC, TRG, COND, device, logger)

    if hasattr(args, 'encoder_test'):
        fast_test_encoder(args, toklen_data, scaler, SRC, TRG, COND, device)
