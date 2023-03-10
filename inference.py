import os
import re
import joblib
import argparse
import numpy as np
import pandas as pd
import torch

from Configuration.config_default import benchmark_settings

from Utils.scaler import get_scaler
from Utils.log import get_logger
from Utils.seed import set_seed 
from Configuration.config import hard_constraints_opts
from Utils import allocate_gpu
from Utils.field import smiles_fields, condition_fields
from Inference.uniform_generation import fast_uniform_generation
# from Inference.generate_z import generate_z
# from Inference.varying_z_generate import varying_z_generate
from Inference.continuity_check import continuity_check
from Inference.fast_continuity_check import fast_continuity_check
from Inference.src_generation import fast_src_generation
from Inference.src_rotator_generation import fast_src_rotator_generation
from Inference.test_encoder import fast_test_encoder
from Inference.test_decoder import test_decoder
from Inference.src_generation_mmps import fast_src_generation_mmps

# from Inference.atten_generate import atten_generate


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

        
def add_args(parser):
    """ Continuity check on the latent space:
    
    Test if the latent space of the model is continuous
    by sampling multiple points on the line between two
    latent points.
    """
    hard_constraints_opts(parser)

    subparsers = parser.add_subparsers(help='choose test methods')

    parent_parser = argparse.ArgumentParser(add_help=False)

    parent_parser.add_argument('-benchmark', type=str, default='moses')
    parent_parser.add_argument('-property_list', nargs='+', default=['logP', 'tPSA', 'QED'])

    parent_parser.add_argument('-encode_type', type=str, default='encode')
    parent_parser.add_argument('-decode_type', type=str, default='decode')
    parent_parser.add_argument('-decode_algo', type=str, default='greedy')

    parent_parser.add_argument('-inference_path', type=str, default='/fileserver-gamma/chaoting/ML/cvae-transformer/Inference-Dataset')
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
    ug_parser.add_argument('-n_each_sampling', type=int, default=10)
    ug_parser.add_argument('-n_each_prop', type=int, default=5)
    """
    Source generation
    """
    sg_parser = subparsers.add_parser('src-generation', parents=[parent_parser])
    sg_parser.add_argument('-src_generation', action='store_true')
    sg_parser.add_argument('-n_steps', type=int, nargs='+', default=[1])
    sg_parser.add_argument('-n_samples', type=int, default=1000)
    sg_parser.add_argument('-n_selections', type=int, default=5)
    sg_parser.add_argument('-src_smiles', type=str)
    sg_parser.add_argument('-trg_props', type=float, nargs='+')

    """
    Source generation mmps
    """
    sgm_parser = subparsers.add_parser('src-generation-mmps', parents=[parent_parser])
    sgm_parser.add_argument('-src_generation_mmps', action='store_true')

    sgm_parser.add_argument('-data_folder', type=str)
    sgm_parser.add_argument('-data_name', type=str)
    sgm_parser.add_argument('-n_steps', type=int, nargs='+', default=[1])
    sgm_parser.add_argument('-n_samples', type=int, default=1000)
    sgm_parser.add_argument('-n_samples_each_time', type=int, default=1000)

    """
    Source rotator generation
    """
    sg_parser = subparsers.add_parser('src-rotator-generation', parents=[parent_parser])
    sg_parser.add_argument('-src_rotator_generation', action='store_true')
    sg_parser.add_argument('-n_steps', type=int, nargs='+', default=[1])
    sg_parser.add_argument('-n_samples', type=int, default=100)
    sg_parser.add_argument('-n_selections', type=int, default=5)
    sg_parser.add_argument('-src_smiles', type=str)
    sg_parser.add_argument('-trg_props', type=float, nargs='+')
    sg_parser.add_argument('-use_cvaetf_path', type=str, default='/fileserver-gamma/chaoting/ML/cvae-transformer/Experiment/transformer/model_25.pt')

    """
    encoder test
    """
    et_parser = subparsers.add_parser('encoder-test', parents=[parent_parser])
    et_parser.add_argument('-encoder_test', action='store_true')
    
    """
    decoder test
    """
    dt_parser = subparsers.add_parser('decoder-test', parents=[parent_parser])
    dt_parser.add_argument('-decoder_test', action='store_true')


def find_best_source():
    props = pd.read_csv('/fileserver-gamma/chaoting/ML/dataset/moses/raw/train/prop_serial.csv')
    smiles = pd.read_csv('/fileserver-gamma/chaoting/ML/dataset/moses/raw/train/smiles_serial.csv')
    sp = pd.concat([smiles, props], axis=1)

    sp = sp.loc[(abs(sp.logP-2.84) < 0.02) &
                (abs(sp.tPSA-58.11) < 1.1) &
                (abs(sp.QED-0.89) < 0.02)
                ]
    print(sp)
    exit()


if __name__ == "__main__":
    set_seed(0)

    print("Parse the arguments...")
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    print("Add the logger...")
    logger = get_logger()

    print("Looking for gpu...")
    device = allocate_gpu()
    print("device:", device)
    
    bm = benchmark_settings[args.benchmark]
    args.max_strlen = bm['max_strlen']

    scaler = get_scaler(args.property_list, os.path.join(args.data_folder, 'utils'))
    SRC, TRG = smiles_fields(os.path.join(args.data_folder, 'utils'))
    COND = condition_fields(args.property_list)
    
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

    if hasattr(args, 'src_generation_mmps'):
        print('use function - fast_src_generation_mmps...')
        fast_src_generation_mmps(args, toklen_data, train_smiles, 
                                 scaler, SRC, TRG, COND, device, logger)

    if hasattr(args, 'src_rotator_generation'):
        fast_src_rotator_generation(args, toklen_data, train_smiles, 
                                     scaler, SRC, TRG, COND, device, logger)

    if hasattr(args, 'encoder_test'):
        fast_test_encoder(args, toklen_data, scaler, SRC, TRG, COND, device)

    if hasattr(args, 'decoder_test'):
        test_decoder(args, toklen_data, scaler, SRC, TRG, device)