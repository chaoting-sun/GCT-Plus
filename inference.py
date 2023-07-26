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
from Utils.field import smiles_field, condition_fields
from Inference.continuity_check import continuity_check
# from Inference.test_encoder import test_encoder
# from Inference.test_decoder import test_decoder
from Inference.p_sampling import p_sampling
# from Inference.model_selection import model_selection

from Inference.sca_sampling import sca_sampling
from Inference.psca_sampling import psca_sampling
# from Inference.uc_sampling import uc_sampling
from Inference.visualize_attention import visualize_attention



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
    parser.add_argument('-use_scaffold', action='store_true')

    subparsers = parser.add_subparsers(help='choose test methods')

    parent_parser = argparse.ArgumentParser(add_help=False)

    parent_parser.add_argument('-benchmark', type=str, default='moses')
    parent_parser.add_argument('-property_list', nargs='+', default=[])
    # parent_parser.add_argument('-property_list', nargs='+', default=['logP', 'tPSA', 'QED'])

    parent_parser.add_argument('-model_type', type=str, required=True)
    parent_parser.add_argument('-encode_type', type=str, default='encode')
    parent_parser.add_argument('-decode_type', type=str, default='decode')
    parent_parser.add_argument('-decode_algo', type=str, default='greedy')
    parent_parser.add_argument('-top_k', type=int) # top k selection in multinomial

    parent_parser.add_argument('-inference_path', type=str, default='/fileserver-gamma/chaoting/ML/cvae-transformer/Inference-Dataset')
    parent_parser.add_argument('-n_jobs', type=int, default=16)
    parent_parser.add_argument('-data_folder', type=str)
    parent_parser.add_argument('-epoch_list', type=int, nargs='+', default=[21,22,23,24,25])

    parent_parser.add_argument('-train_path', type=str, default='/fileserver-gamma/chaoting/ML/cvae-transformer/Experiment-Dataset')
    parent_parser.add_argument('-infer_path', type=str, default='/fileserver-gamma/chaoting/ML/cvae-transformer/Inference-Dataset')
    parent_parser.add_argument('-model_name', type=str, required=True)
    parent_parser.add_argument('-epoch', type=int, required=True)

    parent_parser.add_argument('-debug', action='store_true')
    """
    Continuity check
    """
    cc_parser = subparsers.add_parser('continuity-check', parents=[parent_parser])
    cc_parser.add_argument('-continuity_check', action='store_true')
    # cc_parser.add_argument('-properties', type=float, nargs='+', default=[3.075,93.411,0.609])
    # cc_parser.add_argument('-toklen', type=int, default=30)
    # cc_parser.add_argument('-n_steps', type=int, default=40)
    # cc_parser.add_argument('-n_samples', type=int, default=50)
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
    sg_parser.add_argument('-n_samples', type=int, default=5000)
    sg_parser.add_argument('-n_selections', type=int, default=5)
    sg_parser.add_argument('-src_smiles', type=str)
    sg_parser.add_argument('-trg_props', type=float, nargs='+')

    """
    Source generation mmps
    """
    sgm_parser = subparsers.add_parser('src-generation-mmps', parents=[parent_parser])
    sgm_parser.add_argument('-src_generation_mmps', action='store_true')

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
    # et_parser.add_argument('-model_type', type=str, required=True)
    
    """
    decoder test
    """
    dt_parser = subparsers.add_parser('decoder-test', parents=[parent_parser])
    dt_parser.add_argument('-decoder_test', action='store_true')
    # dt_parser.add_argument('-model_type', type=str, required=True)

    """
    sample with a scaffold
    """
    sca_parser = subparsers.add_parser('sca-sampling', parents=[parent_parser])
    sca_parser.add_argument('-sca_sampling', action='store_true')
    sca_parser.add_argument('-n_scaffolds', type=int, default=100)
    sca_parser.add_argument('-n_samples', type=int, default=10000)
    sca_parser.add_argument('-batch_size', type=int, default=512)
    sca_parser.add_argument('-sample_from', type=str, default='train')
    sca_parser.add_argument('-molgpt', action='store_true')
    sca_parser.add_argument('-substructure', action='store_true')
    
    """
    sample with a scaffold and properties
    """
    psca_parser = subparsers.add_parser('psca-sampling', parents=[parent_parser])
    psca_parser.add_argument('-psca_sampling', action='store_true')
    psca_parser.add_argument('-n_scaffolds', type=int, default=100)
    psca_parser.add_argument('-batch_size', type=int, default=512)
    psca_parser.add_argument('-sample_from', type=str, default='train')
    psca_parser.add_argument('-n_samples', type=int, default=1000)

    """unconditioned sampling"""
    ss_parser = subparsers.add_parser('uc-sampling', parents=[parent_parser])
    ss_parser.add_argument('-uc_sampling', action='store_true')
    ss_parser.add_argument('-n_samples', type=int, default=30000)

    """property-conditioned sampling"""
    ss_parser = subparsers.add_parser('p-sampling', parents=[parent_parser])
    ss_parser.add_argument('-p_sampling', action='store_true')
    ss_parser.add_argument('-n_samples', type=int, default=30000)
    ss_parser.add_argument('-use_molgct', action='store_true')
    
    """model selection"""
    ms_parser = subparsers.add_parser('model-selection', parents=[parent_parser])
    ms_parser.add_argument('-model_selection', action='store_true')
    ms_parser.add_argument('-n_samples', type=int, default=10000)

    """visualize attention map"""
    ms_parser = subparsers.add_parser('visualize-attention', parents=[parent_parser])
    ms_parser.add_argument('-visualize_attention', action='store_true')
    # ms_parser.add_argument('-n_samples', type=int, default=10000)


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
    

# def tmp():
#     df_old = pd.read_csv(f'/fileserver-gamma/chaoting/ML/dataset/moses/aug/data_sim1.00/train.csv')
#     df_new = pd.read_csv(f'/fileserver-gamma/chaoting/ML/dataset/moses/prepared/train_sca-s1.00.csv')
    
#     update = pd.DataFrame({
#         'src'         : df_old['src'],
#         'src_scaffold': df_new['src_scaffold'],
#         'src_logP'    : df_old['src_logP'],
#         'src_tPSA'    : df_old['src_tPSA'],
#         'src_QED'     : df_old['src_QED'],
#         'trg'         : df_old['trg'],
#         'trg_scaffold': df_new['trg_scaffold'],
#         'trg_logP'    : df_old['trg_logP'],
#         'trg_tPSA'    : df_old['trg_tPSA'],
#         'trg_QED'     : df_old['trg_QED'],        
#     })
#     update.to_csv(f'/fileserver-gamma/chaoting/ML/dataset/moses/prepared/train_v0_logP-tPSA-QED.csv')
#     print(update.head())
    # exit()


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
    
    util_path = os.path.join(args.data_folder, 'utils')
    
    bm = benchmark_settings[args.benchmark]
    args.max_strlen = bm['max_strlen']
    
    # get fields    
    
    if args.model_type in ('ctf', 'vaetf', 'cvaetf', 'scacvaetfv1'):
        SRC, TRG = smiles_field(util_path)
    elif args.model_type in ('scavaetf', 'scacvaetfv3'):
        SRC, TRG = smiles_field(util_path, add_sep=True)

    scaler = None
    if len(args.property_list) > 0:
        scaler = joblib.load(os.path.join(util_path, f'scaler_{"-".join(args.property_list)}.pkl'))
    
    COND = condition_fields(args.property_list)
    
    toklen_data = pd.read_csv(os.path.join(args.data_path, 'raw', 'train', 'toklen_list.csv'))

    # get train/test smiles

    df_train = pd.read_csv(os.path.join(args.data_path, 'raw', 'train.csv'))
    train = df_train['smiles'].tolist()
    df_test = pd.read_csv(os.path.join(args.data_path, 'raw', 'test.csv'))
    test = df_test['smiles'].tolist()
    df_test_scaffolds = pd.read_csv(os.path.join(args.data_path, 'raw', 'test_scaffolds.csv'))
    test_scaffolds = df_test_scaffolds['smiles'].tolist()

    if hasattr(args, 'continuity_check'):
        continuity_check(args, toklen_data, df_train, df_test_scaffolds,
                         scaler, SRC, TRG, device, logger)
    
    # if hasattr(args, 'uniform_generation'):
    #     fast_uniform_generation(args, train_smiles, SRC, TRG, 
    #                             toklen_data, scaler, device, logger)
    
    # if hasattr(args, 'src_generation'):
    #     fast_src_generation(args, toklen_data, train_smiles, 
    #                         scaler, SRC, TRG, COND, device, logger)

    # if hasattr(args, 'src_generation_mmps'):
    #     print('use function - fast_src_generation_mmps...')
    #     fast_src_generation_mmps(args, toklen_data, train_smiles, 
    #                              scaler, SRC, TRG, COND, device, logger)

    # if hasattr(args, 'src_rotator_generation'):
    #     fast_src_rotator_generation(args, toklen_data, train_smiles, 
    #                                  scaler, SRC, TRG, COND, device, logger)

    if hasattr(args, 'encoder_test'):
        test_encoder(args, toklen_data, scaler, SRC, TRG, COND, device)

    if hasattr(args, 'decoder_test'):
        test_decoder(args, toklen_data, df_train, df_test_scaffolds,
                     scaler, SRC, TRG, COND, device)

    elif hasattr(args, 'sca_sampling'):
        sca_sampling(args, toklen_data, df_train,
                     df_test_scaffolds, scaler,
                     SRC, TRG, device, logger)

    elif hasattr(args, 'psca_sampling'):
        psca_sampling(args, toklen_data, df_train, df_test,
                      df_test_scaffolds, scaler, SRC, TRG,
                      device, logger)
        
    elif hasattr(args, 'uc_sampling'):
        uc_sampling(args, train, test, test_scaffolds,
                    toklen_data, scaler, SRC, TRG, device)

    elif hasattr(args, 'p_sampling'):
        p_sampling(args, df_train, df_test, toklen_data, scaler,
                   SRC, TRG, device, logger)

    elif hasattr(args, 'model_selection'):
        model_selection(args, df_train, df_test, test_scaffolds,
                        toklen_data, scaler, SRC, TRG, device)
        
    elif hasattr(args, 'visualize_attention'):
        visualize_attention(args, toklen_data, df_train,
                            df_test, test_scaffolds, scaler,
                            SRC, TRG, device, logger)