""" Implementation of all available options """
from __future__ import print_function
import argparse

model = 'TransVAE'
mode = 'train'

"""
The following website is an useful reference for "parent parser".
The "add_help=False" argument is necessary.
ref: https://stackoverflow.com/questions/7498595/python-argparse-add-argument-to-multiple-subparsers/7498853#7498853
"""
# https://stackoverflow.com/questions/10448200/how-to-parse-multiple-nested-sub-commands-using-python-argparse

def general_opts():
    general_parser = argparse.ArgumentParser(description="General setup")

    """ GENERAL SETUP """
    # general_parser.add_argument('-nconds', type=int, default=1, help="Number of conditions")
    # general_parser.add_argument('-cond_list', nargs='+', default=['logP'], help="Conditions")
    general_parser.add_argument('-nconds', type=int, default=3, help="Number of conditions")
    general_parser.add_argument('-cond_list', nargs='+', default=['logP', 'QED', 'tPSA'], help="Conditions")
    general_parser.add_argument('-n_jobs', type=int, default=1, help="number of CPU cores")
    general_parser.add_argument('-max_strlen', type=int, default=80, help="The expected max. string length")
    general_parser.add_argument('-lang_format', type=str, default='SMILES', help='Path of the original data')
    general_parser.add_argument('-dataset', type=str, default='moses', help='Path of the original data')
    general_parser.add_argument('-data_path', type=str, default='./data/moses', help='Path of the preprocessed data')
    
    general_parser.add_argument('-load_field', action='store_true', help="load the weights of fields")
    general_parser.add_argument('-field_path', type=str, default="weights", help="weights of fields")
    general_parser.add_argument('-load_scalar', action='store_true', help="load the weights of fields")
    # general_parser.add_argument('-verbose', action='store_true', help="If the results will be printed")
    
    """ MODEL ARCHITECTURE """
    general_parser.add_argument('-N', type=int, default=6, help="number of encoder/decoder")
    general_parser.add_argument('-d_model', type=int, default=512, help="embedding dimension")
    general_parser.add_argument('-d_ff', type=int, default=2048, help="dimension in feed forward network")
    general_parser.add_argument('-H', type=int, default=8, help="heads of attention")
    general_parser.add_argument('-latent_dim', type=int, default=128)
    general_parser.add_argument('-dropout', type=float, default=0.1, help="Dropout probability")
    general_parser.add_argument('-use_cond2dec', type=bool, default=False)
    general_parser.add_argument('-use_cond2lat', type=bool, default=True)
    general_parser.add_argument('-variational', action='store_true', help="if using variational")
    general_parser.add_argument('--label_smoothing', type=float, default=0.0, help="see: https://arxiv.org/abs/1512.00567")

    subparsers = general_parser.add_subparsers(help='Choose to train or test')

    """ TRAINING """
    train_opts(subparsers)

    """ TESTING """
    test_parser = subparsers.add_parser('testing')
    test_opts(test_parser)

    return general_parser


def train_opts(parser):
    parent_parser = argparse.ArgumentParser(add_help=False)

    """ KL Divergence """
    parent_parser.add_argument('-kl_beta_init', type=float, default=1e-8)
    parent_parser.add_argument('-kl_beta', type=float, default=1)
    parent_parser.add_argument('-kl_cycle', type=int, default=10)
    
    """ Optimization Tasks """
    parent_parser.add_argument('--factor', type=float, default=1.0, help="see https://arxiv.org/pdf/1706.03762.pdf")
    parent_parser.add_argument('-warmup_steps', type=int, default=4000, help="Number of warmup steps for custom decay.")
    parent_parser.add_argument('-adam_beta1', type=float, default=0.9, help="The beta1 parameter for Adam optimizer")
    parent_parser.add_argument('-adam_beta2', type=float, default=0.98, help="The beta2 parameter for Adam optimizer")
    parent_parser.add_argument('-adam_eps', type=float, default=1e-9, help="The eps parameter for Adam optimizer")

    """ Others """
    parent_parser.add_argument('-train_verbose', action='store_true', help="If the results will be printed")
    parent_parser.add_argument('-save_directory', default='train', help="Result save directory")

    """ First-Stage Training """
    train1_parser = parser.add_parser('train-1st', parents=[parent_parser])
    train1_parser.add_argument('-batch_size', type=int, default=256, help='Batch size for training')
    train1_parser.add_argument('-num_epoch', type=int, default=30, help='Number of training steps')
    train1_parser.add_argument('-starting_epoch', type=int, default=1, help="Starting epoch for training")

    """ Second-Stage Training """
    train2_parser = parser.add_parser('train-2nd', parents=[parent_parser])
    train2_parser.add_argument('-batch_size', type=int, default=256, help='Batch size for training')
    train2_parser.add_argument('-num_epoch', type=int, default=30, help='Number of training steps')
    train2_parser.add_argument('-starting_epoch', type=int, default=1, help="Starting epoch for training")


def test_opts(parser):
    """ INPUT/OUTPUT """
    group = parser.add_argument_group('Input-Output')
    group.add_argument('--data_path', required=True, help="Input data path")
    group.add_argument('--test_file-name', required=True, help="test file name without .csv")
    group.add_argument('--save-directory', default='evaluation', help="Result save directory")

    """ MODEL """
    group = parser.add_argument_group('Model')
    group.add_argument('--model-path', help="""Model path""", required=True)
    group.add_argument('--epoch', type=int, help="""Which epoch to use""", required=True)

    """ [ GENERAL ] """
    group = parser.add_argument_group('General')
    group.add_argument('--batch-size', type=int, default=128, help='Batch size for training')
    group.add_argument('--num-samples', type=int, default=100, help='Number of molecules to be generated')
    group.add_argument('--decode-type',type=str, default='multinomial',help='decode strategy')


# newly added parser
def generate_obabel_opts(parser):    
    '''Input output settings'''
    group = parser.add_argument_group('Input-Output')
    group.add_argument('--data-path', type=str, default='./data/chembl_02', help='the path of the data')
    group.add_argument('--test-file-name', required=True, help='name of the test file')
    group.add_argument('--model-path', help="""Model path""", required=True)
    group.add_argument('--save-directory', default='evaluation', help="""Result save directory""")

    group.add_argument('--obabel-selection', action='store_true', help='if the generated molecules are filtered by obabel correction')
    
    # Model to be used for generating molecules
    group = parser.add_argument_group('Model')
    group.add_argument('--epoch', type=int, help='which epoch of the model')
    
    # General
    group = parser.add_argument_group('General')
    group.add_argument('--num-steps', type=int, required=True, nargs='+', help='number of steps to target property ex: --num-steps 1 2 3')
    group.add_argument('--target-logp', type=float, default=3.5, help='setted logp of the generated molecules')
    group.add_argument('--num-molecules', type=int, default=100, help='number of starting molecules')
    group.add_argument('--batch-size', type=int, default=128, help='batch size for training')
    group.add_argument('--num-samples', type=int, default=2, help='number of molecules to be generated')
    group.add_argument('--decode-type', type=str, default='multinomial', help='decode strategy')    


def evaluation_opts(parser):
    """Evaluation options (compute properties)"""
    group = parser.add_argument_group('General')
    group.add_argument('--data-path', required=True,
                       help="""Input data path for generated molecules""")
    group.add_argument('--num-samples', type=int, default=100,
                       help='Number of molecules generated')
    group = parser.add_argument_group('Evaluation')
    group.add_argument('--range-evaluation', default='',
                       help='[ , lower, higher]; set lower when evaluating test_unseen_L-1_S01_C10_range')
    group = parser.add_argument_group('MMP')
    group.add_argument('--mmpdb-path', help='mmpdb path; download from https://github.com/rdkit/mmpdb')
    group.add_argument('--train-path', help='Training data path')
    group.add_argument('--only-desirable', help='Only check generated molecules with desirable properties',
                       action="store_true")

