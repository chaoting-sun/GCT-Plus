import argparse

from pkg_resources import require
 

def options(parser):
    """ GENERAL SETUP """
    # general_parser.add_argument('-nconds', type=int, default=1, help="Number of conditions")
    # general_parser.add_argument('-cond_list', nargs='+', default=['logP'], help="Conditions")
    parser.add_argument('-nconds', type=int, default=3, help="Number of conditions")
    parser.add_argument('-conditions', nargs='+', default=['logP', 'tPSA', 'QED'], help="Conditions")
    parser.add_argument('-max_strlen', type=int, default=80, help="max strin length")
    parser.add_argument('-similarity', type=float, default=1, help="lowerbound of similarity between source and target")
    parser.add_argument('-n_jobs', type=int, default=1, help="number of CPU cores")
    parser.add_argument('-lang_format', type=str, default='SMILES', help='Path of the original data')
    parser.add_argument('-data_name', type=str, default='moses', help='Path of the original data')
    parser.add_argument('-condition_path', type=str, default='/fileserver-gamma/chaoting/ML/data/moses/')
    parser.add_argument('-serial_path', type=str, default='/fileserver-gamma/chaoting/ML/data/moses_aug/')
    parser.add_argument('-processed_path', type=str, default='/fileserver-gamma/chaoting/ML/data/moses_aug/')
    
    parser.add_argument('-load_field', action='store_true', help="load the weights of fields")
    parser.add_argument('-field_path', type=str, default="molGCT", help="weights of fields")
    parser.add_argument('-load_scaler', action='store_true', help="load the weights of fields")
    parser.add_argument('-scaler_path', type=str, default='molGCT/scaler.pkl')
    # general_parser.add_argument('-verbose', action='store_true', help="If the results will be printed")
    
    """ MODEL ARCHITECTURE """
    parser.add_argument('-N', type=int, default=6, help="number of encoder/decoder")
    parser.add_argument('-d_model', type=int, default=512, help="embedding dimension")
    parser.add_argument('-d_ff', type=int, default=2048, help="dimension in feed forward network")
    parser.add_argument('-H', type=int, default=8, help="heads of attention")
    parser.add_argument('-latent_dim', type=int, default=128)
    parser.add_argument('-dropout', type=float, default=0.1, help="Dropout probability")
    parser.add_argument('-use_cond2dec', type=bool, default=False)
    parser.add_argument('-use_cond2lat', type=bool, default=True)
    parser.add_argument('-variational', action='store_true', help="if using variational")
    parser.add_argument('--label_smoothing', type=float, default=0.0, help="see: https://arxiv.org/abs/1512.00567")

    subparsers = parser.add_subparsers(help='Choose to train or test')

    """ TRAINING """
    train_opts(subparsers)

    """ TESTING """
    evaluation_parser = subparsers.add_parser('testing')
    evaluation_opts(evaluation_parser)

    return parser


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
    train1_parser.add_argument('-train_stage', type=int, default=1)
    train1_parser.add_argument('-batch_size', type=int, default=256, help='Batch size for training')
    train1_parser.add_argument('-num_epoch', type=int, default=30, help='Number of training steps')
    train1_parser.add_argument('-starting_epoch', type=int, default=1, help="Starting epoch for training")

    """ Second-Stage Training """
    train2_parser = parser.add_parser('train-2nd', parents=[parent_parser])
    train2_parser.add_argument('-train_stage', type=int, default=2)
    train2_parser.add_argument('-batch_size', type=int, default=256, help='Batch size for training')
    train2_parser.add_argument('-num_epoch', type=int, default=30, help='Number of training steps')
    train2_parser.add_argument('-transferring_model_path', type=str, default='molGCT/molgct.pt')
    train2_parser.add_argument('-starting_epoch', type=int, default=1, help="Starting epoch for training")
    train2_parser.add_argument('-save_path', type=str, required=True)


def generate_opts(parser):
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


def evaluation_opts(parser):
    pass