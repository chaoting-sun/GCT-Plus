""" Implementation of all available options """
from __future__ import print_function


def general_opts(parser):
    """ MAIN SETUP """
    group = parser.add_argument_group('General_options')
    group.add_argument('-cond_dim', type=int, default=3, help="Number of conditions")
    group.add_argument('-cond_list', nargs='+', default=['logP', 'QED', 'tPSA'], help="Conditions")
    group.add_argument('-odata_path', type=str, default='./data/moses', help='Path of the original data')
    group.add_argument('-pdata_path', type=str, default='.', help='Path of the preprocessed data')
    group.add_argument('-load_weights', action='store_true', help="load the weights of fields")
    group.add_argument('-verbose', type=bool, default=True, help="If the results will be printed")
    group.add_argument('-max_strlen', type=int, default=80, help="The expected max. string length")

    """ TRAINING SECTION """    
    subparsers = parser.add_subparsers()
    train_parser = subparsers.add_parser('training')
    train_opts(train_parser)

    """ TESTING SECTION """
    subparsers = parser.add_subparsers()
    testing_parser = subparsers.add_parser('testing')
    testing_opts(testing_parser)


def train_opts(parser):
    """ TRAINING CHOICES """
    # Common training options
    group = parser.add_argument_group('Training_options')
    group.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    group.add_argument('--num-epoch', type=int, default=60, help='Number of training steps')
    group.add_argument('--starting-epoch', type=int, default=1, help="Starting epoch for training")

    """ INPUT/OUTPUT """
    # Input output settings
    group = parser.add_argument_group('Input-Output')
    group.add_argument('--data-path', required=True, help="Input data path")
    group.add_argument('--save-directory', default='train', help="Result save directory")
    
    """ MODEL OPTIONS """ 
    subparsers = parser.add_subparsers()
    transformer_parser = subparsers.add_parser('transvae')
    train_opts_transvae(transformer_parser)


def train_opts_transvae(parser):
    """ ARCHITECTURE OPTIONS """
    # Model architecture
    group = parser.add_argument_group('Model')
    group.add_argument('-N', type=int, default=6, help="number of encoder/decoder")
    group.add_argument('-H', type=int, default=8, help="heads of attention")
    group.add_argument('-d_model', type=int, default=256, help="embedding dimension")
    group.add_argument('-d_ff', type=int, default=2048, help="dimension in feed forward network")
    group.add_argument('-latent_dim', type=int, default=128)
    group.add_argument('-use_cond2dec', type=bool, default=True)
    group.add_argument('-use_cond2lat', type=bool, default=False)
    # Regularization
    group.add_argument('--dropout', type=float, default=0.1, help="Dropout probability")
    group.add_argument('--label_smoothing', type=float, default=0.0, help="see: https://arxiv.org/abs/1512.00567")
    # KL Annealing
    group.add_argument('-kl_beta_init', type=float, default=1e-8)
    group.add_argument('-kl_beta', type=float, default=1)
    group.add_argument('-kl_cycle', type=int, default=10)
    
    """ OPTIMIZATION OPTIONS """ 
    group = parser.add_argument_group('Optimization')
    group.add_argument('--factor', type=float, default=1.0, help="see https://arxiv.org/pdf/1706.03762.pdf")
    group.add_argument('-warmup_steps', type=int, default=4000, help="Number of warmup steps for custom decay.")
    group.add_argument('-adam_beta1', type=float, default=0.9, help="The beta1 parameter for Adam optimizer")
    group.add_argument('-adam_beta2', type=float, default=0.98, help="The beta2 parameter for Adam optimizer")
    group.add_argument('-adam_eps', type=float, default=1e-9, help="The eps parameter for Adam optimizer")


def testing_opts(parser):
    """ INPUT/OUTPUT """
    group = parser.add_argument_group('Input-Output')
    group.add_argument('--data_path', required=True, help="Input data path")
    group.add_argument('--test_file-name', required=True, help="test file name without .csv")
    group.add_argument('--save-directory', default='evaluation', help="Result save directory")

    """ MODEL """
    group = parser.add_argument_group('Model')
    group.add_argument('--model-path', help="""Model path""", required=True)
    group.add_argument('--epoch', type=int, help="""Which epoch to use""", required=True)
    group.add_argument('-use_cond2dec', type=bool, default=True)
    group.add_argument('-use_cond2lat', type=bool, default=False)

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