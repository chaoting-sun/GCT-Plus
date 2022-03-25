""" Implementation of all available options """
from __future__ import print_function


def train_opts(parser):
    """ [ TRAINING CHOICES ] """
    # Common training options
    group = parser.add_argument_group('Training_options')
    group.add_argument('--batch-size', type=int, default=16, help='Batch size for training')
    group.add_argument('--num-epoch', type=int, default=60, help='Number of training steps')
    group.add_argument('--starting-epoch', type=int, default=1, help="Starting epoch for training")
    group.add_argument('-nconds', type=int, default=3, help="Number of conditions")

    """ [ INPUT/OUTPUT ] """
    # Input output settings
    group = parser.add_argument_group('Input-Output')
    group.add_argument('--data-path', required=True, help="Input data path")
    group.add_argument('--save-directory', default='train', help="Result save directory")
    
    """ [ MODEL OPTIONS ] """ 
    subparsers = parser.add_subparsers()
    transformer_parser = subparsers.add_parser('transformer')
    train_opts_transvae(transformer_parser)


def train_opts_transvae(parser):
    """ [ MODEL OPTIONS ] """
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
    group.add_argument('--label-smoothing', type=float, default=0.0,
                       help="""Label smoothing value epsilon.
                       Probabilities of all non-true labels
                       will be smoothed by epsilon / (vocab_size - 1).
                       Set to zero to turn off label smoothing.
                       For more detailed information, see:
                       https://arxiv.org/abs/1512.00567""")
    
    """ [ OPTIMIZATION OPTIONS ] """ 
    group = parser.add_argument_group('Optimization')
    group.add_argument('--factor', type=float, default=1.0,
                       help="""Factor multiplied to the learning rate scheduler formula in NoamOpt. 
                       For more information about the formula,  
                       see paper Attention Is All You Need https://arxiv.org/pdf/1706.03762.pdf""")
    group.add_argument('--warmup-steps', type=int, default=4000, help="Number of warmup steps for custom decay.")
    group.add_argument('--adam-beta1', type=float, default=0.9, help="The beta1 parameter for Adam optimizer")
    group.add_argument('--adam-beta2', type=float, default=0.98, help="The beta2 parameter for Adam optimizer")
    group.add_argument('--adam-eps', type=float, default=1e-9, help="The eps parameter for Adam optimizer")
    

def generate_opts(parser):
    """ [ INPUT/OUTPUT ] """
    group = parser.add_argument_group('Input-Output')
    group.add_argument('--data-path', required=True, help="Input data path")
    group.add_argument('--test-file-name', required=True, help="test file name without .csv")
    group.add_argument('--save-directory', default='evaluation', help="Result save directory")

    """ [ MODEL ] """
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