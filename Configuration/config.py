import argparse

def options(parser):
    """ GENERAL SETUP """
    # soft constraints
    parser.add_argument('-similarity', type=float, default=1, help="lowerbound of similarity between source and target")
    parser.add_argument('-loss_fcn', type=str, default='mse', help="the loss function during training phase")

    # hard constraints
    parser.add_argument('-data_path', type=str, default='/fileserver-gamma/chaoting/ML/dataset/moses/', help="Path of dataset")
    parser.add_argument('-nconds', type=int, default=3, help="Number of conditions")
    parser.add_argument('-conditions', nargs='+', default=['logP', 'tPSA', 'QED'], help="Conditions")
    parser.add_argument('-max_strlen', type=int, default=80, help="max strin length")
    parser.add_argument('-n_jobs', type=int, default=1, help="number of CPU cores")
    parser.add_argument('-lang_format', type=str, default='SMILES', help='Path of the original data')
    parser.add_argument('-data_name', type=str, default='moses', help='Path of the original data')

    # hard constraints    
    parser.add_argument('-molgct_path', type=str, default='molGCT/molgct.pt')
    parser.add_argument('-toklen_path', type=str, default='Data/moses/toklen_list.csv')
    parser.add_argument('-load_field', action='store_true', help="load the weights of fields")
    parser.add_argument('-field_path', type=str, default="molGCT", help="weights of fields")
    parser.add_argument('-load_scaler', action='store_true', help="load the weights of fields")
    parser.add_argument('-scaler_path', type=str, default='molGCT/scaler.pkl')
    
    """ MODEL ARCHITECTURE """
    # hard constraints (molGCT)
    parser.add_argument('-N', type=int, default=6, help="number of encoder/decoder")
    parser.add_argument('-d_model', type=int, default=512, help="embedding dimension")
    parser.add_argument('-d_ff', type=int, default=2048, help="dimension in feed forward network")
    parser.add_argument('-H', type=int, default=8, help="heads of attention")
    parser.add_argument('-latent_dim', type=int, default=128)
    parser.add_argument('-dropout', type=float, default=0.1, help="Dropout probability")
    parser.add_argument('-use_cond2dec', type=bool, default=False)
    parser.add_argument('-use_cond2lat', type=bool, default=True)
    parser.add_argument('--label_smoothing', type=float, default=0.0, help="see: https://arxiv.org/abs/1512.00567")
    # soft constraints
    parser.add_argument('-model_type', type=str, default='mlp_transformer')
    parser.add_argument('-variational', action='store_true', help="if using variational")

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
    train2_parser.add_argument('-transfer_path', type=str, default='molGCT/molgct.pt')
    train2_parser.add_argument('-start_epoch', type=int, default=1, help="Starting epoch for training")
    # train2_parser.add_argument('-save_path', type=str, required=True)


def generate_opts(parser):
    """ INPUT/OUTPUT """
    group = parser.add_argument_group('Input-Output')
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
    # soft constraints
    parser.add_argument('-epoch', type=int, default=20)
    parser.add_argument('-encode_type', type=str, default='encode')
    parser.add_argument('-decode_type', type=str, default='mlp_decode')
    parser.add_argument('-samples_each', type=int, default=1000)
    parser.add_argument('-num_points', type=int, default=5)
    parser.add_argument('-model_directory', default='train')
    parser.add_argument('-storage_path', type=str, default='molGCT/inference')
    
    parser.add_argument('-demo', action='store_true')
    parser.add_argument('-test_random', action='store_true')

    # hard constraints
    parser.add_argument('-logp_lb', type=float, default=0.03)
    parser.add_argument('-logp_ub', type=float, default=4.97)
    parser.add_argument('-tpsa_lb', type=float, default=17.92)
    parser.add_argument('-tpsa_ub', type=float, default=112.83)
    parser.add_argument('-qed_lb', type=float, default=0.58)
    parser.add_argument('-qed_ub', type=float, default=0.95)