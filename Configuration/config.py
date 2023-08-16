def model_opts(parser):
    parser.add_argument('-N', type=int, default=6, help="# of encoder/decoder")
    parser.add_argument('-H', type=int, default=8, help="heads of attention")
    parser.add_argument('-d_ff', type=int, default=2048, help="dimension in feed forward network")
    parser.add_argument('-d_model', type=int, default=512, help="embedding dimension")
    parser.add_argument('-latent_dim', type=int, default=128)
    parser.add_argument('-dropout', type=float, default=0.1, help="Dropout probability")
    parser.add_argument('-variational', type=bool, default=True) # should be removed later
    parser.add_argument('-use_cond2dec', action='store_true')
    parser.add_argument('-use_cond2lat', action='store_true')
    parser.add_argument('-get_attn', action='store_true')
    

def train_opts(parser):
    model_opts(parser)
    
    """main options"""
    parser.add_argument('-seed', type=int)
    # parser.add_argument('-benchmark', type=str, default='moses')
    parser.add_argument('-start_epoch', type=int, default=1)
    parser.add_argument('-num_epoch', type=int, default=30)
    parser.add_argument('-batch_size', type=int, default=32)

    parser.add_argument('-property_list', nargs='+', default=[])
    parser.add_argument('-model_type', type=str, required=True)
    parser.add_argument('-model_folder', type=str, required=True)
    parser.add_argument('-use_scaffold', action='store_true')

    parser.add_argument('-randomize_prob', type=float, default=0)
    parser.add_argument('-train_params', type=str, nargs='+')

    parser.add_argument('-prepared_folder', type=str, default='/fileserver-gamma/chaoting/ML/dataset/moses/prepared')
    parser.add_argument('-util_folder', type=str, default='/fileserver-gamma/chaoting/ML/dataset/moses/utils')

    # parser.add_argument('-max_stfrlen', type=int, default=80)
    # parser.add_argument('-load_field', type=bool, default=True)
    # parser.add_argument('-load_scaler', type=bool, default=True)
    
    parser.add_argument('-debug', action='store_true')
    
    """kl annealing"""
    parser.add_argument('-use_KLA', type=bool, default=True,
                        help='Use KL annealing during training')
    parser.add_argument('-KLA_ini_beta', type=float, default=0.02,
                        help='Initial KL annealing beta value')
    parser.add_argument('-KLA_inc_beta', type=float, default=0.02,
                        help='KL annealing beta increment value')
    parser.add_argument('-KLA_max_beta', type=float, default=1.0,
                        help='Maximum KL annealing beta value')
    parser.add_argument('-KLA_beg_epoch', type=int, default=1,
                        help='Epoch at which to begin KL annealing')
    
    """learning rate scheduler"""
    parser.add_argument('-lr_scheduler', type=str, default="WarmUpDefault",
                        help="The learning rate scheduler to use (WarmUpDefault or SGDR)")
    parser.add_argument('-lr_WarmUpSteps', type=int, default=8000,
                        help="The number of warmup steps to use for the WarmUpDefault scheduler")
    parser.add_argument('-lr', type=float, default=0.0001,
                        help="The base learning rate to use for the optimizer")
    parser.add_argument('-lr_beta1', type=float, default=0.9,
                        help="The beta1 value to use for the Adam optimizer")
    parser.add_argument('-lr_beta2', type=float, default=0.98,
                        help="The beta2 value to use for the Adam optimizer")
    parser.add_argument('-lr_eps', type=float, default=1e-9,
                        help="The epsilon value to use for the Adam optimizer")
    

def preprocessing_opts(parser):
    parser.add_argument('-data_folder', type=str, default='/fileserver-gamma/chaoting/ML/dataset/')
    parser.add_argument('-benchmark', type=str, default='moses')
    parser.add_argument('-all_property_list', nargs='+', default=['logP', 'tPSA', 'QED', 'SAS'])


property_bounds = {
    'logP': [ 0.03,   4.97],
    'tPSA': [17.92, 112.83],
    'QED' : [ 0.58,   0.95]
}


def mosesPropConstraints(parser):
    parser.add_argument('-logp_lb', type=float, default=0.03)
    parser.add_argument('-logp_ub', type=float, default=4.97)
    parser.add_argument('-tpsa_lb', type=float, default=17.92)
    parser.add_argument('-tpsa_ub', type=float, default=112.83)
    parser.add_argument('-qed_lb', type=float, default=0.58)
    parser.add_argument('-qed_ub', type=float, default=0.95)


def props_opts(parser):
    # hard constraints
    parser.add_argument('-logp_lb', type=float, default=0.03)
    parser.add_argument('-logp_ub', type=float, default=4.97)
    parser.add_argument('-tpsa_lb', type=float, default=17.92)
    parser.add_argument('-tpsa_ub', type=float, default=112.83)
    parser.add_argument('-qed_lb', type=float, default=0.58)
    parser.add_argument('-qed_ub', type=float, default=0.95)


def hard_constraints_opts(parser):
    model_opts(parser) # model-size options
    mosesPropConstraints(parser)
    
    parser.add_argument('-data_path', type=str, default='/fileserver-gamma/chaoting/ML/dataset/moses/')
    parser.add_argument('-train_path', type=str, default='/fileserver-gamma/chaoting/ML/cvae-transformer/Experiment/')
    parser.add_argument('-molgct_path', type=str, default='/fileserver-gamma/chaoting/ML/molGCT/')

    parser.add_argument('-nconds', type=int, default=3)
    parser.add_argument('-conditions', nargs='+', default=['logP', 'tPSA', 'QED'])
    parser.add_argument('-max_strlen', type=int, default=80)
    parser.add_argument('-lang_format', type=str, default='SMILES')
    parser.add_argument('-data_name', type=str, default='moses')
    parser.add_argument('-load_field', type=bool, default=True)
    parser.add_argument('-load_scaler', type=bool, default=True)
