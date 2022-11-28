import argparse

def options(parser):
    """ GENERAL SETUP """
    # soft constraints
    parser.add_argument('-similarity', type=float, default=1)
    parser.add_argument('-tolerance', type=float, default=0.01)

    # hard constraints
    parser.add_argument('-data_path', type=str, default='/fileserver-gamma/chaoting/ML/dataset/moses/', help="Path of dataset")
    parser.add_argument('-nconds', type=int, default=3, help="Number of conditions")
    parser.add_argument('-conditions', nargs='+', default=['logP', 'tPSA', 'QED'], help="Conditions")
    parser.add_argument('-max_strlen', type=int, default=80, help="max strin length")
    parser.add_argument('-n_jobs', type=int, default=1, help="number of CPU cores")
    parser.add_argument('-lang_format', type=str, default='SMILES', help='Path of the original data')
    parser.add_argument('-data_name', type=str, default='moses', help='Path of the original data')
    
    # hard constraints
    parser.add_argument('-molgct_path', type=str, default='/fileserver-gamma/chaoting/ML/molGCT/')
    parser.add_argument('-load_field', type=bool, default=True)
    parser.add_argument('-load_scaler', type=bool, default=True)
    # parser.add_argument('-load_field', action='store_true', help="load the weights of fields")
    # parser.add_argument('-load_scaler', action='store_true', help="load the weights of fields")
    
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
    # parser.add_argument('--label_smoothing', type=float, default=0.0, help="see: https://arxiv.org/abs/1512.00567")
    # soft constraints
    parser.add_argument('-model_type', type=str, default='mlp_transformer')
    parser.add_argument('-use_molgct', action='store_true')
    parser.add_argument('-model_path', type=str) # new -> model path
    parser.add_argument('-use_epoch', type=int, default=0) # new -> start epoch for train; epoch for validation
    parser.add_argument('-variational', type=bool, default=True)

    # parser.add_argument('-variational', action='store_true', help="if using variational")

    """ PROPERTY BOUNDS """
    # hard constraints
    parser.add_argument('-logp_lb', type=float, default=0.03)
    parser.add_argument('-logp_ub', type=float, default=4.97)
    parser.add_argument('-tpsa_lb', type=float, default=17.92)
    parser.add_argument('-tpsa_ub', type=float, default=112.83)
    parser.add_argument('-qed_lb', type=float, default=0.58)
    parser.add_argument('-qed_ub', type=float, default=0.95)

    subparsers = parser.add_subparsers(help='Choose to train or test')

    """ TRAINING """
    train_opts(subparsers)

    """ TESTING """
    evaluation_opts(subparsers)
    # evaluation_parser = subparsers.add_parser('testing')
    # evaluation_opts(evaluation_parser)
    return parser


def train_opts(parser):
    parent_parser = argparse.ArgumentParser(add_help=False)

    """ KL Annealing """
    parent_parser.add_argument('-use_KLA', type=bool, default=True)
    parent_parser.add_argument('-KLA_ini_beta', type=float, default=0.02)
    parent_parser.add_argument('-KLA_inc_beta', type=float, default=0.02)
    parent_parser.add_argument('-KLA_max_beta', type=float, default=1.0)
    parent_parser.add_argument('-KLA_beg_epoch', type=int, default=1) # KL annealing begin

    """ Optimization Tasks """
    parent_parser.add_argument('-lr_scheduler', type=str, default="WarmUpDefault", help="WarmUpDefault, SGDR")
    parent_parser.add_argument('-lr_WarmUpSteps', type=int, default=8000, help="only for WarmUpDefault")
    parent_parser.add_argument('-lr', type=float, default=0.0001)
    parent_parser.add_argument('-lr_beta1', type=float, default=0.9)
    parent_parser.add_argument('-lr_beta2', type=float, default=0.98)
    parent_parser.add_argument('-lr_eps', type=float, default=1e-9)

    # """ KL Divergence """
    # parent_parser.add_argument('-kl_beta_init', type=float, default=1e-8)
    # parent_parser.add_argument('-kl_beta', type=float, default=1)
    # parent_parser.add_argument('-kl_cycle', type=int, default=10)
    
    # """ Optimization Tasks """
    # parent_parser.add_argument('--factor', type=float, default=1.0, help="see https://arxiv.org/pdf/1706.03762.pdf")
    # parent_parser.add_argument('-warmup_steps', type=int, default=4000, help="Number of warmup steps for custom decay.")
    # parent_parser.add_argument('-adam_beta1', type=float, default=0.9, help="The beta1 parameter for Adam optimizer")
    # parent_parser.add_argument('-adam_beta2', type=float, default=0.98, help="The beta2 parameter for Adam optimizer")
    # parent_parser.add_argument('-adam_eps', type=float, default=1e-9, help="The eps parameter for Adam optimizer")

    """ Others """
    parent_parser.add_argument('-train_verbose', action='store_true', help="If the results will be printed")
    parent_parser.add_argument('-save_directory', default='train', help="Result save directory")

    """
    First-Stage Training
    """
    train1_parser = parser.add_parser('train-1st', parents=[parent_parser])
    train1_parser.add_argument('-train_stage', type=int, default=1)
    train1_parser.add_argument('-batch_size', type=int, default=256, help='Batch size for training')
    train1_parser.add_argument('-num_epoch', type=int, default=30, help='Number of training steps')
    # train1_parser.add_argument('-start_epoch', type=int, default=1, help="Starting epoch for training") -> use "use_epoch"

    """
    Second-Stage Training
    """
    train2_parser = parser.add_parser('train-2nd', parents=[parent_parser])
    train2_parser.add_argument('-train_stage', type=int, default=2)
    # parser.add_argument('-loss_fcn', type=str, default='mse', help="the loss function during training phase")
    train2_parser.add_argument('-batch_size', type=int, default=256, help='Batch size for training')
    train2_parser.add_argument('-num_epoch', type=int, default=30, help='Number of training steps')
    train2_parser.add_argument('-transfer_path', type=str, default='molGCT/molgct.pt')
    # train2_parser.add_argument('-start_epoch', type=int, default=1, help="Starting epoch for training") -> use "use_epoch"
    # train2_parser.add_argument('-save_path', type=str, required=True)


def evaluation_opts(parser):
    parent_parser = argparse.ArgumentParser(add_help=False)
    
    # # hard constraints
    # parent_parser.add_argument('-logp_lb', type=float, default=0.03)
    # parent_parser.add_argument('-logp_ub', type=float, default=4.97)
    # parent_parser.add_argument('-tpsa_lb', type=float, default=17.92)
    # parent_parser.add_argument('-tpsa_ub', type=float, default=112.83)
    # parent_parser.add_argument('-qed_lb', type=float, default=0.58)
    # parent_parser.add_argument('-qed_ub', type=float, default=0.95)

    # soft constraints - model
    # parent_parser.add_argument('-model_directory', default='train')
    # parent_parser.add_argument('-epoch', type=int, default=20)
    parent_parser.add_argument('-encode_type', type=str, default='encode')
    parent_parser.add_argument('-decode_type', type=str, default='mlp_decode')

    # soft constraints - methods to sample smiles
    parent_parser.add_argument('-has_source', action='store_true') # should be removed
    parent_parser.add_argument('-decode_algo', default="multinomial", choices=["greedy", "multinomial", "beam"])
    parent_parser.add_argument('-samples_each', type=int, default=1000)

    # soft constraints - 
    parent_parser.add_argument('-storage_path', type=str, default='molGCT/inference')
    parent_parser.add_argument('-test_random', action='store_true')

    """ Validation on molGCT on the basic metrics:
    
    Sample on the decoder of molGCT to validate the model.
    The metrics are divided into 2 parts:
    - metrics based on distributions: validity, uniqueness,
    novelty, diversity
    - metrics based on property precision: mae, mse, max,
    min, aard, amsd of logP, tPSA, QED
    logP, tPSA, and QED are divided into 5 equal-spacing
    properties between the boundary constraints, which
    leads to 125 groups, in which the decoder sample 100
    smiles with z added by a guassian G(0, 0.2)
    """
    molgctVal_parser = parser.add_parser('molgct-validation', parents=[parent_parser])
    molgctVal_parser.add_argument('-num_points', type=int, default=5)


    """ Continuity check on the latent space:
    
    Test if the latent space of the model is continuous
    by sampling multiple points on the line between two
    latent points.
    """
    cc_parser = parser.add_parser('continuity-check', parents=[parent_parser])
    cc_parser.add_argument('-continuity_check', action='store_true')
    cc_parser.add_argument('-properties', type=float, nargs='+', default=[3.075,93.411,0.609])
    cc_parser.add_argument('-toklen', type=int, default=30)
    cc_parser.add_argument('-n_steps', type=int, default=40)
    cc_parser.add_argument('-n_samples', type=int, default=100)
    cc_parser.add_argument('-test_for', type=str, default="z")

    """ A Demo for model inference:
    
    Show a demo for model inference. Properties are to be
    entered by users, while source smiles can be optionally
    provided.
    """
    demo_parser = parser.add_parser('demo', parents=[parent_parser])
    demo_parser.add_argument('-demo', action='store_true')

    """ Sampling by decoder with self-attention:
    
    
    """
    att_parser = parser.add_parser('self-attention', parents=[parent_parser])
    att_parser.add_argument('-self_attention', action='store_false')
    att_parser.add_argument('-smiles', type=str)
    att_parser.add_argument('-toklen', type=int, default=30)
    att_parser.add_argument('-n_samples', type=int, default=100)
    att_parser.add_argument('-target_props', nargs='+', default=[3.075,93.411,0.609])
