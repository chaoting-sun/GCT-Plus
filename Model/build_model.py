import os
import torch

from .transformer import Transformer
from .mlpcvaetf import MLPCVAETF
from .sepmlpcvaetf import SEPMLPCVAETF
from .mlpcvaetf_encoder import MLPCVAETFEncoder
from .attencvaetf import ATTENCVAETF
from .mlp import MLP_Train


def transfer_params(trained_model, new_model):
    new_dict = new_model.state_dict()
    for name, param in trained_model.named_parameters():
        if name in new_dict:
            new_dict[name].copy_(param)
        else:
            print('Layer Not Found:', name)
            exit()


def freeze_params(model, freeze_names=None, train_names=None):
    """
    freeze parameters of a model.
    freeze_names, train_names are lists
    """
    for name, param in model.named_parameters():
        name_split = name.split('.')
        if freeze_names and name_split[0] in freeze_names:
            param.requires_grad = False
        if train_names and name_split[0] not in train_names:
            param.requires_grad = False


def build_cvaetf(hyperParameters, model_path=None):
    model = Transformer(**hyperParameters)
    if model_path:
        print("Use model path:", model_path)
        model_state = torch.load(model_path)
        if 'molgct.pt' in model_path:
            model.load_state_dict(model_state)
        else:
            model.load_state_dict(model_state['model_state_dict'])
    return model            
 

def build_mlpcvaetf(hyperParams, cvaetf_path, aug_cvaetf_path=None):
    aug_cvaetf = MLPCVAETF(**hyperParams)
    if aug_cvaetf_path:
        aug_cvaetf.load_state_dict(torch.load(aug_cvaetf_path)['model_state_dict'])
    else:
        cvaetf = build_cvaetf(hyperParams, cvaetf_path)
        # cvaetf.load_state_dict(torch.load(cvaetf_path)) # for molgct.pt
        cvaetf.load_state_dict(torch.load(cvaetf_path)['model_state_dict']) # for self-trained cvaetf
        transfer_params(cvaetf, aug_cvaetf)
    freeze_params(aug_cvaetf, train_names=['mlp'])
    return aug_cvaetf


def build_sepmlpcvaetf(hyperParams, cvaetf_path, aug_cvaetf_path=None):
    aug_cvaetf = SEPMLPCVAETF(**hyperParams)
    if aug_cvaetf_path:
        aug_cvaetf.load_state_dict(torch.load(aug_cvaetf_path)['model_state_dict'])
    else:
        cvaetf = build_cvaetf(hyperParams, cvaetf_path)
        # cvaetf.load_state_dict(torch.load(cvaetf_path)) # for molgct.pt
        cvaetf.load_state_dict(torch.load(cvaetf_path)['model_state_dict']) # for self-trained cvaetf
        transfer_params(cvaetf, aug_cvaetf)
    freeze_params(aug_cvaetf, train_names=['mlp_mu', 'mlp_logvar'])
    return aug_cvaetf


def build_mlpcvaetfencoder(hyperParams, cvaetf_path, aug_cvaetf_path=None):
    aug_cvaetf = MLPCVAETFEncoder(**hyperParams)
    if aug_cvaetf_path:
        aug_cvaetf.load_state_dict(torch.load(cvaetf_path)['model_state_dict'])
    else:
        cvaetf = build_cvaetf(hyperParams, cvaetf_path)
        cvaetf.load_state_dict(torch.load(cvaetf_path))
        # cvaetf.load_state_dict(torch.load(cvaetf_path)['model_state_dict'])
        transfer_params(cvaetf, aug_cvaetf)
    freeze_params(aug_cvaetf, train_names=['mlp'])
    return aug_cvaetf


def build_attencvaetf(hyperParams, cvaetf_path, aug_cvaetf_path=None):
    aug_cvaetf = ATTENCVAETF(**hyperParams)
    if aug_cvaetf_path:
        aug_cvaetf.load_state_dict(torch.load(cvaetf_path)['model_state_dict'])
    else:
        cvaetf = build_cvaetf(hyperParams, cvaetf_path)
        # cvaetf.load_state_dict(torch.load(cvaetf_path))
        cvaetf.load_state_dict(torch.load(cvaetf_path)['model_state_dict'])
        transfer_params(cvaetf, aug_cvaetf)
    freeze_params(aug_cvaetf, train_names=['att_mu', 'att_log_var'])
    return aug_cvaetf



def get_model(args, SRC_vocab_len, TRG_vocab_len):
    hyperParams = { 'src_vocab'   : SRC_vocab_len,
                    'trg_vocab'   : TRG_vocab_len,
                    'N'           : args.N, 
                    'd_model'     : args.d_model,
                    'dff'         : args.d_ff, 
                    'h'           : args.H,
                    'latent_dim'  : args.latent_dim,
                    'dropout'     : args.dropout,
                    'nconds'      : args.nconds,
                    'use_cond2dec': args.use_cond2dec,
                    'use_cond2lat': args.use_cond2lat
                }

    if args.model_type == "transformer":
        return build_cvaetf(hyperParams, args.use_model_path)
    elif args.model_type == "mlpcvaetf_encoder":
        return build_mlpcvaetfencoder(hyperParams, args.use_cvaetf_path, args.use_model_path)
    elif args.model_type == "mlpcvaetf":
        args.use_cvaetf_path = None
        return build_mlpcvaetf(hyperParams, args.use_cvaetf_path, args.use_model_path)
    elif args.model_type == "sepmlpcvaetf":
        return build_sepmlpcvaetf(hyperParams, args.use_cvaetf_path, args.use_model_path)
    elif args.model_type == "attencvaetf":
        return build_attencvaetf(hyperParams, args.use_cvaetf_path, args.use_model_path)


# def get_model(args, SRC_vocab_len, TRG_vocab_len):
#     if args.model_type == "transformer":
#         if args.use_molgct:
#             model_path = os.path.join(args.molgct_path, 'molgct.pt')
#             assert args.use_epoch == 0
#         elif args.use_epoch >= 1:
#             model_path = os.path.join(args.model_path, f'model_{args.use_epoch}.pt')
#         elif args.use_epoch == 0:
#             model_path = None
        
#         print(f"Build transformer with model path: {model_path}")
#         model = build_transformer(
#             SRC_vocab_len, TRG_vocab_len, args.N, args.d_model,
#             args.d_ff, args.H, args.latent_dim, args.dropout,
#             args.nconds, args.use_cond2dec, args.use_cond2lat, model_path
#         )
#         return model

                   
# def build_transformer(hyperParameters, model_path, use_molgct):
#     model = Transformer(**hyperParameters)
#     if model_path is not None:
#         model_state = torch.load(model_path)
#         if use_molgct:
#             model.load_state_dict(model_state)
#         else:
#             model.load_state_dict(model_state['model_state_dict'])
#     return model


# def build_transformer(src_vocab_len, trg_vocab_len, N, d_model, d_ff,
#                       H, latent_dim, dropout, nconds, use_cond2dec,
#                       use_cond2lat, model_path):
#     tf = Transformer(src_vocab_len, trg_vocab_len, N, d_model, d_ff, H,
#                      latent_dim, dropout, nconds, use_cond2dec, use_cond2lat)
#     if model_path is not None:
#         tf.load_state_dict(torch.load(model_path))
#     return tfs


# def transfer_parameters(transformer, mlptransformer):
#     mlptf_dict = mlptransformer.state_dict()
#     for name, param in transformer.named_parameters():
#         if name in mlptf_dict:
#             mlptf_dict[name].copy_(param)
#         else:
#             print('Layer Not Found:', name)
#             exit()


# def freeze_parameters(mlptransformer, pass_keywords):
#     for name, param in mlptransformer.named_parameters():
#         name_split = name.split('.')
#         if name_split[0] in pass_keywords:
#             continue
#         param.requires_grad = False
