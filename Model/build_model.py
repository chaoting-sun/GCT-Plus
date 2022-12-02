import os
import torch

from .transformer import Transformer
from .mlp_transformer import MLPTransformer
from .mlp_encoder import MLPEncoder
from .att_encoder import ATTEncoder
from .mlp import MLP_Train




def transfer_parameters(transformer, mlptransformer):
    mlptf_dict = mlptransformer.state_dict()
    for name, param in transformer.named_parameters():
        if name in mlptf_dict:
            mlptf_dict[name].copy_(param)
        else:
            print('Layer Not Found:', name)
            exit()


def freeze_parameters(mlptransformer, pass_keywords):
    for name, param in mlptransformer.named_parameters():
        name_split = name.split('.')
        if name_split[0] in pass_keywords:
            continue
        param.requires_grad = False


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


def build_transformer(hyperParameters, model_path):
    model = Transformer(**hyperParameters)
    if model_path:
        print("Use model path:", model_path)
        model_state = torch.load(model_path)
        if 'molgct.pt' in model_path:
            model.load_state_dict(model_state)
        else:
            model.load_state_dict(model_state['model_state_dict'])
    return model            
                    

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
#     return tf


def build_mlptransformer(src_vocab, trg_vocab, N, d_model, d_ff, H,
                         latent_dim, dropout, nconds, use_cond2dec,
                         use_cond2lat, variational, transfer_path,
                         model_path):
    tf = build_transformer(src_vocab, trg_vocab, N, d_model, d_ff, H,
                           latent_dim, dropout, nconds, use_cond2dec, 
                           use_cond2lat, transfer_path)
    mlptf = MLPTransformer(src_vocab, trg_vocab, N, d_model, d_ff, H, latent_dim, 
                           dropout, nconds, use_cond2dec, use_cond2lat, variational)
    if model_path is not None:
        mlptf.load_state_dict(torch.load(model_path)['model_state_dict'])
    else:
        transfer_parameters(tf, mlptf)
        freeze_parameters(mlptf, pass_keywords=('mlp'))
    return mlptf


def build_mlpencoder(src_vocab, trg_vocab, N, d_model, d_ff, 
                     H, latent_dim, dropout, nconds, use_cond2dec, 
                     use_cond2lat, variational, transfer_path,
                     model_path):
    tf = build_transformer(src_vocab, trg_vocab, N, d_model, d_ff, H,
                          latent_dim, dropout, nconds, use_cond2dec, 
                          use_cond2lat, transfer_path)
    mlptf = MLPEncoder(src_vocab, trg_vocab, N, d_model, d_ff, H, latent_dim,
                       dropout, nconds, use_cond2dec, use_cond2lat, variational)

    if model_path is not None:
        mlptf.load_state_dict(torch.load(model_path)['model_state_dict'])
    transfer_parameters(tf, mlptf)
    freeze_parameters(mlptf, pass_keywords=('mlp'))
    return mlptf


def build_attencoder(src_vocab, trg_vocab, N, d_model, d_ff, 
                     H, latent_dim, dropout, nconds, use_cond2dec, 
                     use_cond2lat, variational, transfer_path,
                     model_path, att_type):
    tf = build_transformer(src_vocab, trg_vocab, N, d_model, d_ff, H,
                          latent_dim, dropout, nconds, use_cond2dec, 
                          use_cond2lat, transfer_path)
    atttf = ATTEncoder(src_vocab, trg_vocab, N, d_model, d_ff, H, latent_dim, dropout,
                       nconds, use_cond2dec, use_cond2lat, variational, att_type)

    if model_path:
        atttf.load_state_dict(torch.load(model_path)['model_state_dict'])
    transfer_parameters(tf, atttf)
    freeze_parameters(atttf, pass_keywords=('att_mu', 'att_log_var'))
    return atttf


def build_mlptrain(src_vocab, trg_vocab, d_model, latent_dim, 
                   dropout, nconds, variational, model_path):
    mlptrain = MLP_Train(src_vocab, trg_vocab, d_model, latent_dim, 
                         dropout, nconds, variational)
    if model_path is not None:
        mlptrain.load_state_dict(torch.load(model_path)['model_state_dict'])
    return mlptrain


def build_mlp(src_vocab, trg_vocab, N, d_model, d_ff, 
              H, latent_dim, dropout, nconds, use_cond2dec, 
              use_cond2lat, variational, transfer_path,
              model_path):
    tf = build_transformer(src_vocab, trg_vocab, N, d_model, d_ff, H,
                          latent_dim, dropout, nconds, use_cond2dec, 
                          use_cond2lat, transfer_path)
    mlp = MLP(src_vocab, trg_vocab, N, d_model, d_ff, H, latent_dim,
              dropout, nconds, use_cond2dec, use_cond2lat, variational)
    if model_path is not None:
        mlp.load_state_dict(torch.load(model_path)['model_state_dict'])
    transfer_parameters(tf, mlp)
    freeze_parameters(mlp)
    return mlp


# def build_model(args, SRC_vocab_len, TRG_vocab_len, model_path=None, **kwargs):
def build_model(args, SRC_vocab_len, TRG_vocab_len, **kwargs):
    molgct_model_path = os.path.join(args.molgct_path, 'molgct.pt')
    if args.model_type != 'transformer' and args.use_epoch >= 1:
        model_path = os.path.join(args.model_path, f'model_{args.use_epoch}.pt')
    else:
        model_path = None

    if args.model_type == "transformer":
        # training phase I
        print(f"Build Transformer from {molgct_model_path}")
        model = build_transformer(SRC_vocab_len, TRG_vocab_len,
                                  args.N, args.d_model, args.d_ff,
                                  args.H, args.latent_dim, args.dropout,
                                  args.nconds, args.use_cond2dec,
                                  args.use_cond2lat, molgct_model_path)
    elif args.model_type == "mlp_transformer":
        # training phase II
        model = build_mlptransformer(SRC_vocab_len, TRG_vocab_len,
                                     args.N, args.d_model, args.d_ff, 
                                     args.H, args.latent_dim, args.dropout,
                                     args.nconds, args.use_cond2dec,
                                     args.use_cond2lat, args.variational,
                                     molgct_model_path, model_path)
    elif args.model_type == "mlp_encoder":
        # training phase II
        model = build_mlpencoder(SRC_vocab_len, TRG_vocab_len,
                                 args.N, args.d_model, args.d_ff, 
                                 args.H, args.latent_dim, args.dropout,
                                 args.nconds, args.use_cond2dec,
                                 args.use_cond2lat, args.variational,
                                 molgct_model_path, model_path)
    elif args.model_type == "att_encoder":
        # training phase II
        model = build_attencoder(
            SRC_vocab_len, TRG_vocab_len, args.N, args.d_model,
            args.d_ff, args.H, args.latent_dim, args.dropout,
            args.nconds, args.use_cond2dec, args.use_cond2lat,
            args.variational, molgct_model_path, model_path,
            kwargs['att_type']
        )
    elif args.model_type == "mlp":
        # training phase II
        if kwargs['train']:
            model = build_mlptrain(SRC_vocab_len, TRG_vocab_len, args.d_model,
                                   args.latent_dim, args.dropout, args.nconds,
                                   args.variational, model_path)
        else:
            model = build_mlp(SRC_vocab_len, TRG_vocab_len,
                              args.N, args.d_model, args.d_ff, 
                              args.H, args.latent_dim, args.dropout,
                              args.nconds, args.use_cond2dec,
                              args.use_cond2lat, args.variational,
                              molgct_model_path, model_path)
    return model

"""
new function for building model. old: biuld_model
"""




def get_model(args, SRC_vocab_len, TRG_vocab_len):
    hyperParams = { 'src_vocab': SRC_vocab_len,
                    'trg_vocab': TRG_vocab_len,
                    'N': args.N, 'd_model': args.d_model,
                    'dff': args.d_ff, 'h': args.H,
                    'latent_dim': args.latent_dim,
                    'dropout': args.dropout,
                    'nconds': args.nconds,
                    'use_cond2dec': args.use_cond2dec,
                    'use_cond2lat': args.use_cond2lat
                    }

    if args.model_type == "transformer":
        model = build_transformer(hyperParams, args.use_model_path)
        return model


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