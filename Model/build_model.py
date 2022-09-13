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


def build_transformer(src_vocab, trg_vocab, N, d_model, d_ff, H,
                      latent_dim, dropout, nconds, use_cond2dec,
                      use_cond2lat, model_path):
    tf = Transformer(src_vocab, trg_vocab, N, d_model, d_ff, H,
                     latent_dim, dropout, nconds, use_cond2dec, use_cond2lat)
    if model_path is not None:
        tf.load_state_dict(torch.load(model_path))
    return tf


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
                     model_path):
    tf = build_transformer(src_vocab, trg_vocab, N, d_model, d_ff, H,
                          latent_dim, dropout, nconds, use_cond2dec, 
                          use_cond2lat, transfer_path)
    atttf = ATTEncoder(src_vocab, trg_vocab, N, d_model, d_ff, H, latent_dim,
                       dropout, nconds, use_cond2dec, use_cond2lat, variational)

    if model_path is not None:
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


def build_model(args, SRC_vocab_len, TRG_vocab_len, model_path=None, train=False):
    if args.model_type == "transformer":
        # training phase I
        model = build_transformer(SRC_vocab_len, TRG_vocab_len,
                                  args.N, args.d_model, args.d_ff,
                                  args.H, args.latent_dim, args.dropout,
                                  args.nconds, args.use_cond2dec,
                                  args.use_cond2lat, model_path)
    elif args.model_type == "mlp_transformer":
        # training phase II
        model = build_mlptransformer(SRC_vocab_len, TRG_vocab_len,
                                     args.N, args.d_model, args.d_ff, 
                                     args.H, args.latent_dim, args.dropout,
                                     args.nconds, args.use_cond2dec,
                                     args.use_cond2lat, args.variational,
                                     args.molgct_path, model_path)
    elif args.model_type == "mlp_encoder":
        # training phase II
        model = build_mlpencoder(SRC_vocab_len, TRG_vocab_len,
                                 args.N, args.d_model, args.d_ff, 
                                 args.H, args.latent_dim, args.dropout,
                                 args.nconds, args.use_cond2dec,
                                 args.use_cond2lat, args.variational,
                                 args.molgct_path, model_path)
    elif args.model_type == "att_encoder":
        # training phase II
        model = build_attencoder(SRC_vocab_len, TRG_vocab_len,
                                 args.N, args.d_model, args.d_ff, 
                                 args.H, args.latent_dim, args.dropout,
                                 args.nconds, args.use_cond2dec,
                                 args.use_cond2lat, args.variational,
                                 args.molgct_path, model_path)
    elif args.model_type == "mlp":
        # training phase II
        if train is True:
            model = build_mlptrain(SRC_vocab_len, TRG_vocab_len, args.d_model,
                                   args.latent_dim, args.dropout, args.nconds,
                                   args.variational, model_path)
        else:
            model = build_mlp(SRC_vocab_len, TRG_vocab_len,
                            args.N, args.d_model, args.d_ff, 
                            args.H, args.latent_dim, args.dropout,
                            args.nconds, args.use_cond2dec,
                            args.use_cond2lat, args.variational,
                            args.molgct_path, model_path)
    return model
