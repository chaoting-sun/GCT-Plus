import torch

from .cvae_Transformer.transformer import Transformer
from .mlpcvae_Transformer.mlptransformer import MLP_Transformer, MLP_Encoder
from .mlpcvae_Transformer.mlptransformer import transfer_parameters, freeze_parameters


def build_transformer(src_vocab, trg_vocab, N, d_model, d_ff,
                      H, latent_dim, dropout, nconds, use_cond2dec, 
                      use_cond2lat, file_path=None):
    model = Transformer(src_vocab, trg_vocab, N, d_model, d_ff, H, 
                        latent_dim, dropout, nconds, use_cond2dec, use_cond2lat)
    if file_path is not None:
        model.load_state_dict(torch.load(file_path))
    return model


def build_mlptransformer(src_vocab, trg_vocab, N, d_model, d_ff, 
                         H, latent_dim, dropout, nconds, use_cond2dec, 
                         use_cond2lat, variational, transfer_path, file_path=None):
    tf = build_transformer(src_vocab, trg_vocab, N, d_model, d_ff,
                              H, latent_dim, dropout, nconds, use_cond2dec, 
                              use_cond2lat, transfer_path)
    mlptf = MLP_Transformer(src_vocab, trg_vocab, N, d_model, d_ff, H, latent_dim, 
                            dropout, nconds, use_cond2dec, use_cond2lat, variational)
    if file_path is not None:
        mlptf.load_state_dict(torch.load(file_path)['model_state_dict'])
    else:
        transfer_parameters(tf, mlptf)
        freeze_parameters(mlptf)
    return mlptf


def build_mlpencoder(src_vocab, trg_vocab, N, d_model, d_ff, 
                     H, latent_dim, dropout, nconds, use_cond2dec, 
                     use_cond2lat, variational, transfer_path, file_path=None):
    tf = build_transformer(src_vocab, trg_vocab, N, d_model, d_ff,
                              H, latent_dim, dropout, nconds, use_cond2dec, 
                              use_cond2lat, transfer_path)
    mlptf = MLP_Encoder(src_vocab, trg_vocab, N, d_model, d_ff, H, latent_dim,
                        dropout, nconds, use_cond2dec, use_cond2lat, variational)

    if file_path is not None:
        mlptf.load_state_dict(torch.load(file_path)['model_state_dict'])
    else:
        transfer_parameters(tf, mlptf)
        freeze_parameters(mlptf)
    return mlptf