import torch

from .cvae_Transformer.transformer import Transformer
from .mlpcvae_Transformer.mlptransformer import MLP_Transformer


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
    model = build_transformer(src_vocab, trg_vocab, N, d_model, d_ff,
                              H, latent_dim, dropout, nconds, use_cond2dec, 
                              use_cond2lat, transfer_path)
    model = MLP_Transformer(model, src_vocab, trg_vocab, N, d_model, d_ff, H, latent_dim, 
                            dropout, nconds, use_cond2dec, use_cond2lat, variational)
    if file_path is not None:
        model.load_state_dict(torch.load(file_path)['model_state_dict'])
    return model
