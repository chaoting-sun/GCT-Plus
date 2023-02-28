import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sublayers import Sampler
from .layers import EncoderLayer, DecoderLayer
from .modules import Embeddings, PositionalEncoding
from .modules import Norm, create_masks, get_clones
from .ctf import CTF


class Encoder(nn.Module):
    "Pass N encoder layers, followed by a layernorm"
    def __init__(self, vocab_size, d_model, N, h, dff, latent_dim, nconds, dropout, variational=True):
        super(Encoder, self).__init__()
        self.N = N
        self.variational = variational
        # input embedding layers
        self.embed_sentence = Embeddings(d_model, vocab_size)
        self.embed_cond2enc = nn.Linear(nconds, d_model*nconds) # nn.Linear() supports TensorFloat32
        # other layers
        self.norm = Norm(d_model)
        self.pe = PositionalEncoding(d_model, dropout=dropout)
        self.layers = get_clones(EncoderLayer(h, d_model, dff, dropout), N)
        # sampling mean and var
        self.fc_mu = nn.Linear(d_model, latent_dim)
        self.fc_log_var = nn.Linear(d_model, latent_dim)

    def forward(self, src, conds, mask):
        cond2enc = self.embed_cond2enc(conds)
        cond2enc = cond2enc.view(conds.size(0), conds.size(1), -1) 
        x = self.embed_sentence(src)
        x = torch.cat([cond2enc, x], dim=1)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)

        x = self.norm(x)
        return x


class Decoder(nn.Module):
    "Pass N decoder layers, followed by a layernorm"
    def __init__(self, vocab_size, d_model, N, h, 
                 dff, latent_dim, nconds, dropout, use_cond2dec, use_cond2lat):
        super(Decoder, self).__init__()
        self.N = N
        self.d_model = d_model
        self.use_cond2dec = use_cond2dec
        self.use_cond2lat = use_cond2lat
        self.embed = Embeddings(d_model, vocab_size)
        if self.use_cond2dec == True:
            self.embed_cond2dec = nn.Linear(nconds, d_model*nconds) #concat to trg_input
        if self.use_cond2lat == True:
            self.embed_cond2lat = nn.Linear(nconds, d_model*nconds) #concat to trg_input
        self.pe = PositionalEncoding(d_model, dropout=dropout)
        self.fc_z = nn.Linear(latent_dim, d_model)
        self.layers = get_clones(DecoderLayer(h, d_model, dff, dropout, use_cond2dec, use_cond2lat), N)
        self.norm = Norm(d_model)

    def forward(self, trg, e_outputs, cond_input, src_mask, trg_mask):
        x = self.embed(trg)

        e_outputs = self.fc_z(e_outputs)

        if self.use_cond2dec == True:
            cond2dec = self.embed_cond2dec(cond_input).view(cond_input.size(0), cond_input.size(1), -1)
            x = torch.cat([cond2dec, x], dim=1) # trg + cond
        elif self.use_cond2lat == True:
            cond2lat = self.embed_cond2lat(cond_input).view(cond_input.size(0), cond_input.size(1), -1)
            e_outputs = torch.cat([cond2lat, e_outputs], dim=1)
        x = self.pe(x)

        for i in range(self.N):
            x = self.layers[i](x, e_outputs, cond_input, src_mask, trg_mask)

        return self.norm(x)
    

class AttenBlock(nn.Module):
    def __init__(self, z_dim, n_heads=8, dropout=0.1):
        assert z_dim % n_heads == 0

        super(AttenBlock, self).__init__()
        self.ln1 = nn.LayerNorm(z_dim)
        self.ln2 = nn.LayerNorm(z_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=z_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(z_dim, 4*z_dim),
            nn.GELU(),
            nn.Linear(4*z_dim, z_dim),
            nn.Dropout(0.1),
        )
    def forward(self, x):
        x1 = self.ln1(x)
        y, attn = self.attn(x1,x1,x1)
        x = x + y
        x = x + self.mlp(self.ln2(x))
        return x, attn

"""
original attenctf
"""

# class StrAttention(nn.Module):
#     def __init__(self, latent_dim, cond_dim, max_strlen,
#                  prop_embed_dim=64, n_layers=8, dropout=0.1):
#         super(StrAttention, self).__init__()
#         # assert (latent_dim + cond_dim) % n_heads == 0

#         self.prop_linear = nn.Linear(cond_dim, prop_embed_dim)
#         self.prop_ln = nn.LayerNorm(prop_embed_dim)    
#         self.atten_blocks = nn.Sequential(*[AttenBlock(latent_dim) for _ in range(n_layers)])

#         self.out = nn.Linear(max_strlen+prop_embed_dim, max_strlen)
#         self.out_ln = nn.LayerNorm(latent_dim)
    
#     def forward(self, x, mconds):
#         props = self.prop_ln(self.prop_linear(mconds))
#         props = torch.stack(tuple(props for _ in range(x.size(-1))),dim=2) # (bs, latent_dim/2, latent_dim)

#         x = torch.cat([props, x], dim=1)
#         for layer in self.atten_blocks:
#             x, atten = layer(x)
#         x = x.permute(0, 2, 1)
#         x = self.out(x)
#         x = x.permute(0, 2, 1)
#         return self.out_ln(x)


# class DimAttention(nn.Module):
#     def __init__(self, latent_dim, cond_dim, max_strlen,
#                  prop_embed_dim=64, n_layers=8):
#         super(DimAttention, self).__init__()
#         self.cond_dim = cond_dim
#         self.max_strlen = max_strlen

#         self.prop_linear = nn.Linear(cond_dim, prop_embed_dim)
#         self.prop_ln = nn.LayerNorm(prop_embed_dim)        
#         self.atten_blocks = nn.Sequential(*[AttenBlock(max_strlen) for _ in range(n_layers)])

#         self.out = nn.Linear(latent_dim+prop_embed_dim, latent_dim)
#         self.out_ln = nn.LayerNorm(latent_dim)

#     def forward(self, x, mconds):
#         props = self.prop_ln(self.prop_linear(mconds))
#         props = torch.stack(tuple(props for _ in range(x.size(1))), dim=1)
        
#         x = torch.cat([props, x], dim=2)
#         x = x.permute(0, 2, 1)
#         for layer in self.atten_blocks:
#             x, atten = layer(x)
#         x = x.permute(0, 2, 1)
#         x = self.out(x)
#         return self.out_ln(x)

"""
test the method from molgpt
"""

class StrAttention(nn.Module):
    def __init__(self, latent_dim, nconds, max_strlen,
                 n_layers=8, dropout=0.1):
        super(StrAttention, self).__init__()
        # assert (latent_dim + cond_dim) % n_heads == 0
        
        self.prop_linear = nn.Linear(nconds, latent_dim)
        self.prop_ln = nn.LayerNorm(latent_dim)
        self.type_emb = nn.Embedding(nconds, latent_dim)

        self.atten_blocks = nn.Sequential(*[AttenBlock(latent_dim) for _ in range(n_layers)])

        self.out = nn.Linear(max_strlen+1, max_strlen)
        self.out_ln = nn.LayerNorm(latent_dim)
    
    def forward(self, x, mconds):
        props = self.prop_ln(self.prop_linear(mconds.unsqueeze(1)))
        type_embeddings = self.type_emb(torch.ones((x.size(0), 1),
                                                   dtype=torch.long,
                                                   device=mconds.device))
        props += type_embeddings
        
        x = torch.cat([props, x], dim=1)
        for layer in self.atten_blocks:
            x, atten = layer(x)
        x = x.permute(0, 2, 1)
        x = self.out(x)
        x = x.permute(0, 2, 1)
        return self.out_ln(x)


class DimAttention(nn.Module):
    def __init__(self, latent_dim, nconds, max_strlen, n_layers=8):
        super(DimAttention, self).__init__()
        self.max_strlen = max_strlen

        self.prop_linear = nn.Linear(nconds, max_strlen)
        self.prop_ln = nn.LayerNorm(max_strlen)        
        self.type_emb = nn.Embedding(nconds, max_strlen)

        self.atten_blocks = nn.Sequential(*[AttenBlock(max_strlen) for _ in range(n_layers)])

        self.out = nn.Linear(latent_dim+1, latent_dim)
        self.out_ln = nn.LayerNorm(latent_dim)

    def forward(self, x, mconds):
        props = self.prop_ln(self.prop_linear(mconds.unsqueeze(1)))
        type_embeddings = self.type_emb(torch.ones((x.size(0), 1),
                                                   dtype=torch.long,
                                                   device=mconds.device))
        props += type_embeddings
        x = x.permute(0, 2, 1)
        x = torch.cat([props, x], dim=1)
        for layer in self.atten_blocks:
            x, atten = layer(x)
        x = x.permute(0, 2, 1)
        x = self.out(x)
        return self.out_ln(x)


class RotatorAttention(nn.Module):
    def __init__(self, latent_dim, nconds, max_strlen):
        super(RotatorAttention, self).__init__()

        self.string_atten = StrAttention(latent_dim, nconds, max_strlen)
        self.latdim_atten = DimAttention(latent_dim, nconds, max_strlen)

    def forward(self, x, mconds):
        x_sa = self.string_atten(x, mconds)
        x_la = self.latdim_atten(x, mconds)
        return x_sa + x_la


class ATTENCTF(CTF):
    def __init__(self, src_vocab, trg_vocab, N=6, d_model=256, dff=2048, h=8, latent_dim=64, 
                 dropout=0.1, nconds=3, use_cond2dec=False, use_cond2lat=False, variational=True):
        super().__init__(src_vocab, trg_vocab, N, d_model, dff, h, latent_dim,
                         dropout, nconds, use_cond2dec, use_cond2lat, variational)
        
        # encoder/decoder
        self.encoder = Encoder(src_vocab, d_model, N, h, dff, latent_dim,
                               nconds, dropout, variational)
        self.sampler = Sampler(d_model, latent_dim, variational)
        self.rotator = RotatorAttention(latent_dim, nconds, max_strlen=80)
        self.decoder = Decoder(trg_vocab, d_model, N, h, dff, latent_dim,
                               nconds, dropout, use_cond2dec, use_cond2lat)
        # other layers
        if self.use_cond2dec == True:
            self.prop_fc = nn.Linear(trg_vocab, 1)
        self.out = nn.Linear(d_model, trg_vocab)

    def encode(self, src, conds, src_mask):
        assert isinstance(conds, tuple) is True
        econds, mconds = conds
        x = self.encoder(src, econds, src_mask)
        z, mu, logvar = self.sampler(x)
        z = self.rotator(z, mconds)
        return z, mu, logvar

    def forward(self, src, trg, econds, mconds,
                dconds, src_mask, trg_mask):
        z, _, _ = self.encode(src, (econds, dconds), src_mask) # dconds or mconds
        output = self.decode(trg, z, dconds, src_mask, trg_mask)

        if self.use_cond2dec == True:
            output_prop = self.prop_fc(output[:, :self.nconds, :])
            output_mol = output[:, self.nconds:, :]
        else:
            output_prop = torch.zeros(output.size(0), self.nconds, 1)
            output_mol = output
        return output_prop, output_mol



