"""
- Training
  - Phase I : Model -> Encoder + Sampler + Decoder
  - Phase II: Model -> Encoder + Sampler + MLP + Sampler + Decoder
- Inference
  - Method0: Decode -> Sampler + Decoder (Phase I)
  - Method1: Decode -> Encoder + Sampler + MLP + Sampler + Decoder
  - Method2: Decode -> MLP + Sampler + Decoder
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sublayers import Sampler
from .layers import EncoderLayer, DecoderLayer
from .modules import Embeddings, PositionalEncoding
from .modules import Norm, nopeak_mask, create_source_mask, get_clones, create_target_mask
from Model.cvaetf import CVAETF


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


class StrAttention(nn.Module):
    def __init__(self, latent_dim, nconds, n_layers=16, dropout=0.1):
        super(StrAttention, self).__init__()
        # assert (latent_dim + cond_dim) % n_heads == 0
        
        self.prop_linear = nn.Linear(nconds, latent_dim*nconds)
        self.pe = PositionalEncoding(latent_dim, dropout=dropout)
        
        self.atten_blocks = nn.Sequential(*[AttenBlock(latent_dim) for _ in range(n_layers)])
        
        self.out = nn.Linear(latent_dim, latent_dim)
        self.out_norm = Norm(latent_dim) # new
    
    def forward(self, x, mconds):
        props = self.prop_linear(mconds).view(mconds.size(0), mconds.size(1), -1)        
        x = torch.cat([props, x], dim=1)
        x = self.pe(x)
        for layer in self.atten_blocks:
            x, _ = layer(x)
        x = self.out_norm(self.out(x))
        return x[:, mconds.size(1):, :]


class RotatorAttention(nn.Module):
    def __init__(self, latent_dim, nconds):
        super(RotatorAttention, self).__init__()
        self.string_atten = StrAttention(latent_dim, nconds)
        # self.latdim_atten = DimAttention(latent_dim, nconds, max_strlen)

    def forward(self, x, mconds):
        x_sa = self.string_atten(x, mconds)
        # x_la = self.latdim_atten(x, mconds)
        # return x_sa + x_la
        return x_sa


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
        # dim: -> (batch_size, d_model*nconds)
        cond2enc = self.embed_cond2enc(conds)
        # dim: -> (batch_size, nconds, d_model)
        cond2enc = cond2enc.view(conds.size(0), conds.size(1), -1)
        # dim: -> (batch_size, src_maxstr, d_model)
        x = self.embed_sentence(src)
        # dim: -> (batch_size, nconds+src_maxtr, d_model)
        x = torch.cat([cond2enc, x], dim=1)
        x = self.pe(x)
        for i in range(self.N):
            # dim: -> (batch_size, src_maxstr, d_model)
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
        elif self.use_cond2lat == True:
            self.embed_cond2lat = nn.Linear(nconds, d_model*nconds) #concat to trg_input
        self.pe = PositionalEncoding(d_model, dropout=dropout)
        self.fc_z = nn.Linear(latent_dim, d_model)
        self.layers = get_clones(DecoderLayer(h, d_model, dff, dropout, use_cond2dec, use_cond2lat), N)
        self.norm = Norm(d_model)

    def forward(self, trg, e_outputs, cond_input, src_mask, trg_mask):
        # dim: -> (batch_size, trg_maxstr-1, d_model)
        x = self.embed(trg)

        # dim: -> (batch_size, trg_maxstr, d_model)
        e_outputs = self.fc_z(e_outputs)

        if self.use_cond2dec == True:
            cond2dec = self.embed_cond2dec(cond_input).view(cond_input.size(0), cond_input.size(1), -1)
            x = torch.cat([cond2dec, x], dim=1) # trg + cond
        elif self.use_cond2lat == True:
            # dim: -> (batch_size, nconds, d_model)
            cond2lat = self.embed_cond2lat(cond_input).view(cond_input.size(0), cond_input.size(1), -1)
            # dim: -> (batch_size, nconds+maxstr, d_model)
            e_outputs = torch.cat([cond2lat, e_outputs], dim=1) # cond + lat

        x = self.pe(x)

        for i in range(self.N):
            x = self.layers[i](x, e_outputs, cond_input, src_mask, trg_mask)
        return self.norm(x)
    

class ATTENCVAETF(CVAETF):
    def __init__(self, src_vocab, trg_vocab, N=6, d_model=256,
                 dff=2048, h=8, latent_dim=64, dropout=0.1,
                 nconds=3, use_cond2dec=False, use_cond2lat=False,
                 variational=True):
        super(ATTENCVAETF, self).__init__(src_vocab, trg_vocab, N, d_model,
                                          dff, h, latent_dim, dropout, nconds,
                                          use_cond2dec, use_cond2lat, variational)
        # settings
        self.nconds = nconds
        self.use_cond2dec = use_cond2dec
        self.use_cond2lat = use_cond2lat

        # model architecture
        self.encoder = Encoder(src_vocab, d_model, N, h, dff, latent_dim,
                               nconds, dropout, variational)
        self.sampler = Sampler(d_model, latent_dim, variational)

        self.atten_mu = RotatorAttention(latent_dim, nconds)
        self.atten_logvar = RotatorAttention(latent_dim, nconds)

        self.decoder = Decoder(trg_vocab, d_model, N, h, dff, latent_dim,
                               nconds, dropout, use_cond2dec, use_cond2lat)
        self.out = nn.Linear(d_model, trg_vocab)

        self.reset_parameters()

        # other layers
        if self.use_cond2dec == True:
            self.prop_fc = nn.Linear(trg_vocab, 1)


    def encode(self, src, conds, src_mask):        
        assert isinstance(conds, tuple) is True
        econds, mconds = conds
        x = self.encoder(src, econds, src_mask)
        _, mu, logvar = self.sampler(x)
        mu = self.atten_mu(mu, mconds)
        logvar = self.atten_logvar(logvar, mconds)
        z = self.sampler.sampling(mu, logvar)
        
        return z, mu, logvar


    def forward(self, src, trg, econds, mconds,
                dconds, src_mask, trg_mask):
        z, mu, logvar = self.encode(src, (econds, mconds), src_mask)
        output = self.decode(trg, z, dconds, src_mask, trg_mask)

        if self.use_cond2dec == True:
            output_prop = self.prop_fc(output[:, :self.nconds, :])
            output_mol = output[:, self.nconds:, :]
        elif self.use_cond2lat == True:
            output_prop = torch.zeros(output.size(0), self.nconds, 1)
            output_mol = output
        return output_prop, output_mol, mu, logvar, z


def decode(model, src, econds, mconds, dconds, sos_idx, eos_idx,
           pad_idx, max_strlen, decode_type, use_cond2dec=False):
    src_mask = create_source_mask(src, pad_idx, econds)
    z, _, _ = model.encode_att_sample(src, econds, mconds, src_mask)

    # initialize the record for break condition. 0 for non-stop, while 1 for stop 
    break_condition = torch.zeros(src.shape[0], dtype=torch.bool)
    
    # create a batch of starting tokens (1)
    ys = (torch.ones(src.shape[0], 1, requires_grad=True)*sos_idx).type_as(src.data)

    with torch.no_grad():
        for i in range(max_strlen-1):
            # create a target padding/nopeaking mask
            trg_mask = create_target_mask(ys, pad_idx, dconds, use_cond2dec)

            # predict given current sequence and latent space
            output = model.decode(ys, z, dconds, src_mask, trg_mask) # (bs, len(ys[0])+1, tar_vocab)

            # take the probability distribution of the next token
            output = output[:, -1, :]

            # normalize the distribution
            prob = F.softmax(output, dim=-1)

            if decode_type == 'greedy':
                _, next_word = torch.max(prob, dim=1)
                ys = torch.cat([ys, next_word.unsqueeze(-1)], dim=1)  # [batch_size, i]
            elif decode_type == 'multinomial':
                next_word = torch.multinomial(prob, 1) # shape: (batch_size, 1)
                ys = torch.cat([ys, next_word], dim=1) #[batch_size, i]
                next_word = torch.squeeze(next_word) # shape: (batch_size)
            
            # update the break condition. 2 is the stop token
            break_condition = (break_condition | (next_word.to('cpu')==eos_idx))
            
            # If all satisfies the break condition, then break the loop.
            if all(break_condition):
                break

    return ys