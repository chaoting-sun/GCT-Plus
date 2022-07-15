import torch
import torch.nn as nn
import torch.nn.functional as F

from .sublayers import Norm
from .layers import EncoderLayer, DecoderLayer
from .embeddings import Embeddings, PositionalEncoding
from .mask import create_masks


import copy
import numpy as np


def get_clones(layer, N):
    return nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):

        return F.log_softmax(self.proj(x), dim=-1)


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
            x, q_k_enc_tmp = self.layers[i](x, mask)
            q_k_enc_tmp = q_k_enc_tmp.cpu().detach().numpy()[:, np.newaxis, :, :]
            if i == 0:
                q_k_enc = q_k_enc_tmp
            else:
                q_k_enc = np.concatenate((q_k_enc, q_k_enc_tmp), axis=1)

        x = self.norm(x)

        mu = self.fc_mu(x) # d_model -> opt.latent_dim
        log_var = self.fc_log_var(x) # d_model -> opt.latent_dim
        return self.sampling(mu, log_var), mu, log_var, q_k_enc

    def sampling(self, mu, log_var):
        if self.variational:
            std = torch.exp(0.5*log_var)
            eps = torch.randn_like(std)
            return eps.mul(std).add(mu)
        else:
            return mu


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
        print('trg:', trg)
        print('e_outputs:', e_outputs)
        print('cond_input:', cond_input)
        print('src_mask:', src_mask)
        print('trg_mask:', trg_mask)

        x = self.embed(trg)

        print('x:', x[0])
        exit()

        e_outputs = self.fc_z(e_outputs)

        if self.use_cond2dec == True:
            cond2dec = self.embed_cond2dec(cond_input).view(cond_input.size(0), cond_input.size(1), -1)
            x = torch.cat([cond2dec, x], dim=1) # trg + cond
        elif self.use_cond2lat == True:
            cond2lat = self.embed_cond2lat(cond_input).view(cond_input.size(0), cond_input.size(1), -1)
            e_outputs = torch.cat([cond2lat, e_outputs], dim=1) # cond + lat

        x = self.pe(x)

        for i in range(self.N):
            x, q_k_dec1_tmp, q_k_dec2_tmp = self.layers[i](x, e_outputs, cond_input, src_mask, trg_mask)
            q_k_dec1_tmp = q_k_dec1_tmp.cpu().detach().numpy()[:, np.newaxis, :, :]
            q_k_dec2_tmp = q_k_dec2_tmp.cpu().detach().numpy()[:, np.newaxis, :, :]
            if i != 0:
                q_k_dec1 = np.concatenate((q_k_dec1, q_k_dec1_tmp), axis=1)
                q_k_dec2 = np.concatenate((q_k_dec2, q_k_dec2_tmp), axis=1)
            else:
                q_k_dec1 = q_k_dec1_tmp
                q_k_dec2 = q_k_dec2_tmp
        return self.norm(x), q_k_dec1, q_k_dec2
    

class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, N=6, d_model=256, dff=2048, h=8, latent_dim=64, 
                 dropout=0.1, nconds=3, use_cond2dec=False, use_cond2lat=False, variational=True):
        super(Transformer, self).__init__()
        # settings
        self.nconds = nconds
        self.use_cond2dec = use_cond2dec
        self.use_cond2lat = use_cond2lat
        
        # encoder/decoder
        self.encoder = Encoder(src_vocab, d_model, N, h, dff, latent_dim,
                               nconds, dropout, variational)
        self.decoder = Decoder(trg_vocab, d_model, N, h, dff, latent_dim,
                               nconds, dropout, use_cond2dec, use_cond2lat)
        # other layers
        if self.use_cond2dec == True:
            self.prop_fc = nn.Linear(trg_vocab, 1)
        # generator
        # self.generator = Generator(d_model, trg_vocab)
        self.out = nn.Linear(d_model, trg_vocab)

        # initialize parameters
        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, trg, conds):
        src_mask, trg_mask = create_masks(src, trg, conds, self.use_cond2dec)

        z, mu, log_var, q_k_enc = self.encode(src, conds, src_mask)
        d_output, q_k_dec1, q_k_dec2 = self.decode(trg, z, conds, src_mask, trg_mask)
        
        # output = self.generator(d_output)
        output = self.out(d_output)

        if self.use_cond2dec == True:
            output_prop = self.prop_fc(output[:, :self.nconds, :])
            output_mol = output[:, self.nconds:, :]
        else:
            output_prop = torch.zeros(output.size(0), self.nconds, 1)
            output_mol = output
        return output_prop, output_mol, mu, log_var, z, q_k_enc, q_k_dec1, q_k_dec2

    def encode(self, src, conds, src_mask):
        return self.encoder(src, conds, src_mask)

    def decode(self, trg, z, conds, src_mask, trg_mask):
        return self.decoder(trg, z, conds, src_mask, trg_mask)


