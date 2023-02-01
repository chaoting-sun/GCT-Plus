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


"""ATT-v1"""
class ATTEN1(nn.Module):
    def __init__(self, latent_dim, n_heads=2, cond_dim=6, d_in=512,
                 d_mid=256, dropout=0.1, batch_first=True):
        super(ATTEN1, self).__init__()
        # assert (latent_dim + cond_dim) % n_heads == 0
        self.linear_in = nn.Linear(latent_dim+cond_dim, d_in)
        self.att = nn.MultiheadAttention(embed_dim=d_in,
                                         num_heads=n_heads,
                                         dropout=dropout,
                                         batch_first=batch_first
                                         )
        self.norm_1 = Norm(d_in)
        self.linear_1 = nn.Linear(d_in, d_mid)
        self.dropout1 = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_mid, latent_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.norm_2 = Norm(latent_dim)
    
    def forward(self, x, mconds):
        """
        concatenate x and mconds to get input of dimension
        -> (batch_size, max_strlen+mconds.size(-1), latent_dim)
        N = batch_size
        L = max_strlen+mconds.size(-1)
        Eq = Ek = Ev = latent_dim
        """ 
        mconds = torch.stack(tuple(mconds for _ in range(x.size(1))),dim=1)
        x = torch.cat([mconds, x], dim=2)

        x1 = self.linear_in(x)
        attn_out, attn_out_weights = self.att(x1, x1, x1)
        x1 = x1 + attn_out
        x1 = self.norm_1(x1)

        x = self.dropout1(self.linear_1(x1))
        x = self.dropout2(self.linear_2(x))
        # x = x + x1
        x = self.norm_2(x)
        return x


"""ATT-v2"""
class ATTEN2(nn.Module):
    def __init__(self, latent_dim, n_heads=2, cond_dim=6, d_in=512,
                 d_mid=256, dropout=0.1, batch_first=True, max_strlen=80):
        super(ATTEN2, self).__init__()
        self.cond_dim = cond_dim
        self.max_strlen = max_strlen

        # assert (latent_dim + cond_dim) % n_heads == 0
        # PROBLEM: string length cannot change
        self.linear_in = nn.Linear(cond_dim+max_strlen, d_in)
        self.att = nn.MultiheadAttention(embed_dim=d_in,
                                         num_heads=n_heads,
                                         dropout=dropout,
                                         batch_first=batch_first
                                         )
        self.norm_1 = Norm(d_in)
        self.linear_1 = nn.Linear(d_in, d_mid)
        self.dropout1 = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_mid, max_strlen)
        self.dropout2 = nn.Dropout(dropout)
        self.norm_2 = Norm(max_strlen)
    
    def padding(self, x):
        delta_length = self.cond_dim+self.max_strlen-x.size(2)
        pad = torch.zeros(x.size(0), x.size(1), 
            delta_length, dtype=torch.float32).to(x.get_device())
        return torch.cat([x, pad], axis=2)
        

    def forward(self, x, mconds):
        """
        concatenate x and mconds to get input of dimension
        -> (batch_size, max_strlen+mconds.size(-1), latent_dim)
        N = batch_size
        L = max_strlen+mconds.size(-1)
        Eq = Ek = Ev = latent_dim
        """ 
        print('x size:', x.size())
        x = torch.transpose(x, 1, 2)
        mconds = torch.stack(tuple(mconds for _ in range(x.size(1))),dim=1)
        x = torch.cat([mconds, x], dim=2)
        x = self.padding(x)
        print('mconds size:', mconds.size())
        print('x size2:', x.size())        

        x1 = self.linear_in(x)
        attn_out, attn_out_weights = self.att(x1, x1, x1)
        x1 = x1 + attn_out
        x1 = self.norm_1(x1)

        x = self.dropout1(self.linear_1(x1))
        x = self.dropout2(self.linear_2(x))
        # x = x + x1
        x = self.norm_2(x)
        x = torch.transpose(x, 1, 2)
        return x


class BI_ATTEN(nn.Module):
    def __init__(self, latent_dim, n_heads=2, cond_dim=6, d_in=512,
                 d_mid=256, dropout=0.1, batch_first=True, max_strlen=80):
        super(BI_ATTEN, self).__init__()
        # assert (latent_dim + cond_dim) % n_heads == 0

        self.atten1 = ATTEN1(latent_dim, n_heads, cond_dim, d_in,
                             d_mid, dropout, batch_first)
        self.atten2 = ATTEN2(latent_dim, n_heads, cond_dim, d_in,
                             d_mid, dropout, batch_first, max_strlen)

    def forward(self, x, mconds):
        x1 = self.atten1(x, mconds)
        x2 = self.atten2(x, mconds)
        print('att1 size:', x1.size())
        print('att2 size:', x2.size())
        return x1 + x2


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
    

class ATTENCVAETF(nn.Module):
    def __init__(self, src_vocab, trg_vocab, N=6, d_model=256, dff=2048, h=8, latent_dim=64, 
                 dropout=0.1, nconds=3, use_cond2dec=False, use_cond2lat=False, variational=True, att_type='ATT_v1'):
        super(ATTENCVAETF, self).__init__()
        # settings
        self.nconds = nconds
        self.use_cond2dec = use_cond2dec
        self.use_cond2lat = use_cond2lat

        # model architecture
        self.encoder = Encoder(src_vocab, d_model, N, h, dff, latent_dim,
                               nconds, dropout, variational)
        self.sampler = Sampler(d_model, latent_dim, variational)
        self.att_mu = BI_ATTEN(latent_dim)
        self.att_log_var = BI_ATTEN(latent_dim)

        self.decoder = Decoder(trg_vocab, d_model, N, h, dff, latent_dim,
                               nconds, dropout, use_cond2dec, use_cond2lat)
        self.out = nn.Linear(d_model, trg_vocab)

        self.reset_parameters()

        # other layers
        if self.use_cond2dec == True:
            self.prop_fc = nn.Linear(trg_vocab, 1)

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def encode_att_sample(self, src, conds, src_mask):
        assert len(conds) == 2
        econds, mconds = conds[0], conds[1]
        x = self.encoder(src, econds, src_mask)
        z, mu1, log_var1 = self.sampler(x)
        mu2 = self.att_mu(mu1, mconds)
        log_var2 = self.att_log_var(log_var1, mconds)
        return mu2, mu2, log_var2
        # return self.sampler.sampling(mu2, log_var2), mu2, log_var2, q_k_enc

    def encode_sample(self, src, econds, src_mask):
        x = self.encoder(src, econds, src_mask)
        z, mu, log_var = self.sampler1(x)
        return z, mu, log_var

    def encode(self, src, econds, src_mask):
        x = self.encoder(src, econds, src_mask)
        return x
    
    def decode(self, trg, z, dconds, src_mask, trg_mask):
        x = self.decoder(trg, z, dconds, src_mask, trg_mask)
        return self.out(x)

    def forward(self, src, trg, econds, mconds,
                dconds, src_mask, trg_mask):
        x = self.encode(src, econds, src_mask)
        _, mu, logvar = self.sampler(x)
        mu_pred = self.att_mu(mu, mconds)
        logvar_pred = self.att_log_var(logvar, mconds)
        z, mu, logvar = self.sampler.sampling(mu_pred, logvar_pred)
        output = self.decode(self, trg, z, dconds, src_mask, trg_mask)

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