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
from .modules import Norm, nopeak_mask, create_source_mask, get_clones


# att-v2
class ATT(nn.Module):
    def __init__(self, latent_dim, n_heads=2, cond_dim=6, d_in=512,
                 d_mid=256, dropout=0.1, batch_first=True):
        super(ATT, self).__init__()
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

# att-v1
# class ATT(nn.Module):
#     def __init__(self, latent_dim, n_heads=2, cond_dim=6, d_in=512, d_mid=256,
#                  dropout=0.1, batch_first=True):
#         super(ATT, self).__init__()
#         # assert (latent_dim + cond_dim) % n_heads == 0
#         self.linear_in = nn.Linear(latent_dim+cond_dim, d_in)
#         self.att = nn.MultiheadAttention(embed_dim=d_in,
#                                          num_heads=n_heads,
#                                          dropout=dropout,
#                                          batch_first=batch_first
#                                          )
#         self.linear_1 = nn.Linear(d_in, d_mid)
#         self.dropout = nn.Dropout(dropout)
#         self.linear_2 = nn.Linear(d_mid, latent_dim)
    
#     def forward(self, x, mconds):
#         """
#         concatenate x and mconds to get input of dimension
#         -> (batch_size, max_strlen+mconds.size(-1), latent_dim)
#         N = batch_size
#         L = max_strlen+mconds.size(-1)
#         Eq = Ek = Ev = latent_dim
#         """

#         mconds = torch.stack(tuple(mconds for _ in range(x.size(1))),dim=1)
#         x = torch.cat([mconds, x], dim=2)
#         x1 = self.linear_in(x)
#         # attn_output: (N,L,E). E is embed_dim
#         attn_output, attn_output_weights = self.att(x1, x1, x1)
#         x = self.linear_1(attn_output)
#         x = self.dropout(x)
#         x = self.linear_2(x)
#         return x


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
            x, q_k_enc_tmp = self.layers[i](x, mask)
            q_k_enc_tmp = q_k_enc_tmp.cpu().detach().numpy()[:, np.newaxis, :, :]
            if i == 0:
                q_k_enc = q_k_enc_tmp
            else:
                q_k_enc = np.concatenate((q_k_enc, q_k_enc_tmp), axis=1)

        x = self.norm(x)
        return x, q_k_enc


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
    

class ATTEncoder(nn.Module):
    def __init__(self, src_vocab, trg_vocab, N=6, d_model=256, dff=2048, h=8, latent_dim=64, 
                 dropout=0.1, nconds=3, use_cond2dec=False, use_cond2lat=False, variational=True):
        super(ATTEncoder, self).__init__()
        # settings
        self.nconds = nconds
        self.use_cond2dec = use_cond2dec
        self.use_cond2lat = use_cond2lat

        # model architecture
        self.encoder = Encoder(src_vocab, d_model, N, h, dff, latent_dim,
                               nconds, dropout, variational)
        self.sampler = Sampler(d_model, latent_dim, variational)
        self.att_mu = ATT(latent_dim)
        self.att_log_var = ATT(latent_dim)
        # self.sampler2 = Sampler(d_model, latent_dim, variational)
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
    
    def encode_sample_mlp_sample(self, src, econds, mconds, src_mask):
        x, _ = self.encoder(src, econds, src_mask)
        z1, _, _ = self.sampler1(x)
        x = self.mlp(z1, mconds)
        z2, mu1, log_var2 = self.sampler2(x)
        return z2, mu1, log_var2

    def encode_sample(self, src, econds, src_mask):
        x, _ = self.encoder(src, econds, src_mask)
        z, mu, log_var = self.sampler1(x)
        return z, mu, log_var

    def att_decode(self, trg, e_outputs, conds, src_mask, trg_mask):
        mconds, dconds = conds[0], conds[1]
        self.att_mu(trg)
        x = self.mlp(e_outputs, mconds)
        e_outputs, _, _ = self.sampler2(x)
        return self.out(self.decoder(trg, e_outputs,
                        dconds, src_mask, trg_mask)[0])

    def encode(self, src, econds, src_mask):
        x, _ = self.encoder(src, econds, src_mask)
        return x
    
    def decode(self, trg, e_outputs, dconds, src_mask, trg_mask):
        decoded = self.decoder(trg, e_outputs, dconds, 
                               src_mask, trg_mask)[0]
        return self.out(decoded)

    def forward(self, src, trg_en, econds, mconds, dconds, src_pad_mask, trg_pad_mask):
        x, _ = self.encoder(src, econds, src_pad_mask)
        z, mu1, log_var1 = self.sampler(x)

        mu2 = self.att_mu(mu1, mconds)
        log_var2 = self.att_log_var(log_var1, mconds)
        trg_z_pred = self.sampler.sampling(mu2, log_var2)

        x, _ = self.encoder(trg_en, dconds, trg_pad_mask)
        trg_z_truth, _, _ = self.sampler(x)

        return trg_z_pred, trg_z_truth


def decode(model, src, econds, mconds, dconds, sos_idx, eos_idx,
           pad_idx, max_strlen, decode_type, use_cond2dec=False):
    src_mask = create_source_mask(src, pad_idx, econds)
    z, _, _ = model.encode_sample_mlp_sample(src, econds, mconds, src_mask)

    # initialize the record for break condition. 0 for non-stop, while 1 for stop 
    break_condition = torch.zeros(src.shape[0], dtype=torch.bool)
    
    # create a batch of starting tokens (1)
    ys = (torch.ones(src.shape[0], 1, requires_grad=True)*sos_idx).type_as(src.data)
    
    for i in range(max_strlen-1):
        with torch.no_grad():
            # create a sequence (nopeak) mask for target
            # use_cond2dec should be true s.t. trg_mask considers both the conditions and smiles tokens
            trg_mask = nopeak_mask(ys.size(-1), dconds.size(1), 
                                   use_cond2dec, src.get_device())
            # dim. of output: (bs, ys.size(-1)+1, d_model)
            output = model.decoder(ys, z, dconds, src_mask, trg_mask)[0]
            # dim. of output: (bs, ys.size(-1)+1, vocab_size)
            output = model.out(output)
            # 1.
            output = output[:, -1, :]
            prob = F.softmax(output, dim=-1)
            # 2.
            # output = output[:, -1, :]
            # prob = torch.exp(output)
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