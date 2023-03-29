import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


from .layers import EncoderLayer, DecoderLayer
from .sublayers import Sampler
from .modules import Embeddings, PositionalEncoding, Norm, create_masks, get_clones


class Encoder(nn.Module):
    "Pass N encoder layers, followed by a layernorm"
    def __init__(self, vocab_size, d_model, N, h, dff, latent_dim,
                 nconds, dropout, variational=True):
        super(Encoder, self).__init__()
        self.N = N
        self.variational = variational
        # embedding layers
        self.src_embed = Embeddings(d_model, vocab_size)
        self.embed_cond2enc = nn.Linear(nconds, d_model*nconds)
        # other layers
        self.norm = Norm(d_model)
        self.pe = PositionalEncoding(d_model, dropout=dropout)
        self.layers = get_clones(EncoderLayer(h, d_model, dff, dropout), N)

    def forward(self, src, sca_scaffold, econds, mask):
        cond2enc = self.embed_cond2enc(econds).view(econds.size(0), econds.size(1), -1)
        x_src = self.src_embed(src)
        x_sca = self.src_embed(sca_scaffold)
        
        x = torch.cat([cond2enc, x_sca, x_src], dim=1)
        # input: p1, p2, ..., pn + sca + smi
        
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        x = self.norm(x)
        return x


class Decoder(nn.Module):
    "Pass N decoder layers, followed by a layernorm"
    def __init__(self, vocab_size, d_model, N, h,
                 dff, latent_dim, nconds, dropout,
                 use_cond2dec, use_cond2lat):
        super(Decoder, self).__init__()
        self.N = N
        self.d_model = d_model
        self.use_cond2dec = use_cond2dec
        self.use_cond2lat = use_cond2lat
        # embedding layers
        self.trg_embed = Embeddings(d_model, vocab_size)
        if self.use_cond2dec == True:
            self.embed_cond2dec = nn.Linear(nconds, d_model*nconds) #concat to trg_input
        if self.use_cond2lat == True:
            self.embed_cond2lat = nn.Linear(nconds, d_model*nconds) #concat to trg_input
        # other layers
        self.pe = PositionalEncoding(d_model, dropout=dropout)
        self.fc_z = nn.Linear(latent_dim, d_model)
        self.layers = get_clones(DecoderLayer(h, d_model, dff, dropout, use_cond2dec, use_cond2lat), N)
        self.norm = Norm(d_model)

    def forward(self, trg, trg_scaffold, e_outputs, cond_input,
                src_enc_mask, src_dec_mask, trg_mask):        
        x = self.trg_embed(trg)
        x_sca = self.trg_embed(trg_scaffold)
        e_outputs = self.fc_z(e_outputs)
        # size: p1, p2, p3 + sca + src_sca
        
        if self.use_cond2dec == True:
            cond2dec = self.embed_cond2dec(cond_input).view(cond_input.size(0), cond_input.size(1), -1)
            x = torch.cat([cond2dec, x_sca, x], dim=1) # trg + cond
        elif self.use_cond2lat == True:
            cond2lat = self.embed_cond2lat(cond_input).view(cond_input.size(0), cond_input.size(1), -1)
            e_outputs = torch.cat([cond2lat, x_sca, e_outputs], dim=1)
            # size: p1, p2, ..., p3 + <sos>sca<eos> + z

        x = self.pe(x)
        
        if self.use_cond2lat == True:
            src_mask = torch.cat([src_dec_mask, src_enc_mask], dim=2)
            # src_mask: (bs, 1, nc+sca+(nc+sca+src_len))
        
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, cond_input, src_mask, trg_mask)
        return self.norm(x)


class ScaCvaetfV2(nn.Module):
    def __init__(self, src_vocab, trg_vocab, N=6, d_model=256, dff=2048, h=8,
                 latent_dim=64,  dropout=0.1, nconds=3, use_cond2dec=False,
                 use_cond2lat=False, variational=True):
        super(ScaCvaetfV2, self).__init__()
        # settings
        self.nconds = nconds
        self.use_cond2dec = use_cond2dec
        self.use_cond2lat = use_cond2lat
        
        # encoder/decoder
        self.encoder = Encoder(src_vocab, d_model, N, h, dff, latent_dim,
                               nconds, dropout, variational)
        self.decoder = Decoder(trg_vocab, d_model, N, h, dff, latent_dim,
                               nconds, dropout, use_cond2dec, use_cond2lat)
        self.sampler = Sampler(d_model, latent_dim, variational)
        
        # other layers
        if self.use_cond2dec == True:
            self.prop_fc = nn.Linear(trg_vocab, 1)
        self.out = nn.Linear(d_model, trg_vocab)

        # initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_scaffold, conds, src_mask):
        x = self.encoder(src, src_scaffold, conds, src_mask)
        z, mu, log_var = self.sampler(x)
        return z, mu, log_var

    def decode(self, trg, trg_scaffold, z, conds,
               src_enc_mask, src_dec_mask, trg_mask):
        x = self.decoder(trg, trg_scaffold, z, conds,
                         src_enc_mask, src_dec_mask, trg_mask)
        return self.out(x)

    def forward(self, src, trg, src_scaffold, trg_scaffold,
                econds, dconds, src_enc_mask, src_dec_mask,
                trg_mask):
        z, mu, log_var = self.encode(src, src_scaffold,
                                     econds, src_enc_mask)
        output = self.decode(trg, trg_scaffold, z, dconds,
                             src_enc_mask, src_dec_mask,
                             trg_mask)

        if self.use_cond2dec == True:
            output_prop = self.prop_fc(output[:, :self.nconds, :])
            output_mol = output[:, self.nconds:, :]
        else:
            output_prop = torch.zeros(output.size(0), self.nconds, 1)
            output_mol = output
        return output_prop, output_mol, mu, log_var, z
    
    
# def collate_fcn(ins, SRC, TRG, device):
#     outs = {}

#     src = ['src', 'src_scaffold']
#     trg = ['trg', 'trg_scaffold']
#     props = ['econds', 'dconds', 'mconds']
    
#     for s in src:
#         if s in ins[0]:
#             outs[s] = SRC.process([e[s] for e in ins]).to(device)
#             if not SRC.batch_first:
#                 outs[s] = outs[s].T
#             # List[int]: [t1, ..., tn, <pad>, ...]
    
#     for t in trg:
#         if t in ins[0]:
#             outs[t] = TRG.process([e[t] for e in ins]).to(device)
#             if not TRG.batch_first:
#                 outs[t] = outs[t].T
#             # List[int]: [<sos>, t1, ..., tn, <eos>, <pad>, ...]
        
#     for p in props:
#         if p in ins[0]:
#             outs[p] = torch.tensor([e[p] for e in ins],
#                 dtype=torch.float32).to(device)
#     return outs