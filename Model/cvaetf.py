import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


from .layers import EncoderLayer, DecoderLayer
from .modules import Embeddings, PositionalEncoding
from .modules import Norm, create_masks, get_clones


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

        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return self.sampling(mu, log_var), mu, log_var

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
        # e_outputs: (bs, nc+src_len, lat_dim)
        # src_mask: (bs, 1, nc+src_len)
        # trg_mask: (bs, trg_len-1, trg_len-1)
        x = self.embed(trg)
        # x: (bs, trg_len-1, d_model)        
        e_outputs = self.fc_z(e_outputs)
        # e_outputs: (bs, nc+src_len, d_model)

        if self.use_cond2dec == True:
            cond2dec = self.embed_cond2dec(cond_input).view(cond_input.size(0), cond_input.size(1), -1)
            x = torch.cat([cond2dec, x], dim=1)
            # x: (bs, (nc+src_len)+(trg_len-1), d_model)
        elif self.use_cond2lat == True:
            cond2lat = self.embed_cond2lat(cond_input).view(cond_input.size(0), cond_input.size(1), -1)
            # cond2lat: (bs, nc, d_model)
            e_outputs = torch.cat([cond2lat, e_outputs], dim=1)
            # e_outputs: (bs, nc+(nc+src_len), d_model)
        x = self.pe(x)

        if self.use_cond2lat == True:
            cond_mask = torch.unsqueeze(cond_input, -2)
            cond_mask = torch.ones_like(cond_mask, dtype=bool)
            src_mask = torch.cat([cond_mask, src_mask], dim=2)
            # src_mask: (bs, 1, nc+(nc+src_len))

        for i in range(self.N):
            x = self.layers[i](x, e_outputs, cond_input, src_mask, trg_mask)
        return self.norm(x)

class Cvaetf(nn.Module):
    def __init__(self, src_vocab, trg_vocab, N=6, d_model=256, dff=2048, h=8, latent_dim=64, 
                 dropout=0.1, nconds=3, use_cond2dec=False, use_cond2lat=False, variational=True):
        super(Cvaetf, self).__init__()
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
        self.out = nn.Linear(d_model, trg_vocab)

        # initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, conds, src_mask):
        z, mu, log_var = self.encoder(src, conds, src_mask)
        return z, mu, log_var

    def decode(self, trg, z, conds, src_mask, trg_mask):
        x = self.decoder(trg, z, conds, src_mask, trg_mask)
        return self.out(x)

    def forward(self, src, trg, econds, dconds, src_mask, trg_mask):
        z, mu, log_var = self.encoder(src, econds, src_mask)
        d_output = self.decoder(trg, z, dconds, src_mask, trg_mask)
        output = self.out(d_output)

        if self.use_cond2dec == True:
            output_prop = self.prop_fc(output[:, :self.nconds, :])
            output_mol = output[:, self.nconds:, :]
        else:
            output_prop = torch.zeros(output.size(0), self.nconds, 1)
            output_mol = output
        return output_prop, output_mol, mu, log_var, z


def collate_fcn(ins, SRC, TRG, device):
    outs = {}
    
    src = ['src', 'src_scaffold']
    trg = ['trg', 'trg_scaffold']
    props = ['econds', 'dconds', 'mconds']

    src_ids = SRC.process([e['src'] for e in ins])
    src_scafold_ids = SRC.process([e['src_scafold'] for e in ins])
    # if not SRC.batch_first:
        


    for s in src:
        if s in ins[0]:
            outs[s] = SRC.process([e[s] for e in ins]).to(device)
            if not SRC.batch_first:
                outs[s] = outs[s].T

    for t in trg:
        if t in ins[0]:
            outs[t] = TRG.process([e[t] for e in ins]).to(device)
            if not TRG.batch_first:
                outs[t] = outs[t].T

    # if 'src' in outs and 'src_scaffold' in outs:
    #     outs['src'] = 
    
    for p in props:
        if p in ins[0]:
            outs[p] = torch.tensor([e[p] for e in ins],
                dtype=torch.float32).to(device)




    
    # if 'src' in raw_batch[0]:
    #     batch['src'] = SRC.process([+b['src'] for b in raw_batch]).to(device)
    #     if not SRC.batch_first:
    #         batch['src'] = batch['src'].T

    # if 'src' in raw_batch[0]:
    #     batch['src'] = SRC.process([+b['src'] for b in raw_batch]).to(device)
    #     if not SRC.batch_first:
    #         batch['src'] = batch['src'].T


    # if 'src_scaffold' in raw_batch[0]:
    #     src = [b['src_scaffold']+b['src'] for b in raw_batch]            
    # elif 'src' in raw_batch[0]:
    #     src = [b['src'] for b in raw_batch]

    # if 'trg_scaffold' in raw_batch[0]:
    #     trg = [b['trg_scaffold']+b['trg'] for b in raw_batch]     
    # elif 'trg' in raw_batch[0]:
    #     trg = [b['trg'] for b in raw_batch]

    # print('src:', SRC.vocab.stoi)
    # print('trg:', TRG.vocab.stoi)

    # if src is not None:
    #     batch['src'] = SRC.process(src).to(device)
    #     if not SRC.batch_first:
    #         batch['src'] = batch['src'].T
    #     print(batch['src'])
    # if trg is not None:
    #     batch['trg'] = TRG.process(trg).to(device)
    #     if not TRG.batch_first:
    #         batch['trg'] = batch['trg'].T        
    #     print(batch['trg'])
    # exit()

    # for prop in ('econds', 'dconds', 'mconds'):
    #     if prop in raw_batch[0]:
    #         batch[prop] = torch.tensor(
    #             [b[prop] for b in raw_batch],
    #             dtype=torch.float32).to(device)
    # return batch
