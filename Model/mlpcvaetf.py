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
from .modules import Norm, nopeak_mask, create_masks, create_source_mask, get_clones


class MLP(nn.Module):
    def __init__(self, vocab_size, latent_dim, mcond_dim, d_model):
        super(MLP, self).__init__()
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        # TODO: change the layer settings
        mlp_stacks = [latent_dim + mcond_dim, 256, 128, 64, 128, 256, d_model] # 1
        # mlp_stacks = [latent_dim + mcond_dim, 256, 128, 64, 128, 256, d_model] # 1
        # mlp_stacks = [latent_dim + mcond_dim, 256, 128, 64, 128, 256, d_model] # 1
        # mlp_stacks = [latent_dim + mcond_dim, 128, 64, 32, 64, 128, d_model] # 2
        # mlp_stacks = [latent_dim + mcond_dim, 256, 128, 256, d_model] # 3
        # mlp_stacks = [latent_dim + mcond_dim, 128, 64, 128, d_model] # 4
        # mlp_stacks = [latent_dim + mcond_dim, 64, 32, 64, d_model] # 5
        # mlp_stacks = [latent_dim + mcond_dim, 32, d_model] # 6
        self.mlp_layers = self.build_mlp(mlp_stacks)
        self.dropout = nn.Dropout(0.1)
        self.actFcn = nn.ReLU()
        self.norm = Norm(d_model)

    def build_mlp(self, stacks):
        layers = []        
        for i in range(len(stacks) - 1):
            layers.extend([nn.Linear(stacks[i], stacks[i + 1]), self.dropout, self.actFcn])
        return nn.ModuleList(layers)

    def forward(self, x, mconds):
        mconds = torch.stack(tuple(mconds for _ in range(x.size(1))),dim=1)
        x = torch.cat([x, mconds], dim=2)

        for i in range(len(self.mlp_layers)):
            x = self.mlp_layers[i](x)
        return self.norm(x) # should add to avoid inf


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
    

class MLPCVAETF(nn.Module):
    def __init__(self, src_vocab, trg_vocab, N=6, d_model=256, dff=2048, h=8, latent_dim=64, 
                 dropout=0.1, nconds=3, use_cond2dec=False, use_cond2lat=False, variational=True):
        super(MLPCVAETF, self).__init__()
        # settings
        self.nconds = nconds
        self.use_cond2dec = use_cond2dec
        self.use_cond2lat = use_cond2lat

        # model architecture
        self.encoder = Encoder(src_vocab, d_model, N, h, dff, latent_dim,
                               nconds, dropout, variational)
        self.sampler1 = Sampler(d_model, latent_dim, variational)
        self.mlp = MLP(src_vocab, latent_dim, 2*nconds, d_model)
        self.sampler2 = Sampler(d_model, latent_dim, variational)
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

    # def encoder_mlp(self, src, econds, mconds, src_mask):
    #     x, _ = self.encoder(src, econds, src_mask)
    #     z1, _, _ = self.sampler1(x)
    #     x = self.mlp(z1, mconds)
    #     z2, _, _ = self.sampler2(x)
    #     return z2
    
    def encode(self, src, conds, src_mask):
        econds, mconds = conds
        x = self.encoder(src, econds, src_mask)
        z, _, _ = self.sampler1(x)
        x = self.mlp(z, mconds)
        z, mu, log_var = self.sampler2(x)        
        return z, mu, log_var

    def decode(self, trg, z, dconds, src_mask, trg_mask):
        x = self.decoder(trg, z, dconds, src_mask, trg_mask)
        return self.out(x)

    def forward(self, src, trg, econds, mconds, 
                dconds, src_mask, trg_mask):
        # src_mask, trg_mask = create_masks(src, trg, econds, self.use_cond2dec)
        x = self.encode(src, econds, src_mask)
        z1, mu, logvar = self.sampler1(x) # (batch_size, max_source_len, latent_dim)
        x = self.mlp(z1, mconds)
        z2, mu, logvar = self.sampler2(x)
        output = self.decode(trg, z2, dconds, src_mask, trg_mask)
        
        if self.use_cond2dec == True:
            output_prop = self.prop_fc(output[:, :self.nconds, :])
            output_mol = output[:, self.nconds:, :]
        elif self.use_cond2lat == True:
            output_prop = torch.zeros(output.size(0), self.nconds, 1)
            output_mol = output

        output_mol = F.log_softmax(output_mol, dim=-1)
        return output_prop, output_mol, mu, logvar, z2


def decode(model, src, econds, mconds, dconds, sos_idx, eos_idx,
           pad_idx, max_strlen, decode_type, use_cond2dec=False):
    src_mask = create_source_mask(src, pad_idx, econds)
    z = model.encode_mlp(src, econds, mconds, src_mask)

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