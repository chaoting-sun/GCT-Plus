"""
- Training
  - Phase I : Model -> Encoder + Sampler + Decoder
  - Phase II: Model -> Encoder + Sampler + MLP + Sampler + Decoder
- Inference
  - Method0: Decode -> Sampler + Decoder (Phase I)
  - Method1: Decode -> Encoder + Sampler + MLP + Sampler + Decoder
  - Method2: Decode -> MLP + Sampler + Decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from Model import (
    Sampler,
    EncoderLayer,
    DecoderLayer,
    Embeddings,
    PositionalEncoding,
    Norm,
    get_clones
) 


class BLOCK(nn.Module):
    def __init__(self, z_dim):
        super(BLOCK, self).__init__()
        self.ln1 = nn.LayerNorm(z_dim)
        self.ln2 = nn.LayerNorm(z_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=z_dim,
            num_heads=2,
            dropout=0.1,
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


class ROTATOR(nn.Module):
    def __init__(self, latent_dim, mcond_dim, n_layer=3):
        super(ROTATOR, self).__init__()
        self.block1 = nn.Sequential(*[BLOCK(2*latent_dim) for _ in range(n_layer)])
        self.block2 = nn.Sequential(*[BLOCK(latent_dim) for _ in range(n_layer)])

        self.prop_linear = nn.Linear(mcond_dim, latent_dim)
        self.rot_linear = nn.Linear(2*latent_dim, latent_dim)

        self.prop_ln = nn.LayerNorm(latent_dim)
        self.rot_ln = nn.LayerNorm(2*latent_dim)
        self.out_ln = nn.LayerNorm(latent_dim)
    
    def forward(self, x, mconds):
        props = self.prop_ln(self.prop_linear(mconds))
        props = torch.stack(tuple(props for _ in range(x.size(1))),dim=1)

        x = torch.cat([props, x], dim=2)

        for layer in self.block1:
            x, attn = layer(x)
        
        x = self.rot_linear(self.rot_ln(x))

        for layer in self.block2:
            x, attn = layer(x)
        return x


class Encoder(nn.Module):
    "Pass N encoder layers, followed by a layernorm"
    def __init__(self, vocab_size, d_model, N, h, dff, latent_dim, nconds, dropout, variational=True):
        super(Encoder, self).__init__()
        self.N = N
        self.variational = variational
        # input embedding layers
        self.embed_sentence = Embeddings(d_model, vocab_size)
        self.embed_cond2enc = nn.Linear(nconds, d_model*nconds)
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
    

class SEPCVAETF(nn.Module):
    def __init__(self, src_vocab, trg_vocab, N=6, d_model=256, dff=2048, h=8, latent_dim=64, 
                 dropout=0.1, nconds=3, use_cond2dec=False, use_cond2lat=False, variational=True):
        super(SEPCVAETF, self).__init__()
        # settings
        self.nconds = nconds
        self.use_cond2dec = use_cond2dec
        self.use_cond2lat = use_cond2lat

        # model architecture
        self.encoder = Encoder(src_vocab, d_model, N, h, dff, latent_dim,
                               nconds, dropout, variational)
        self.sampler = Sampler(d_model, latent_dim, variational)
        self.mu_rotator = ROTATOR(latent_dim, 2*nconds)
        self.logvar_rotator = ROTATOR(latent_dim, 2*nconds)
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
    
    def encode(self, src, conds, src_mask):
        econds, mconds = conds
        x = self.encoder(src, econds, src_mask)
        _, mu, logvar = self.sampler(x)
        mu = self.mu_rotator(mu, mconds)
        logvar = self.logvar_rotator(logvar, mconds)
        z = self.sampler.sampling(mu, logvar)
        return z, mu, logvar

    def decode(self, trg, z, dconds, src_mask, trg_mask):
        x = self.decoder(trg, z, dconds, src_mask, trg_mask)
        return self.out(x)

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

        # output_mol = F.log_softmax(output_mol, dim=-1) # ??? comment and test this -> 可加可不加
        return output_prop, output_mol, mu, logvar, z


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