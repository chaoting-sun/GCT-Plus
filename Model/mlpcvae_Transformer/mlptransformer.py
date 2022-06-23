import torch
import torch.nn as nn
import torch.nn.functional as F

from .sublayers import Norm
from .layers import EncoderLayer, DecoderLayer
from .embeddings import Embeddings, PositionalEncoding
from .mask import create_masks


import copy
import numpy as np
from collections import OrderedDict


def get_clones(layer, N):
    return nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):

        return F.log_softmax(self.proj(x), dim=-1)


class MLP(nn.Module):
    def __init__(self, vocab_size, latent_dim, mcond_dim, d_model):
        super(MLP, self).__init__()
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        
        mlp_stacks = [latent_dim + mcond_dim, 64, 32, 64, d_model]
        self.mlp_layers = self.build_mlp(mlp_stacks)

    def build_mlp(self, stacks):
        layers = []        
        for i in range(len(stacks) - 1):
            layers.extend([nn.Linear(stacks[i], stacks[i + 1]), nn.PReLU()])
        return nn.ModuleList(layers)

    def forward(self, x, mconds):
        mconds = torch.stack(tuple(mconds for _ in range(x.size(1))),dim=1)
        x = torch.cat([x, mconds], dim=2)
        for i in range(len(self.mlp_layers)):
            x = self.mlp_layers[i](x)
        return x


class Sampler(nn.Module):
    def __init__(self, d_model, latent_dim, variational):
        super(Sampler, self).__init__()
        self.variational = variational
        self.fc_mu = nn.Linear(d_model, latent_dim)
        self.fc_log_var = nn.Linear(d_model, latent_dim)

    def sampling(self, mu, log_var):
        if self.variational:
            std = torch.exp(0.5*log_var)
            eps = torch.randn_like(std)
            return eps.mul(std).add(mu)
        else:
            return mu

    def forward(self, x):
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return self.sampling(mu, log_var), mu, log_var


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
        return x, q_k_enc
        # mu = self.fc_mu(x) # d_model -> opt.latent_dim
        # log_var = self.fc_log_var(x) # d_model -> opt.latent_dim
        # return self.sampling(mu, log_var), mu, log_var, q_k_enc

    # def sampling(self, mu, log_var):
    #     if self.variational:
    #         std = torch.exp(0.5*log_var)
    #         eps = torch.randn_like(std)
    #         return eps.mul(std).add(mu)
    #     else:
    #         return mu


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
        x = self.embed(trg)

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
    

class MLP_Transformer(nn.Module):
    def __init__(self, transformer, src_vocab, trg_vocab, N=6, d_model=256, dff=2048, h=8, latent_dim=64, 
                 dropout=0.1, nconds=3, use_cond2dec=False, use_cond2lat=False, variational=True):
        super(MLP_Transformer, self).__init__()
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

        self._reset_parameters()
        self._freeze_parameters()
        self._transfer_parameters(transformer)

        # other layers
        if self.use_cond2dec == True:
            self.prop_fc = nn.Linear(trg_vocab, 1)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _freeze_parameters(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.sampler1.parameters():
            param.requires_grad = False
        for param in self.out.parameters():
            param.requires_grad = False
        for param in self.sampler2.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False

    def _transfer_parameters(self, transformer):
        for name, param in transformer.named_parameters():
            name_split = name.split('.')
            sub_name = '.'.join(name_split[1:])

            if name_split[-2] in ('mu', 'log_var'):
                self.sampler1.state_dict()[sub_name].copy_(param)
                self.sampler2.state_dict()[sub_name].copy_(param)
            elif name_split[0] == 'out':
                self.out.state_dict()[sub_name] = param
            elif name_split[0] == 'encoder':
                self.encoder.state_dict()[sub_name].copy_(param)
            elif name_split[0] == 'decoder':
                self.decoder.state_dict()[sub_name].copy_(param)
            else:
                exit("There may be something missing.")

    def encoder_mlp(self, src, econds, mconds, src_mask):
        x, _ = self.encoder(src, econds, src_mask)
        z1, _, _ = self.sampler1(x)
        x = self.mlp(z1, mconds)
        z2, _, _ = self.sampler2(x)
        return z2

    def forward(self, src, trg, econds, mconds, dconds):
        src_mask, trg_mask = create_masks(src, trg, econds, self.use_cond2dec)
        x, q_k_enc = self.encoder(src, econds, src_mask)
        z1, mu1, log_var1 = self.sampler1(x) # (batch_size, max_source_len, latent_dim)
        x = self.mlp(z1, mconds)
        z2, mu2, log_var2 = self.sampler2(x)

        print(">>", trg.size(), z1.size(), dconds.size(), src_mask.size(), trg_mask.size())
        d_output, q_k_dec1, q_k_dec2 = self.decoder(trg, z2, dconds, src_mask, trg_mask)
        
        output = self.out(d_output)
        
        if self.use_cond2dec == True:
            output_prop = self.prop_fc(output[:, :self.nconds, :])
            output_mol = output[:, self.nconds:, :]
        elif self.use_cond2lat == True:
            output_prop = torch.zeros(output.size(0), self.nconds, 1)
            output_mol = output
        return (output_prop, output_mol,
                (mu1, log_var1, z1),
                (mu2, log_var2, z2),
                (q_k_enc, q_k_dec1, q_k_dec2))
        # return output_prop, output_mol, mu, log_var, z, q_k_enc, q_k_dec1, q_k_dec2