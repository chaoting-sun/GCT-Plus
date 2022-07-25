import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .sublayers import Sampler
from .layers import EncoderLayer, DecoderLayer
from .modules import Embeddings, PositionalEncoding, Norm, create_masks, create_source_mask


def get_clones(layer, N):
    return nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])


class MLP(nn.Module):
    def __init__(self, vocab_size, latent_dim, mcond_dim, d_model):
        super(MLP, self).__init__()
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        mlp_stacks = [latent_dim + mcond_dim, 256, 128, 64, 128, 256, d_model] # 1
        # mlp_stacks = [latent_dim + mcond_dim, 128, 64, 32, 64, 128, d_model] # 2
        # mlp_stacks = [latent_dim + mcond_dim, 256, 128, 256, d_model] # 3
        # mlp_stacks = [latent_dim + mcond_dim, 128, 64, 128, d_model] # 4
        # mlp_stacks = [latent_dim + mcond_dim, 64, 32, 64, d_model] # 5
        # mlp_stacks = [latent_dim + mcond_dim, 32, d_model] # 6
        self.mlp_layers = self.build_mlp(mlp_stacks)
        self.norm = Norm(d_model)

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
    

class MLP_Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, N=6, d_model=256, dff=2048, h=8, latent_dim=64, 
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
        self.reset_parameters()

        # other layers
        if self.use_cond2dec == True:
            self.prop_fc = nn.Linear(trg_vocab, 1)

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

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
        d_output, q_k_dec1, q_k_dec2 = self.decoder(trg, z2, dconds, src_mask, trg_mask)
        output = self.out(d_output)
        
        if self.use_cond2dec == True:
            output_prop = self.prop_fc(output[:, :self.nconds, :])
            output_mol = output[:, self.nconds:, :]
        elif self.use_cond2lat == True:
            output_prop = torch.zeros(output.size(0), self.nconds, 1)
            output_mol = output

        output_mol = F.log_softmax(output_mol, dim=-1)

        return (output_prop, output_mol,
                (mu1, log_var1, z1),
                (mu2, log_var2, z2),
                (q_k_enc, q_k_dec1, q_k_dec2))
        # return output_prop, output_mol, mu, log_var, z, q_k_enc, q_k_dec1, q_k_dec2


class MLP_Encoder(nn.Module):
    def __init__(self, src_vocab, trg_vocab, N=6, d_model=256, dff=2048, h=8, latent_dim=64, 
                 dropout=0.1, nconds=3, use_cond2dec=False, use_cond2lat=False, variational=True):
        super(MLP_Encoder, self).__init__()
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
    
    def encode_mlp(self, src, econds, mconds, src_mask):
        x, _ = self.encoder(src, econds, src_mask)
        z1, _, _ = self.sampler1(x)
        x = self.mlp(z1, mconds)
        z2, _, _ = self.sampler2(x)
        return z2

    def mlp_decode(self, trg, e_outputs, conds, src_mask, trg_mask):
        mconds, dconds = conds[0], conds[1]
        x = self.mlp(e_outputs, mconds)
        e_outputs, _, _ = self.sampler2(x)
        return self.out(self.decoder(trg, e_outputs,
                        dconds, src_mask, trg_mask)[0])

    def decode(self, trg, e_outputs, dconds, src_mask, trg_mask):
        decoded = self.decoder(trg, e_outputs, dconds, 
                               src_mask, trg_mask)[0]
        return self.out(decoded)

    def forward(self, src, trg_en, econds, mconds, dconds):
        src_pad_mask = create_source_mask(src, econds)
        x, _ = self.encoder(src, econds, src_pad_mask)

        z, _, _ = self.sampler1(x)
        x = self.mlp(z, mconds)
        trg_z_pred, _, _ = self.sampler2(x) # output of mlp
        trg_z_pred = F.log_softmax(trg_z_pred, dim=-1)

        trg_pad_mask = create_source_mask(trg_en, dconds)
        x, _ = self.encoder(trg_en, dconds, trg_pad_mask)
        trg_z_truth, _, _ = self.sampler2(x)

        return trg_z_pred, trg_z_truth


def transfer_parameters(transformer, mlptransformer):
    mlptf_dict = mlptransformer.state_dict()
    for name, param in transformer.named_parameters():
        if name in mlptf_dict:
            mlptf_dict[name].copy_(param)
        else:
            print('Layer Not Found:', name)
            exit()


def freeze_parameters(mlptransformer):
    for name, param in mlptransformer.named_parameters():
        name_split = name.split('.')
        if name_split[0] == 'mlp':
            continue
        param.requires_grad = False
