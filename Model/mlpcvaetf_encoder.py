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
from .modules import Norm, create_target_mask, create_source_mask, get_clones


class MLP(nn.Module):
    def __init__(self, vocab_size, latent_dim, mcond_dim, d_model, dropout=0.1):
        super(MLP, self).__init__()
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        mlp_stacks = [latent_dim + mcond_dim, 512,
                      256, 128, 64, 128, 256, 512, d_model]
        # mlp_stacks = [latent_dim + mcond_dim, 256, 128, 64, 128, 256, d_model] # 1
        # mlp_stacks = [latent_dim + mcond_dim, 128, 64, 32, 64, 128, d_model] # 2
        # mlp_stacks = [latent_dim + mcond_dim, 256, 128, 256, d_model] # 3
        # mlp_stacks = [latent_dim + mcond_dim, 128, 64, 128, d_model] # 4
        # mlp_stacks = [latent_dim + mcond_dim, 64, 32, 64, d_model] # 5
        # mlp_stacks = [latent_dim + mcond_dim, 32, d_model] # 6
        self.mlp_layers = self.build_mlp(mlp_stacks)
        self.dropout = nn.Dropout(p=dropout)
        self.norm = Norm(d_model)

    def build_mlp(self, stacks):
        layers = []
        for i in range(len(stacks) - 1):
            layers.extend([nn.Linear(stacks[i], stacks[i + 1]), nn.ReLU()])
        return nn.ModuleList(layers)

    def forward(self, x, mconds):
        mconds = torch.stack(tuple(mconds for _ in range(x.size(1))), dim=1)
        x = torch.cat([x, mconds], dim=2)

        for i in range(len(self.mlp_layers)):
            x = self.dropout(self.mlp_layers[i](x))

        return self.norm(x)  # should add to avoid inf


class Encoder(nn.Module):
    "Pass N encoder layers, followed by a layernorm"

    def __init__(self, vocab_size, d_model, N, h, dff, latent_dim, nconds, dropout, variational=True):
        super(Encoder, self).__init__()
        self.N = N
        self.variational = variational
        # input embedding layers
        self.embed_sentence = Embeddings(d_model, vocab_size)
        # nn.Linear() supports TensorFloat32
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
            self.embed_cond2dec = nn.Linear(
                nconds, d_model*nconds)  # concat to trg_input
        elif self.use_cond2lat == True:
            self.embed_cond2lat = nn.Linear(
                nconds, d_model*nconds)  # concat to trg_input
        self.pe = PositionalEncoding(d_model, dropout=dropout)
        self.fc_z = nn.Linear(latent_dim, d_model)
        self.layers = get_clones(DecoderLayer(
            h, d_model, dff, dropout, use_cond2dec, use_cond2lat), N)
        self.norm = Norm(d_model)

    def forward(self, trg, e_outputs, cond_input, src_mask, trg_mask):
        # dim: -> (batch_size, trg_maxstr-1, d_model)
        x = self.embed(trg)

        # dim: -> (batch_size, trg_maxstr, d_model)
        e_outputs = self.fc_z(e_outputs)

        if self.use_cond2dec == True:
            cond2dec = self.embed_cond2dec(cond_input).view(
                cond_input.size(0), cond_input.size(1), -1)
            x = torch.cat([cond2dec, x], dim=1)  # trg + cond
        elif self.use_cond2lat == True:
            # dim: -> (batch_size, nconds, d_model)
            cond2lat = self.embed_cond2lat(cond_input).view(
                cond_input.size(0), cond_input.size(1), -1)
            # dim: -> (batch_size, nconds+maxstr, d_model)
            e_outputs = torch.cat([cond2lat, e_outputs], dim=1)  # cond + lat

        x = self.pe(x)

        for i in range(self.N):
            x, q_k_dec1_tmp, q_k_dec2_tmp = self.layers[i](
                x, e_outputs, cond_input, src_mask, trg_mask)
            q_k_dec1_tmp = q_k_dec1_tmp.cpu().detach().numpy()[
                :, np.newaxis, :, :]
            q_k_dec2_tmp = q_k_dec2_tmp.cpu().detach().numpy()[
                :, np.newaxis, :, :]
            if i != 0:
                q_k_dec1 = np.concatenate((q_k_dec1, q_k_dec1_tmp), axis=1)
                q_k_dec2 = np.concatenate((q_k_dec2, q_k_dec2_tmp), axis=1)
            else:
                q_k_dec1 = q_k_dec1_tmp
                q_k_dec2 = q_k_dec2_tmp
        return self.norm(x), q_k_dec1, q_k_dec2


class MLPCVAETFEncoder(nn.Module):
    def __init__(self, src_vocab, trg_vocab, N=6, d_model=256, dff=2048, h=8, latent_dim=64,
                 dropout=0.1, nconds=3, use_cond2dec=False, use_cond2lat=False, variational=True):
        super(MLPCVAETFEncoder, self).__init__()
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

    def encode(self, src, econds, src_mask):
        x, q_k_enc = self.encoder(src, econds, src_mask)
        return x, q_k_enc

    # genz_type (I think it's better)
    def encode_sample(self, src, econds, src_mask):
        x, q_k_enc = self.encoder(src, econds, src_mask)
        z, mu, logvar = self.sampler1(x)
        return z, mu, logvar, q_k_enc

    # # genz_type
    # def encode_sample_mlp_sample(self, src, econds, mconds, src_mask):
    #     x, _ = self.encoder(src, econds, src_mask)
    #     z1, _, _ = self.sampler1(x)
    #     x = self.mlp(z1, mconds)
    #     z2, mu1, log_var2 = self.sampler2(x)
    #     return z2, mu1, log_var2


    # # decode_type
    # def decode(self, trg, e_outputs, dconds, src_mask, trg_mask):
    #     decoded = self.decoder(trg, e_outputs, dconds,
    #                            src_mask, trg_mask)[0]
    #     return self.out(decoded)

    # decode_type
    def mlp_decode(self, trg, z1, conds, src_mask, trg_mask):
        assert len(conds) == 2
        mconds, dconds = conds[0], conds[1]
        x = self.mlp(z1, mconds)
        z2, _, _ = self.sampler2(x)
        x, q_k_dec1, q_k_dec2 = self.decoder(trg, z2, dconds, src_mask, trg_mask)
        return self.out(x), q_k_dec1, q_k_dec2

    def forward(self, src, trg_en, econds, mconds, dconds, src_pad_mask, trg_pad_mask):
        x = self.encoder(src, econds, src_pad_mask)

        z, _, _ = self.sampler1(x)
        x = self.mlp(z, mconds)
        trg_z_pred, _, _ = self.sampler2(x)  # output of mlp
        trg_z_pred = F.log_softmax(trg_z_pred, dim=-1)

        x = self.encoder(trg_en, dconds, trg_pad_mask)
        trg_z_truth, _, _ = self.sampler2(x)

        return trg_z_pred, trg_z_truth


def decode(z_sampler, smi_predictor, src, econds, mconds,
           dconds, sos_idx, eos_idx, pad_idx, max_strlen,
           decode_algo, use_cond2dec=False):
           
    src_mask = create_source_mask(src, pad_idx, econds)
    # z, _, _ = model.encode_sample_mlp_sample()
    z, _, _ = z_sampler(src, econds, mconds, src_mask)

    # initialize the record for break condition. 0 for non-stop, while 1 for stop
    break_condition = torch.zeros(src.shape[0], dtype=torch.bool)

    # create a batch of starting tokens (1)
    ys = (torch.ones(src.shape[0], 1,
          requires_grad=True)*sos_idx).type_as(src.data)

    with torch.no_grad():
        for i in range(max_strlen-1):
            # create a padding/nopeaking mask of target
            trg_mask = create_target_mask(
                ys.size(-1), pad_idx, dconds.size(1), use_cond2dec)

            # predict next token given current sequence with normalized probabilities
            # (bs, len(ys[0])+1, tar_vocab)
            prob = smi_predictor(ys, z, dconds, src_mask, trg_mask)

            # take the probability distribution of the next token
            prob = prob[:, -1, :]

            if decode_algo == 'greedy':
                _, next_word = torch.max(prob, dim=1)
                ys = torch.cat([ys, next_word.unsqueeze(-1)],
                               dim=1)  # [batch_size, i]
            elif decode_algo == 'multinomial':
                next_word = torch.multinomial(
                    prob, 1)  # shape: (batch_size, 1)
                ys = torch.cat([ys, next_word], dim=1)  # [batch_size, i]
                next_word = torch.squeeze(next_word)  # shape: (batch_size)

            # update the break condition. 2 is the stop token
            break_condition = (break_condition | (
                next_word.to('cpu') == eos_idx))
            # If all satisfies the break condition, then break the loop.
            if all(break_condition):
                break
    return ys


# def decode(model, src, econds, mconds, dconds, sos_idx, eos_idx,
#            pad_idx, max_strlen, decode_algo, use_cond2dec=False):
#     src_mask = create_source_mask(src, pad_idx, econds)
#     z, _, _ = model.encode_sample_mlp_sample(src, econds, mconds, src_mask)

#     # initialize the record for break condition. 0 for non-stop, while 1 for stop
#     break_condition = torch.zeros(src.shape[0], dtype=torch.bool)

#     # create a batch of starting tokens (1)
#     ys = (torch.ones(src.shape[0], 1, requires_grad=True)*sos_idx).type_as(src.data)

#     with torch.no_grad():
#         for i in range(max_strlen-1):
#             # create a padding/nopeaking mask of target
#             trg_mask = create_target_mask(ys.size(-1), pad_idx, dconds.size(1), use_cond2dec)

#             # predict given current sequence and latent space
#             output = model.decode(ys, z, dconds, src_mask, trg_mask) # (bs, len(ys[0])+1, tar_vocab)

#             # take the probability distribution of the next token
#             output = output[:, -1, :]

#             # normalize the distribution
#             prob = F.softmax(output, dim=-1)

#             if decode_algo == 'greedy':
#                 _, next_word = torch.max(prob, dim=1)
#                 ys = torch.cat([ys, next_word.unsqueeze(-1)], dim=1)  # [batch_size, i]
#             elif decode_algo == 'multinomial':
#                 next_word = torch.multinomial(prob, 1) # shape: (batch_size, 1)
#                 ys = torch.cat([ys, next_word], dim=1) #[batch_size, i]
#                 next_word = torch.squeeze(next_word) # shape: (batch_size)

#             # update the break condition. 2 is the stop token
#             break_condition = (break_condition | (next_word.to('cpu')==eos_idx))
#             # If all satisfies the break condition, then break the loop.
#             if all(break_condition):
#                 break

#     return ys