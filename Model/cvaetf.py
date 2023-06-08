import torch
import torch.nn as nn
from Model import (
    Sampler,
    EncoderLayer,
    DecoderLayer,
    Embeddings,
    PositionalEncoding,
    Norm,
    get_clones
) 


class Encoder(nn.Module):
    "Pass N encoder layers, followed by a layernorm"
    def __init__(self, vocab_size, d_model, N, h, dff, latent_dim, nconds, dropout, variational=True):
        super(Encoder, self).__init__()
        self.N = N
        self.nconds = nconds
        self.variational = variational
        # input embedding layers
        self.embed_sentence = Embeddings(d_model, vocab_size)
        if nconds > 0:
            self.embed_cond2enc = nn.Linear(nconds, d_model*nconds) # nn.Linear() supports TensorFloat32
        # other layers
        self.norm = Norm(d_model)
        self.pe = PositionalEncoding(d_model, dropout=dropout)
        self.layers = get_clones(EncoderLayer(h, d_model, dff, dropout), N)
        # sampling mean and var
        self.fc_mu = nn.Linear(d_model, latent_dim)
        self.fc_log_var = nn.Linear(d_model, latent_dim)

    def forward(self, src, mask, econds=None):
        x = self.embed_sentence(src)
        if self.nconds > 0:
            cond2enc = self.embed_cond2enc(econds)
            cond2enc = cond2enc.view(econds.size(0), econds.size(1), -1)
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
    def __init__(self, vocab_size, d_model, N, h, dff, latent_dim,
                 nconds, dropout, use_cond2dec, use_cond2lat):
        super(Decoder, self).__init__()
        self.N = N
        self.nconds = nconds
        self.d_model = d_model
        self.use_cond2dec = use_cond2dec
        self.use_cond2lat = use_cond2lat
        self.embed = Embeddings(d_model, vocab_size)
        if self.use_cond2dec and nconds > 0:
            self.embed_cond2dec = nn.Linear(nconds, d_model*nconds) #concat to trg_input
        if self.use_cond2lat and nconds > 0:
            self.embed_cond2lat = nn.Linear(nconds, d_model*nconds) #concat to trg_input
        self.pe = PositionalEncoding(d_model, dropout=dropout)
        self.fc_z = nn.Linear(latent_dim, d_model)
        self.layers = get_clones(DecoderLayer(h, d_model, dff, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, trg, e_outputs, src_mask, trg_mask, dconds=None):
        # e_outputs: (bs, nc+src_len, lat_dim)
        # src_mask: (bs, 1, nc+src_len)
        # trg_mask: (bs, trg_len-1, trg_len-1)
        x = self.embed(trg)
        # x: (bs, trg_len-1, d_model)        
        e_outputs = self.fc_z(e_outputs)
        # e_outputs: (bs, nc+src_len, d_model)

        if self.use_cond2dec and self.nconds > 0:
            cond2dec = self.embed_cond2dec(dconds).view(dconds.size(0), dconds.size(1), -1)
            x = torch.cat([cond2dec, x], dim=1)
            # x: (bs, (nc+src_len)+(trg_len-1), d_model)
        elif self.use_cond2lat and self.nconds > 0:
            cond2lat = self.embed_cond2lat(dconds).view(dconds.size(0), dconds.size(1), -1)
            # cond2lat: (bs, nc, d_model)
            e_outputs = torch.cat([cond2lat, e_outputs], dim=1)
            # e_outputs: (bs, nc+(nc+src_len), d_model)
        x = self.pe(x)

        if self.use_cond2lat and self.nconds > 0:
            cond_mask = torch.ones_like(torch.unsqueeze(dconds, -2), dtype=bool)
            src_mask = torch.cat([cond_mask, src_mask], dim=2)
            # src_mask: (bs, 1, nc+(nc+src_len))

        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)


class Cvaetf(nn.Module):
    def __init__(self, src_vocab, trg_vocab, N=6, d_model=256, dff=2048,
                 h=8, latent_dim=64,  dropout=0.1, nconds=3,
                 use_cond2dec=False, use_cond2lat=False, variational=True):
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
        if self.use_cond2dec and nconds > 0:
            self.prop_fc = nn.Linear(trg_vocab, 1)
        self.out = nn.Linear(d_model, trg_vocab)

        # initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            
        # for p in self.parameters():
        #     if p.dim() > 1:
        #         nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask, econds=None):
        z, mu, log_var = self.encoder(src, src_mask, econds)
        return z, mu, log_var

    def decode(self, trg, z, src_mask, trg_mask, dconds=None):
        x = self.decoder(trg, z, src_mask, trg_mask, dconds)
        return self.out(x)

    def forward(self, src, trg, src_mask, trg_mask, econds=None, dconds=None):
        z, mu, log_var = self.encoder(src, src_mask, econds)
        d_output = self.decoder(trg, z, src_mask, trg_mask, dconds)
        output = self.out(d_output)

        if self.use_cond2dec and self.nconds > 0:
            output_prop = self.prop_fc(output[:, :self.nconds, :])
            output_mol = output[:, self.nconds:, :]
        elif self.nconds > 0:
            output_prop = torch.zeros(output.size(0), self.nconds, 1)
            output_mol = output
        else:
            output_prop = None
            output_mol = output
        return output_prop, output_mol, mu, log_var, z
