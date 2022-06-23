import torch
import torch.nn as nn
import copy
import torch.nn.functional as F

from .sublayers import Norm
from .layers import EncoderLayer, DecoderLayer
from .embeddings import Embeddings, PositionalEncoding


def get_clones(layer, N):
    return nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


class Encoder(nn.Module):
    "Pass N encoder layers, followed by a layernorm"
    def __init__(self, layer, norm, N):
        super(Encoder, self).__init__()
        self.layers = get_clones(layer, N)
        self.norm = norm
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)                
        return self.norm(x)


class Decoder(nn.Module):
    "Pass N decoder layers, followed by a layernorm"
    def __init__(self, layer, norm, N):
        super(Decoder, self).__init__()
        self.layers = get_clones(layer, N)
        self.norm = norm

    def forward(self, x, m, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, m, src_mask, tgt_mask)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, N=6, d_model=256, d_ff=2048, h=8, dropout=0.1, num_properties=1):
        super(Transformer, self).__init__()

        cp = copy.deepcopy
        position = PositionalEncoding(d_model, dropout)
        # self.properties_nn = nn.Sequential(nn.Linear(num_properties, d_model), nn.ReLU())

        # source/target embedding
        self.src_embed = nn.Sequential(Embeddings(d_model, src_vocab), cp(position))
        self.tgt_embed = nn.Sequential(Embeddings(d_model, tgt_vocab), cp(position))
        # encoder
        encoder_layer = EncoderLayer(h, d_model, d_ff, dropout)
        self.encoder = Encoder(encoder_layer, Norm(d_model), N)
        # decoder
        decoder_layer = DecoderLayer(h, d_model, d_ff, dropout)
        self.decoder = Decoder(decoder_layer, Norm(d_model), N)
        # generator
        self.generator = Generator(d_model, tgt_vocab)

        self._reset_parameters()
    

    def encode(self, src, src_mask):
        embed = self.src_embed(src)
        encoded = self.encoder(embed, src_mask)
        return encoded     

        
    def decode(self, memory, src_mask, tgt, tgt_mask):
        embed = self.tgt_embed(tgt)
        decoded = self.decoder(embed, memory, src_mask, tgt_mask)
        return decoded


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, src, tgt, src_mask, tgt_mask):
        encoded = self.encode(src, src_mask)
        decoded = self.decode(encoded, src_mask, tgt, tgt_mask)
        return decoded
        

def build_transformer(src_vocab, tgt_vocab, N, d_model, 
                      d_ff, h, dropout, num_properties):
    return Transformer(src_vocab, tgt_vocab, N, d_model, 
                       d_ff, h, dropout, num_properties)


def load_from_file(file_path):
    # Load model
    checkpoint = torch.load(file_path, map_location='cuda:0')
    # checkpoint = torch.load(file_path) # change
    para_dict = checkpoint['model_parameters']
    model = Transformer(para_dict['vocab_size'], para_dict['vocab_size'], 
                        para_dict['N'], para_dict['d_model'], para_dict['d_ff'], 
                        para_dict['H'], para_dict['dropout'], para_dict['num_properties'])
    model.load_state_dict(checkpoint['model_state_dict'])
    return model