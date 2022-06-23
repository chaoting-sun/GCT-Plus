import torch.nn as nn
from .sublayers import FeedForward, MultiHeadAttention, Norm


class EncoderLayer(nn.Module):
    def __init__(self, heads, d_model, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        # multi-head self-attention
        self.sa_norm = Norm(d_model)
        self.self_attn = MultiHeadAttention(heads, d_model, dropout)
        self.sa_dropout = nn.Dropout(dropout)
        # position-wise feed-forward
        self.ff_norm = Norm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.ff_dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        # multi-head self-attention
        x = self.sa_norm(x)
        x2 = self.self_attn(x, x, x, mask)
        x = x + self.sa_dropout(x2)
        # position-wise feed-forward
        x = self.ff_norm(x)
        x2 = self.ff(x)
        x = x + self.ff_dropout(x2)
        return x


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, heads, d_model, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        # masked multi-head self-attention
        self.norm_1 = Norm(d_model)
        self.self_attn = MultiHeadAttention(heads, d_model, dropout)
        self.dropout_1 = nn.Dropout(dropout)
        # multi-head self-attention
        self.norm_2 = Norm(d_model)
        self.src_attn = MultiHeadAttention(heads, d_model, dropout)
        self.dropout_2 = nn.Dropout(dropout)
        # positionwise feed-forward
        self.norm_3 = Norm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.dropout_3 = nn.Dropout(dropout)

    def forward(self, x, m, src_mask, tgt_mask):
        # masked multi-head self-attention
        x = self.norm_1(x)
        x2 = self.self_attn(x, x, x, tgt_mask)
        x = x + self.dropout_1(x2)
        # multi-head self-attention
        x = self.norm_2(x)
        x2 = self.src_attn(x, m, m, src_mask)
        x = x + self.dropout_2(x2)
        # position-wise feed-forward
        x = self.norm_3(x)
        x2 = self.ff(x)
        x = x + self.norm_3(x2)

        return x