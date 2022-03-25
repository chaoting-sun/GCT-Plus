import torch
import torch.nn as nn
from .sublayers import FeedForward, MultiHeadAttention, Norm


class EncoderLayer(nn.Module):
    def __init__(self, heads, d_model, dff, dropout):
        super(EncoderLayer, self).__init__()
        # multi-head self-attention
        self.sa_norm = Norm(d_model)
        self.self_attn = MultiHeadAttention(heads, d_model, dropout)
        self.sa_dropout = nn.Dropout(dropout)
        # position-wise feed-forward
        self.ff_norm = Norm(d_model)
        self.ff = FeedForward(d_model, dff, dropout)
        self.ff_dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        # multi-head self-attention
        x = self.sa_norm(x)
        x2, concat_attn = self.self_attn(x, x, x, mask)
        x = x + self.sa_dropout(x2)
        # position-wise feed-forward
        x = self.ff_norm(x)
        x2 = self.ff(x)
        x = x + self.ff_dropout(x2)
        return x, concat_attn


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, heads, d_model, dff, dropout, use_cond2dec, use_cond2lat):
        self.use_cond2dec = use_cond2dec
        self.use_cond2lat = use_cond2lat

        super(DecoderLayer, self).__init__()
        # masked multi-head self-attention
        self.norm_1 = Norm(d_model)
        self.attn_1 = MultiHeadAttention(heads, d_model, dropout)
        self.dropout_1 = nn.Dropout(dropout)
        # multi-head self-attention
        self.norm_2 = Norm(d_model)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout)
        self.dropout_2 = nn.Dropout(dropout)
        # positionwise feed-forward
        self.norm_3 = Norm(d_model)
        self.ff = FeedForward(d_model, dff, dropout)
        self.dropout_3 = nn.Dropout(dropout)

    def forward(self, x, e_outputs, cond_input, src_mask, tgt_mask):
        # masked multi-head self-attention
        x = self.norm_1(x)
        attn_1, concat_attn_1 = self.attn_1(x, x, x, tgt_mask)
        x = x + self.dropout_1(attn_1)
        # multi-head self-attention
        x = self.norm_2(x)
        if self.use_cond2lat == True:
            cond_mask = torch.unsqueeze(cond_input, -2)
            cond_mask = torch.ones_like(cond_mask, dtype=bool)
            src_mask = torch.cat([cond_mask, src_mask], dim=2)
        attn_2, concat_attn_2 = self.attn_2(x, e_outputs, e_outputs, src_mask)
        x = x + self.dropout_2(attn_2)
        # position-wise feed-forward
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x, concat_attn_1, concat_attn_2