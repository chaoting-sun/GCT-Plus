import torch.nn as nn
import torch
from torch.autograd import Variable


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.00):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size # vocabulary size
        
        true_dist = x.data.clone() # (batch_size*(max_src_seq_length-1), vocab_len)
        true_dist.fill_(self.smoothing / (self.size - 2)) # (batch_size*(max_src_seq_length-1), vocab_len)
        
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        
        true_dist[:, self.padding_idx] = 0 # don't know why        
        mask = torch.nonzero(target.data == self.padding_idx) # don't know why

        if mask.dim() > 0: # mask.dim() = 2
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
            
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))
