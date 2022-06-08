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

        # dim. of true_dist: (batch_size*max_trg_len, vocab_len)
        true_dist = x.data.clone()

        # fill true_dist with "self.smoothing / (self.size - 2)" 
        # -2 as we are not distributing the smoothing mass over
        # the pad token idx and over the ground truth index
        true_dist.fill_(self.smoothing / (self.size - 2))
        
        # change the value at the ground truth index to be "self.confidence"
        true_dist.scatter_(dim=1, index=target.data.unsqueeze(1), value=self.confidence)

        # The pad index does not have prob. Set it to be 0
        true_dist[:, self.padding_idx] = 0       

        # the padding positions are not revalent to the sequence, and does not have prob.
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
            
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))
