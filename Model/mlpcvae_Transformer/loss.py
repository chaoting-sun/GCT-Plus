import torch.nn as nn
import torch
from torch.autograd import Variable


class LossCompute:
    """ 
    - function: compute loss and train model
        - dependence: loss function
    """
    def __init__(self, loss_function, optim):
        self.loss_function = loss_function
        self.optim = optim

    def __call__(self, x, y):
        loss = self.loss_function(x.contiguous().view(-1),
                                  y.contiguous().view(-1))
        if self.optim is not None: # training section
            loss.backward()
            self.optim.step()
            self.optim.optimizer.zero_grad()
        return loss.data


# class LossCompute:
#     """ 
#     - function: compute loss and train model
#         - dependence: loss function
#     """
#     def __init__(self, loss_function, optim):
#         self.loss_function = loss_function
#         self.optim = optim

#     def __call__(self, x, y):
#         loss = self.loss_function(x.contiguous().view(-1, x.size(-1)),
#                                   y.contiguous().view(-1))
#         if self.optim is not None: # training section
#             loss.backward()
#             self.optim.step()
#             self.optim.optimizer.zero_grad()
#         return loss.data


class Criterion(nn.Module):
    """ 
    - function: compute reconstruction loss (contain label smoothing) and KL divergence
        - dependence 1: ReconstructionLoss
        - dependence 2: KLDivergence
    """
    def __init__(self):
        super(Criterion, self).__init__()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, predict, target):
        return self.kl_loss(predict, target)


def ReconstructionLoss(x, target, size, smoothing, confidence, padding_idx):
    # dim. of true_dist: (batch_size*max_trg_len, vocab_len)
    true_dist = x.data.clone()
    # torch.set_printoptions(threshold=10_000)
    
    # fill true_dist with "self.smoothing / (self.size - 2)"
    # -2 as we are not distributing the smoothing mass over
    # the pad token idx and over the ground truth index
    true_dist.fill_(smoothing / (size - 2))
    
    # change the value at the ground truth index to be "self.confidence"
    true_dist.scatter_(dim=1, index=target.data.unsqueeze(1), value=confidence)

    # The pad index does not have prob. Set it to be 0
    true_dist[:, padding_idx] = 0

    # the padding positions are not revalent to the sequence, and does not have prob.
    mask = torch.nonzero(target.data == padding_idx)
    if mask.dim() > 0:
        true_dist.index_fill_(0, mask.squeeze(), 0.0)

    # return recontruction loss (sum of a batch)
    # One should notice that x should pass through log_softmax before we compute kldiv
    return nn.KLDivLoss(reduction='batchmean')(x, Variable(true_dist, requires_grad=False))
