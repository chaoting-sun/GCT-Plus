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
        loss = self.loss_function(x.contiguous().view(-1, x.size(-1)),
                                  y.contiguous().view(-1))
        if self.optim is not None: # training section
            loss.backward()
            self.optim.step()
            self.optim.optimizer.zero_grad()
        return loss.data


class Criterion(nn.Module):
    """ 
    - function: compute reconstruction loss (contain label smoothing) and KL divergence
        - dependence 1: ReconstructionLoss
        - dependence 2: KLDivergence
    """
    def __init__(self, size, padding_idx, smoothing=0.00):
        super(Criterion, self).__init__()
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        rec_loss = ReconstructionLoss(x, target, self.size, self.smoothing, 
                                      self.confidence, self.padding_idx)
        return rec_loss


class VAELossCompute:
    """ 
    - function: compute loss and train model
        - dependence: loss function
    """
    def __init__(self, loss_function, optim):
        self.loss_function = loss_function
        self.optim = optim

    def __call__(self, x, y, norm, mu, logvar, beta):
        rec_loss, KL_div = self.loss_function(x.contiguous().view(-1, x.size(-1)),
                                              y.contiguous().view(-1), mu, logvar, beta)
        loss = rec_loss + KL_div

        if self.optim is not None: # training section
            loss.backward()
            self.optim.step()
            self.optim.optimizer.zero_grad()

        return rec_loss.data, KL_div


class VAECriterion(nn.Module):
    """ 
    - function: compute reconstruction loss (contain label smoothing) and KL divergence
        - dependence 1: ReconstructionLoss
        - dependence 2: KLDivergence
    """
    def __init__(self, size, padding_idx, smoothing=0.00):
        super(VAECriterion, self).__init__()
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target, mu, logvar, beta):
        rec_loss = ReconstructionLoss(x, target, self.size, self.smoothing, 
                                      self.confidence, self.padding_idx)
        _, kl_div_beta = KLDivergence(logvar, mu, beta)
        return rec_loss, kl_div_beta


def ReconstructionLoss(x, target, size, smoothing, confidence, padding_idx):
    # dim. of true_dist: (batch_size*max_trg_len, vocab_len)
    true_dist = x.data.clone()

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
    return nn.KLDivLoss(reduction='batchmean')(x, Variable(true_dist, requires_grad=False))


def KLDivergence(logvar, mu, beta):
    # ref: https://github.com/oriondollar/TransVAE/tree/578cb2b015e205da362332336d8f4815bd373edc
    KL_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    # return a scaled kl divergence
    return KL_div, KL_div * beta


class KLAnnealer:
    """
    - function: Scales KL weight by beta in a cyclical schedule
    - literature: https://aclanthology.org/N19-1021.pdf
    - github: https://github.com/haofuml/cyclical_annealing/blob/master/language_model/
    """
    def __init__(self, low_value, high_value, cycle, p=0.7):
        self.low_value = low_value
        self.high_value = high_value
        self.cycle = cycle
        self.cycle_top = int(cycle * p)
        self.slope = (high_value - low_value) / self.cycle_top

    def __call__(self, rounds):
        beta = min(self.high_value, (rounds % self.cycle) * self.slope)
        return beta