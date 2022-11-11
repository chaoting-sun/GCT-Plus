import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from Utils.chrono import Chrono, Timer


class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, predict, target):
        return self.mse_loss(predict, target)


class KLDiv(nn.Module):
    def __init__(self):
        super(KLDiv, self).__init__()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, predict, target):
        predict = predict.contiguous().view(-1, predict.size()[-1])
        target = target.contiguous().view(-1, target.size()[-1])
        return self.kl_loss(F.log_softmax(predict, dim=-1), F.softmax(target, dim=-1))


class MSE_KLDiv(nn.Module):
    def __init__(self):
        super(MSE_KLDiv, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='mean') # default: mean
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.beta = 10

    def forward(self, predict, target):
        predict = predict.contiguous().view(-1, predict.size()[-1])
        target = target.contiguous().view(-1, target.size()[-1])
        self.mse_loss = self.mse_loss(predict, target)
        self.kl_loss = self.kl_loss(F.log_softmax(predict, dim=-1), F.softmax(target, dim=-1))
        return self.mse_loss + self.beta*self.kl_loss


# https://discuss.pytorch.org/t/jensen-shannon-divergence/2626/10
class JSD(nn.Module):
    def __init__(self):
        super(JSD, self).__init__()
        self.kl = nn.KLDivLoss(reduction='batchmean', log_target=True)

    def forward(self, p: torch.tensor, q: torch.tensor):
        p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
        m = (0.5 * (p + q)).log()
        return 0.5 * (self.kl(m, p.log()) + self.kl(m, q.log()))


def tfRecLoss(x, target, size, smoothing, confidence, padding_idx):
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
    # One should notice that x should pass through log_softmax before we compute kldiv
    return nn.KLDivLoss(reduction='batchmean')(x, Variable(true_dist, requires_grad=False))


# # ref: https://github.com/oriondollar/TransVAE/tree/578cb2b015e205da362332336d8f4815bd373edc
# def KLDiv(logvar, mu, beta):
#     return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()) * beta


class tfCriterion(nn.Module):
    """
    compute loss reconstruction loss and KL divergence for cvae-transformer
    """
    def __init__(self, size, padding_idx, smoothing=0.00, kldiv=False):
        super(tfCriterion, self).__init__()
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        self.kldiv = kldiv

    def forward(self, x, target, mu, logvar, beta):
        rec_loss = tfRecLoss(x, target, self.size, self.smoothing, 
                             self.confidence, self.padding_idx)
        if self.kldiv is False:
            return rec_loss
        kl_div_beta = KLDiv(logvar, mu, beta)
        return rec_loss, kl_div_beta


# paper: https://aclanthology.org/N19-1021.pdf
# github: https://github.com/haofuml/cyclical_annealing/blob/master/language_model/
class KLAnnealer:
    """
    scales KL weight by beta in a cyclical schedule
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


# class LossCompute:
#     """ 
#     update model parameters
#     """
#     def __init__(self, loss_function, optim):
#         self.loss_function = loss_function
#         self.optim = optim

#     def __call__(self, x, y, norm, mu, logvar, beta):
#         rec_loss, KL_div = self.loss_function(x.contiguous().view(-1, x.size(-1)),
#                                               y.contiguous().view(-1), mu, logvar, beta)
#         loss = rec_loss + KL_div

#         if self.optim is not None: # training section
#             loss.backward()
#             self.optim.step()
#             self.optim.optimizer.zero_grad()

#         return rec_loss.data, KL_div

class LossCompute:
    """ 
    - function: compute loss and update the model parameters
        - dependence: loss function
    """
    def __init__(self, loss_function, optim):
        self.loss_function = loss_function
        self.optim = optim
    
    def __call__(self, x, y):
        loss = self.loss_function(x.contiguous(), y.contiguous())
        # loss = self.loss_function(x.contiguous().view(-1), y.contiguous().view(-1))
        if self.optim is not None: # training section
            loss.backward()
            self.optim.step()
            # initialization is faster than setting all to 0
            self.optim.optimizer.zero_grad(set_to_none=True)
        return loss.data