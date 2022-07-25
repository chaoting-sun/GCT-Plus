import math
import numpy as np

import torch

from Inference.toklen_sampling import tokenlen_gen_from_data_distribution
from Utils.mask import nopeak_mask


class BeamSearchTool(object):
    def __init__(self, cond_dim, latent_dim, max_strlen, model, use_cond2dec):
        self.k = 4 # fixed
        self.cond_dim = cond_dim
        self.latent_dim = latent_dim
        self.max_strlen = max_strlen
        self.model = model
        self.use_cond2dec = use_cond2dec


    def k_best_outputs(self, outputs, out, log_scores, i):
        probs, ix = out[:, -1].data.topk(self.k)
        # the log probabilities from init. token to now (dim=(k,1))
        log_probs = torch.Tensor([math.log(p) for p in probs.data.view(-1)]).view(self.k, -1)\
                                 + log_scores.transpose(0, 1)
        k_probs, k_ix = log_probs.view(-1).topk(self.k)

        row = torch.div(k_ix, self.k, rounding_mode='floor')
        col = k_ix % self.k # row = k_ix // k

        outputs[:, :i] = outputs[row, :i] # dim=(k,max_str)
        outputs[:, i] = ix[row, col]

        log_scores = k_probs.unsqueeze(0)
        return outputs, log_scores


    def init_vars(self, predictor, conds, init_tok, toklen, z, device):
        trg_in = torch.LongTensor([[init_tok]]).to(device) # dim=(1,1)
        src_mask = (torch.ones(1, 1, toklen) != 0).to(device)
        trg_mask = nopeak_mask(1, self.cond_dim, self.use_cond2dec).to(device)

        out_mol = predictor.predict(trg_in, z, conds, src_mask, trg_mask)

        # return the k largest elements: value/index (dim=(1,k))
        probs, ix = out_mol[:, -1].data.topk(self.k)
        # the log-scale scores (dim=(1,k))
        log_scores = torch.Tensor([math.log(prob)
                                for prob in probs.data[0]]).unsqueeze(0)
        # k outputs
        outputs = torch.zeros(self.k, self.max_strlen).long().to(device)
        outputs[:, 0] = init_tok
        outputs[:, 1] = ix[0]

        e_outputs = torch.zeros(self.k, z.size(-2), z.size(-1)).to(device)
        e_outputs[:, :] = z[0]

        return outputs, e_outputs, log_scores


    def beam_search(self, conds, predictor, TRG, toklen, z, device):
        sos_tok = TRG.vocab.stoi['<sos>']
        eos_tok = TRG.vocab.stoi['<eos>']

        # cond = cond.view(1, -1)

        # 维持三个变量，e_outputs,outputs,log_scores
        # outputs 维度(beam_size,max_len) e_outputs(beam_size,seq_len,d_model)

        outputs, e_outputs, log_scores = self.init_vars(predictor, conds, sos_tok,
                                                        toklen, z, device)
        if type(conds) == list:
            conds[0], conds[1] = conds[0].repeat(self.k, 1), conds[1].repeat(self.k, 1)
        else:
            conds = conds.repeat(self.k, 1)

        ind = None
        src_mask = (torch.ones(1, 1, toklen) != 0)
        src_mask = src_mask.repeat(self.k, 1, 1).to(device)

        for i in range(2, self.max_strlen):
            trg_mask = nopeak_mask(i, self.cond_dim, self.use_cond2dec)
            trg_mask = trg_mask.repeat(self.k, 1, 1).to(device)

            out_mol = predictor.predict(outputs[:, :i], e_outputs, conds, src_mask, trg_mask)
            outputs, log_scores = self.k_best_outputs(outputs, out_mol, log_scores, i)

            # Occurrences of end symbols for all input sentences. (index)
            ones = (outputs == eos_tok).nonzero()
            # len(outputs) == k
            sentence_lengths = torch.zeros(len(outputs), dtype=torch.long).to(device)

            for vec in ones:
                i = vec[0]  # i-th
                if sentence_lengths[i] == 0:  # First end symbol has not been found yet
                    sentence_lengths[i] = vec[1]  # Position of first end symbol

            num_finished_sentences = len([s for s in sentence_lengths if s > 0])

            if num_finished_sentences == self.k:
                alpha = 0.7
                div = 1/(sentence_lengths.type_as(log_scores)**alpha)
                _, ind = torch.max(log_scores * div, 1)
                ind = ind.data[0]
                break

        if ind is None:
            length = (outputs[0] == eos_tok).nonzero()[0]
            outs = ' '.join([TRG.vocab.itos[tok] for tok in outputs[0][1:length]])
            print(outs)
            return outs
        else:
            length = (outputs[ind] == eos_tok).nonzero()[0]
            return ' '.join([TRG.vocab.itos[tok] for tok in outputs[ind][1:length]])


    def sample_molecule(self, conds, toklen_data, predictor, TRG, scaler, device):
        toklen = int(tokenlen_gen_from_data_distribution(data=toklen_data, nBins=int(
            toklen_data.max() - toklen_data.min()), size=1)) + 3  # +3 due to cond2enc

        z = torch.Tensor(np.random.normal(size=(1, toklen, self.latent_dim))).to(device)

        if type(conds) == tuple:
            econds, dconds = torch.Tensor(scaler.transform(conds[0])).to(device),\
                             torch.Tensor(scaler.transform(conds[1])).to(device)
            mconds = torch.cat((econds, torch.sub(dconds, econds)), axis=1)
            conds = [mconds, dconds]
            n = len(mconds)
        else:
            conds = torch.Tensor(scaler.transform(conds)).to(device)
            n = len(conds)

        molecule = []
        for i in range(n):
            molecule.append(self.beam_search(conds, predictor, TRG, toklen, z, device))
        toklen_gen = molecule[0].count(" ") + 1
        molecule = ''.join(molecule).replace(" ", "")
        return molecule, toklen_gen, toklen