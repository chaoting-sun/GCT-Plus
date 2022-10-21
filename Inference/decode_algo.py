import math
import numpy as np

import torch

from Inference.toklen_sampling import tokenlen_gen_from_data_distribution
from Model.modules import Norm, create_target_mask, create_source_mask, get_clones, nopeak_mask


def generate_latent_space(toklen_data, latent_dim, device, n=1):
    # 3 properties + number of source tokens sampled from training set
    
    toklen = 3 + int(tokenlen_gen_from_data_distribution(data=toklen_data,
                 nBins=int(toklen_data.max() - toklen_data.min()), size=1))
    z = torch.Tensor(np.random.normal(size=(n, toklen, latent_dim)))
    return z.to(device), toklen


class Sampling(object):
    def __init__(self, predictor, latent_dim, TRG, toklen_data,
                 scaler, max_strlen, use_cond2dec, device):
        self.predictor = predictor
        self.latent_dim = latent_dim

        self.pad_idx = TRG.vocab.stoi['<pad>']
        self.eos_idx = TRG.vocab.stoi['<eos>']
        self.sos_idx = TRG.vocab.stoi['<sos>']
        self.trg_itos = TRG.vocab.itos

        self.toklen_data = toklen_data
        self.scaler = scaler
        self.max_strlen = max_strlen
        self.use_cond2dec = use_cond2dec
        self.device = device

        self.cond_dim = 3

    # def generate_latent_space(self):
    #     # 3 properties + number of source tokens sampled from training set
        
    #     toklen = 3 + int(tokenlen_gen_from_data_distribution(data=self.toklen_data,
    #                                                          nBins=int(self.toklen_data.max() - self.toklen_data.min()), size=1))
    #     z = torch.Tensor(np.random.normal(size=(1, toklen, self.latent_dim)))
    #     return z.to(self.device), toklen

    def scaler_transform(self, properties):
        """
        1. has no src
            - oritf: decoder ; dcond
            - mlptf: mlp + sampler + decoder ; econds + dconds
            - atttf: don't know yet ; 
        2. has src
            - oritf: encoder + sampler + decoder ; dconds
            - mlptf: encoder + sampler + mlp + sampler + decoder ; econds + dconds
            - atttf: encoder + sampler(half) + self-att + sampler + decoder ; econds + dconds
        """
        if isinstance(properties, tuple):
            econds = torch.Tensor(self.scaler.transform(
                properties[0])).to(self.device)
            dconds = torch.Tensor(self.scaler.transform(
                properties[1])).to(self.device)
            mconds = torch.cat((econds, torch.sub(dconds, econds)), axis=1)
            conds = (mconds, dconds)
        else:
            conds = torch.Tensor(self.scaler.transform(
                properties)).to(self.device)
        return conds

    def idx_sequence_to_smiles(self, idx_sequence):
        # only for 1 smiles
        idx_sequence = [idx for idx in idx_sequence if idx != self.pad_idx]
        smi_list = [self.trg_itos[token] for token in idx_sequence][1:-1]
        return ''.join(smi_list)

    def get_z_from_src(self, src, econds):
        if not getattr(self.predictor, 'encode'):
            exit(f"Class method not found: {self.predictor}.encode")
        src_mask = create_source_mask(
            src, self.pad_idx, econds)  # conds1->dconds
        z, mu, log_var, q_k_enc = self.predictor.encode(src, econds, src_mask)
        return z, mu, log_var, q_k_enc


class MultinomialSearch(Sampling):
    def __init__(self, predictor, latent_dim, TRG, toklen_data,
                 scaler, max_strlen, use_cond2dec, device):
        super().__init__(predictor, latent_dim, TRG, toklen_data,
                         scaler, max_strlen, use_cond2dec, device)
        # self.decode_algo = "greedy"
        self.decode_algo = "multinomial"

    def decode(self, z, conds, src_mask):
        c = conds[1] if isinstance(conds, tuple) else conds

        break_condition = torch.zeros(z.size(0), dtype=torch.bool)

        # create a batch of starting tokens (1)
        ys = (torch.ones(z.size(0), 1, requires_grad=True)
              * self.sos_idx).to(dtype=torch.long, device=self.device)

        with torch.no_grad():
            for i in range(self.max_strlen - 1):
                # create a padding/nopeaking mask of target
                trg_mask = create_target_mask(ys, self.pad_idx, c,
                                              self.use_cond2dec).to(self.device)

                prob = self.predictor.predict(ys, z, conds, src_mask, trg_mask)
                prob = prob[:, -1, :]

                if self.decode_algo == 'greedy':
                    _, next_word = torch.max(prob, dim=1)
                    ys = torch.cat([ys, next_word.unsqueeze(-1)],
                                   dim=1)  # [batch_size, i]
                elif self.decode_algo == 'multinomial':
                    next_word = torch.multinomial(
                        prob, 1)  # shape: (batch_size, 1)
                    ys = torch.cat([ys, next_word], dim=1)  # [batch_size, i]
                    next_word = torch.squeeze(next_word)  # shape: (batch_size)

                # update the break condition. 2 is the stop token
                end_condition = (next_word.to('cpu') == self.eos_idx)
                break_condition = (break_condition | end_condition)

                # If all satisfies the break condition, then break the loop.
                if all(break_condition):
                    break
        return ys

    def sample_smiles(self, conds, z=None, transform=True):
        if transform:
            conds = self.scaler_transform(conds)

        if z is not None:
            toklen = z.size(1)
        else:
            z, toklen = generate_latent_space(self.toklen_data, self.latent_dim, self.device)
        src_mask = (torch.ones(1, 1, toklen) != 0).to(
            self.device)  # (bs,1,toklen=nc+src_smi)

        pred_seq = self.decode(z, conds, src_mask)
        smiles = self.idx_sequence_to_smiles(pred_seq.cpu().numpy()[0])
        toklen_gen = len(smiles)
        return smiles, toklen_gen, toklen


class MultinomialSearchFromSource(MultinomialSearch):
    def __init__(self, predictor, latent_dim, TRG,
                 toklen_data, scaler, max_strlen, use_cond2dec, device):
        super().__init__(predictor, latent_dim, TRG, toklen_data,
                         scaler, max_strlen, use_cond2dec, device)

    def sample_smiles(self, src, conds, transform=True):
        toklen = src.size(-1)
        if transform:
            conds = self.scaler_transform(conds)

        if isinstance(conds, tuple):
            src_mask = create_source_mask(
                src, self.pad_idx, conds[1])  # conds1->dconds
        else:
            src_mask = create_source_mask(src, self.pad_idx, conds)
        # z, mu, logvar, q_k_enc = self.predictor.encode(src, conds, src_mask)
        z = self.predictor.encode(src, conds, src_mask)[0]
        pred_seq = self.decode(z, conds, src_mask)
        smiles = self.idx_sequence_to_smiles(pred_seq.cpu().numpy()[0])
        toklen_gen = len(smiles)
        return smiles, toklen_gen, toklen


class BeamSearch(Sampling):
    def __init__(self, predictor, latent_dim, TRG, toklen_data,
                 scaler, max_strlen, use_cond2dec, device, k=4):
        super().__init__(predictor, latent_dim, TRG, toklen_data,
                         scaler, max_strlen, use_cond2dec, device)
        self.k = k

    def k_best_outputs(self, outputs, out, log_scores, i):
        probs, ix = out[:, -1].data.topk(self.k)
        # the log probabilities from init. token to now (dim=(k,1))
        log_probs = torch.Tensor([math.log(p) for p in probs.data.view(-1)]).view(self.k, -1)\
            + log_scores.transpose(0, 1)
        k_probs, k_ix = log_probs.view(-1).topk(self.k)

        row = torch.div(k_ix, self.k, rounding_mode='floor')
        col = k_ix % self.k  # row = k_ix // k

        outputs[:, :i] = outputs[row, :i]  # dim=(k,max_str)
        outputs[:, i] = ix[row, col]

        log_scores = k_probs.unsqueeze(0)
        return outputs, log_scores

    def init_vars(self, conds, toklen, z):
        trg_in = torch.LongTensor([[self.sos_idx]]).to(
            self.device)  # dim=(1,1)
        src_mask = (torch.ones(1, 1, toklen) != 0).to(self.device)
        # trg_mask = create_target_mask(1, self.pad_idx, conds, self.use_cond2dec)

        trg_mask = nopeak_mask(1, self.use_cond2dec, self.pad_idx,
                               self.cond_dim).to(self.device)

        out_mol = self.predictor.predict(trg_in, z, conds, src_mask, trg_mask)

        # return the k elements with the highest probability
        probs, ix = out_mol[:, -1].data.topk(self.k)  # value/index (dim=(1,k))
        # the log-scale scores (dim=(1,k))
        log_scores = torch.Tensor([math.log(prob)
                                   for prob in probs.data[0]]).unsqueeze(0)
        # k outputs
        outputs = torch.zeros(self.k, self.max_strlen).long().to(self.device)
        outputs[:, 0] = self.sos_idx
        outputs[:, 1] = ix[0]

        e_outputs = torch.zeros(self.k, z.size(-2), z.size(-1)).to(self.device)
        e_outputs[:, :] = z[0]

        return outputs, e_outputs, log_scores

    def beam_search(self, conds, toklen, z):
        # cond = cond.view(1, -1)

        # 维持三个变量，e_outputs,outputs,log_scores
        # outputs 维度(beam_size,max_len) e_outputs(beam_size,seq_len,d_model)

        outputs, e_outputs, log_scores = self.init_vars(conds, toklen, z)
        print('init:', outputs.size())

        if len(conds) == 2:
            conds = (conds[0].repeat(self.k, 1), conds[1].repeat(self.k, 1))
        else:
            conds = conds.repeat(self.k, 1)

        ind = None
        src_mask = (torch.ones(1, 1, toklen) != 0)
        src_mask = src_mask.repeat(self.k, 1, 1).to(self.device)

        for i in range(2, self.max_strlen):
            trg_mask = nopeak_mask(i, self.use_cond2dec,
                                   self.pad_idx, self.cond_dim)
            trg_mask = trg_mask.repeat(self.k, 1, 1).to(self.device)

            out_mol = self.predictor.predict(
                outputs[:, :i], e_outputs, conds, src_mask, trg_mask)
            outputs, log_scores = self.k_best_outputs(
                outputs, out_mol, log_scores, i)  # len(outputs) == k

            # Occurrences of end symbols for all input sentences. (index)
            ones = (outputs == self.eos_idx).nonzero()
            sentence_lengths = torch.zeros(
                len(outputs), dtype=torch.long).to(self.device)

            for vec in ones:
                i = vec[0]  # i-th
                if sentence_lengths[i] == 0:  # First end symbol has not been found yet
                    # Position of first end symbol
                    sentence_lengths[i] = vec[1]

            num_finished_sentences = len(
                [s for s in sentence_lengths if s > 0])

            if num_finished_sentences == self.k:
                alpha = 0.7
                div = 1/(sentence_lengths.type_as(log_scores)**alpha)
                _, ind = torch.max(log_scores * div, 1)
                ind = ind.data[0]
                break

        if ind is None:
            length = (outputs[0] == self.eos_idx).nonzero()[0]
            outs = ' '.join([self.trg_itos[tok]
                            for tok in outputs[0][1:length]])
            print(outs)
            return outs
        else:
            length = (outputs[ind] == self.eos_idx).nonzero()[0]
            return ' '.join([self.trg_itos[tok] for tok in outputs[ind][1:length]])

    def sample_smiles(self, conds, z=None, transform=True):
        if transform:
            conds = self.scaler_transform(conds)

        if z is not None:
            toklen = z.size(1)
        else:
            z, toklen = generate_latent_space(self.toklen_data, self.latent_dim, self.device)
        smiles = self.beam_search(conds, toklen, z)

        toklen_gen = smiles.count(" ") + 1
        smiles = ''.join(smiles).replace(" ", "")

        return smiles, toklen_gen, toklen


class BeamSearchFromSource(BeamSearch):
    def __init__(self, predictor, latent_dim, TRG, toklen_data,
                 scaler, max_strlen, use_cond2dec, device, k=4):
        super().__init__(predictor, latent_dim, TRG, toklen_data,
                         scaler, max_strlen, use_cond2dec, device, k)

    def sample_smiles(self, src, conds, transform=True):
        if transform:
            conds = self.scaler_transform(conds)

        if isinstance(conds, tuple):
            src_mask = create_source_mask(
                src, self.pad_idx, conds[1])  # conds1->dconds
        else:
            src_mask = create_source_mask(src, self.pad_idx, conds)
        # z, mu, logvar, q_k_enc = self.z_generator(src, conds, src_mask)
        z = self.predictor.encode(src, conds, src_mask)[0]
        toklen = src.size(-1)
        smiles = self.beam_search(conds, toklen, z)
        toklen_gen = smiles.count(" ") + 1
        smiles = ''.join(smiles).replace(" ", "")
        return smiles, toklen_gen, toklen


# class BeamSearchTool(object):
#     def __init__(self, predictor, use_cond2dec, cond_dim, latent_dim, trg_vocab,
#                  max_strlen, toklen_data, scaler, device, z_generator=generate_z_from_toklen):
#         self.k = 4  # fixed

#         self.predictor = predictor
#         self.use_cond2dec = use_cond2dec

#         self.cond_dim = cond_dim
#         self.latent_dim = latent_dim

#         self.pad_idx = trg_vocab.stoi['<pad>']
#         self.eos_idx = trg_vocab.stoi['<eos>']
#         self.sos_idx = trg_vocab.stoi['<sos>']
#         self.trg_itos = trg_vocab.itos

#         self.max_strlen = max_strlen
#         self.toklen_data = toklen_data
#         self.scaler = scaler
#         self.device = device
#         self.z_generator = z_generator

#     def k_best_outputs(self, outputs, out, log_scores, i):
#         probs, ix = out[:, -1].data.topk(self.k)
#         # the log probabilities from init. token to now (dim=(k,1))
#         log_probs = torch.Tensor([math.log(p) for p in probs.data.view(-1)]).view(self.k, -1)\
#             + log_scores.transpose(0, 1)
#         k_probs, k_ix = log_probs.view(-1).topk(self.k)

#         row = torch.div(k_ix, self.k, rounding_mode='floor')
#         col = k_ix % self.k  # row = k_ix // k

#         outputs[:, :i] = outputs[row, :i]  # dim=(k,max_str)
#         outputs[:, i] = ix[row, col]

#         log_scores = k_probs.unsqueeze(0)
#         return outputs, log_scores

#     def init_vars(self, predictor, conds, init_tok, toklen, z, device):
#         trg_in = torch.LongTensor([[init_tok]]).to(device)  # dim=(1,1)
#         src_mask = (torch.ones(1, 1, toklen) != 0).to(device)
#         # trg_mask = create_target_mask(1, self.pad_idx, conds, self.use_cond2dec)
#         trg_mask = nopeak_mask(1, self.cond_dim, self.use_cond2dec).to(device)

#         out_mol = predictor.predict(trg_in, z, conds, src_mask, trg_mask)

#         # return the k elements with the highest probability
#         probs, ix = out_mol[:, -1].data.topk(self.k)  # value/index (dim=(1,k))
#         # the log-scale scores (dim=(1,k))
#         log_scores = torch.Tensor([math.log(prob)
#                                    for prob in probs.data[0]]).unsqueeze(0)
#         # k outputs
#         outputs = torch.zeros(self.k, self.max_strlen).long().to(device)
#         outputs[:, 0] = init_tok
#         outputs[:, 1] = ix[0]

#         e_outputs = torch.zeros(self.k, z.size(-2), z.size(-1)).to(device)
#         e_outputs[:, :] = z[0]

#         return outputs, e_outputs, log_scores

#     def beam_search(self, conds, predictor, toklen, z, device):
#         # cond = cond.view(1, -1)

#         # 维持三个变量，e_outputs,outputs,log_scores
#         # outputs 维度(beam_size,max_len) e_outputs(beam_size,seq_len,d_model)

#         outputs, e_outputs, log_scores = self.init_vars(predictor, conds, self.sos_idx,
#                                                         toklen, z, device)
#         if isinstance(conds, tuple):
#             conds[0], conds[1] = conds[0].repeat(
#                 self.k, 1), conds[1].repeat(self.k, 1)
#         else:
#             conds = conds.repeat(self.k, 1)

#         ind = None
#         src_mask = (torch.ones(1, 1, toklen) != 0)
#         src_mask = src_mask.repeat(self.k, 1, 1).to(device)

#         for i in range(2, self.max_strlen):
#             trg_mask = nopeak_mask(i, self.cond_dim, self.use_cond2dec)
#             trg_mask = trg_mask.repeat(self.k, 1, 1).to(device)

#             out_mol = predictor.predict(
#                 outputs[:, :i], e_outputs, conds, src_mask, trg_mask)
#             outputs, log_scores = self.k_best_outputs(
#                 outputs, out_mol, log_scores, i)

#             # Occurrences of end symbols for all input sentences. (index)
#             ones = (outputs == self.eos_idx).nonzero()
#             # len(outputs) == k
#             sentence_lengths = torch.zeros(
#                 len(outputs), dtype=torch.long).to(device)

#             for vec in ones:
#                 i = vec[0]  # i-th
#                 if sentence_lengths[i] == 0:  # First end symbol has not been found yet
#                     # Position of first end symbol
#                     sentence_lengths[i] = vec[1]

#             num_finished_sentences = len(
#                 [s for s in sentence_lengths if s > 0])

#             if num_finished_sentences == self.k:
#                 alpha = 0.7
#                 div = 1/(sentence_lengths.type_as(log_scores)**alpha)
#                 _, ind = torch.max(log_scores * div, 1)
#                 ind = ind.data[0]
#                 break

#         if ind is None:
#             length = (outputs[0] == self.eos_idx).nonzero()[0]
#             outs = ' '.join([self.trg_itos[tok]
#                             for tok in outputs[0][1:length]])
#             print(outs)
#             return outs
#         else:
#             length = (outputs[ind] == self.eos_idx).nonzero()[0]
#             return ' '.join([self.trg_itos[tok] for tok in outputs[ind][1:length]])


#     def sample_smiles(self, properties):
#         # +3 -> 可能是作者不小心寫錯了，作者是用 cond2lat
#         # toklen = int(tokenlen_gen_from_data_distribution(data=self.toklen_data, nBins=int(
#         #     self.toklen_data.max() - self.toklen_data.min()), size=1)) + 3  # +3 due to cond2enc
#         z, toklen = self.z_generator(
#             self.toklen_data, self.latent_dim, self.device)

#         if type(properties) == tuple:
#             econds = torch.Tensor(self.scaler.transform(
#                 properties[0])).to(self.device)
#             dconds = torch.Tensor(self.scaler.transform(
#                 properties[1])).to(self.device)
#             mconds = torch.cat((econds, torch.sub(dconds, econds)), axis=1)
#             conds = (mconds, dconds)
#         else:
#             conds = torch.Tensor(self.scaler.transform(
#                 properties)).to(self.device)

#         molecule = self.beam_search(
#             conds, self.predictor, toklen, z, self.device)
#         toklen_gen = molecule.count(" ") + 1
#         molecule = ''.join(molecule).replace(" ", "")
#         return molecule, toklen_gen, toklen

    # def sample_molecule(self, conds, toklen_data, predictor, TRG, scaler, device):
    #     toklen = int(tokenlen_gen_from_data_distribution(data=toklen_data, nBins=int(
    #         toklen_data.max() - toklen_data.min()), size=1)) + 3  # +3 due to cond2enc

    #     z = torch.Tensor(np.random.normal(size=(1, toklen, self.latent_dim))).to(device)

    #     if type(conds) == tuple:
    #         econds, dconds = torch.Tensor(scaler.transform(conds[0])).to(device),\
    #                          torch.Tensor(scaler.transform(conds[1])).to(device)
    #         mconds = torch.cat((econds, torch.sub(dconds, econds)), axis=1)
    #         conds = [mconds, dconds]
    #         n = len(mconds)
    #     else:
    #         conds = torch.Tensor(scaler.transform(conds)).to(device)
    #         n = len(conds)

    #     molecule = []
    #     for i in range(n):
    #         molecule.append(self.beam_search(conds, predictor, TRG, toklen, z, device))
    #     toklen_gen = molecule[0].count(" ") + 1
    #     molecule = ''.join(molecule).replace(" ", "")
    #     return molecule, toklen_gen, toklen
