import math
import numpy as np
import torch
from time import time

from Inference.toklen_sampling import tokenlen_gen_from_data_distribution
from Model.modules import create_target_mask, create_source_mask, nopeak_mask


class Sampling(object):
    def __init__(self, predictor, kwargs):
        self.predictor = predictor
        self.latent_dim   = kwargs['latent_dim']
        self.max_strlen   = kwargs['max_strlen']
        self.use_cond2dec = kwargs['use_cond2dec']
        self.decode_algo  = kwargs['decode_algo']
        self.toklen_data  = kwargs['toklen_data']
        self.scaler       = kwargs['scaler']
        self.device       = kwargs['device']

        TRG               = kwargs['TRG']
        self.pad_id       = TRG.vocab.stoi['<pad>']
        self.eos_id       = TRG.vocab.stoi['<eos>']
        self.sos_id       = TRG.vocab.stoi['<sos>']
        self.trg_itos     = TRG.vocab.itos

        self.cond_dim     = 3

    def sample_z_from_src(self, src, econds, mconds=None):
        """1. sample z from source and properties"""
        src_mask = create_source_mask(src, self.pad_id, econds)
        src_mask = src_mask.to(self.device)
        if mconds is not None:
            econds = (econds, mconds)
        try:
            return self.predictor.encode(src, econds, src_mask)
        except AttributeError:
            exit('Predictor has no attribute: encode')

    def sample_z_from_data(self, n=1):
        """2. sample z with toklen from data distribution"""
        z, toklens = [], []
        for i in range(n):
            # smiles length + three conditions
            toklen = int(tokenlen_gen_from_data_distribution(
                         data=self.toklen_data, size=1,
                         nBins=int(self.toklen_data.max()
                                   - self.toklen_data.min()))) + 3
            toklens.append(toklen)
            z.append(torch.Tensor(np.random.normal(
                size=(1, toklen, self.latent_dim))))
        return z, toklens

    def sample_fixed_len_z(self, toklen, n):
        """3. sample z given toklen"""
        return torch.Tensor(np.random.normal(
            size=(n, toklen, self.latent_dim)))

    def seq_to_smiles(self, seq):
        smiles = ''
        for id in seq:
            if id == self.eos_id:
                break
            if id != self.sos_id:
                smiles += self.trg_itos[id]
        return smiles

    def transform_props(self, props):
        return self.scaler.transform(props)

    def transform(self, properties):
        transformed = self.scaler.transform(properties)
        return torch.Tensor(transformed).to(self.device)

    def n_transform(self, properties, n=1):
        if n > 1:
            props = [self.transform(p) for p in properties]
            return torch.cat(props, dim=0)
        return self.transform(properties)

    def scaler_transform(self, properties):
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

    def encode_smiles(self, src, props, transform=True):
        if transform:
            props = self.transform_props(props)
        if props.dtype != torch.float32:
            props = torch.Tensor(props)
        
        src_mask = create_source_mask(src, self.pad_id, props)

        src = src.to(self.device)
        props = props.to(self.device)
        src_mask = src_mask.to(self.device)
        z, mu, log_var = self.predictor.encode(src, props, src_mask)
        return z, mu, log_var
    

class MultinomialSearch(Sampling):
    def __init__(self, predictor, kwargs):
        super().__init__(predictor, kwargs)

    def decode(self, z, conds, src_mask):
        c = conds[1] if isinstance(conds, tuple) else conds

        break_condition = torch.zeros(z.size(0), dtype=torch.bool)

        # create a batch of starting tokens (1)
        ys = (torch.ones(z.size(0), 1, requires_grad=True)
              * self.sos_id).to(dtype=torch.long, device=self.device)

        with torch.no_grad():
            for i in range(self.max_strlen - 1):
                # create a padding/nopeaking mask of target
                trg_mask = create_target_mask(ys, self.pad_id, c,
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
                end_condition = (next_word.to('cpu') == self.eos_id)
                break_condition = (break_condition | end_condition)

                # If all satisfies the break condition, then break the loop.
                if all(break_condition):
                    break
        return ys

    def sample_smiles(self, conds, z=None, transform=True):
        # toklen, toklen_gen 要改
        if transform:
            conds = self.scaler_transform(conds)
        conds = conds.to(self.device)

        if z is None:
            z, _ = self.sample_z_from_data(len(conds))
        
        toklen_gen, toklen, sequence = [], [], []
        
        if isinstance(z, torch.Tensor):
            toklen = [z.size(1) for i in range(len(z))]
            z = z.to(self.device)
            src_mask = (torch.ones(conds.size(0), 1, toklen[0]) !=
                0).to(self.device)  # (bs,1,nc+src_smi)
            sequence = self.decode(z, conds, src_mask)
            sequence = sequence.cpu().numpy()            
        else:
            for i in range(len(z)):
                z_in = z[i].to(self.device)
                props_in = conds[i].to(self.device)
                toklen.append(z_in.size(1))

                if z_in.dim() == 2:
                    z_in = torch.unsqueeze(z_in, 0)
                if props_in.dim() == 1:
                    props_in = torch.unsqueeze(props_in, 0)
                src_mask = (torch.ones(1, 1, toklen[-1]) !=
                    0).to(self.device)  # (bs,1,nc+src_smi)
                                
                seq = self.decode(z_in, props_in, src_mask)
                sequence.append(seq.cpu().numpy()[0])
                # print(f'{i} seq:', seq)

        smiles = []
        for i in range(conds.size(0)):
            smi = self.seq_to_smiles(sequence[i])
            smiles.append(smi)
            toklen_gen.append(len(smi))
            
        return smiles, toklen_gen, toklen


# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/132907dd272e2cc92e3c10e6c4e783a87ff8893d/transformer/Translator.py
# https://github.com/dreamgonfly/transformer-pytorch/blob/master/beam.py
class BeamSearch(Sampling):
    def __init__(self, predictor, kwargs):
        super().__init__(predictor, kwargs)
        self.k = 4

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
        trg_in = torch.LongTensor([[self.sos_id]]).to(self.device)
        src_mask = (torch.ones(1, 1, toklen) != 0).to(self.device)
        # trg_mask = create_target_mask(1, self.pad_id, conds, self.use_cond2dec)

        trg_mask = nopeak_mask(1, self.use_cond2dec, self.pad_id,
                               self.cond_dim).to(self.device)
        out_mol = self.predictor.predict(trg_in, z, conds, src_mask, trg_mask)

        # return the k elements with the highest probability
        probs, ix = out_mol[:, -1].data.topk(self.k)  # value/index (dim=(1,k))
        # the log-scale scores (dim=(1,k))
        log_scores = torch.Tensor([math.log(prob)
                                   for prob in probs.data[0]]).unsqueeze(0)
        # k outputs
        outputs = torch.zeros(self.k, self.max_strlen).long().to(self.device)
        outputs[:, 0] = self.sos_id
        outputs[:, 1] = ix[0]

        e_outputs = torch.zeros(self.k, z.size(-2), z.size(-1)).to(self.device)
        e_outputs[:, :] = z[0]

        return outputs, e_outputs, log_scores

    def beam_search(self, conds, toklen, z):
        # cond = cond.view(1, -1)

        # 维持三个变量，e_outputs,outputs,log_scores
        # outputs 维度(beam_size,max_len) e_outputs(beam_size,seq_len,d_model)

        outputs, e_outputs, log_scores = self.init_vars(conds, toklen, z)

        if len(conds) == 2:
            conds = (conds[0].repeat(self.k, 1), conds[1].repeat(self.k, 1))
        else:
            conds = conds.repeat(self.k, 1)

        ind = None
        src_mask = (torch.ones(1, 1, toklen) != 0)
        src_mask = src_mask.repeat(self.k, 1, 1).to(self.device)

        for i in range(2, self.max_strlen):
            trg_mask = nopeak_mask(i, self.use_cond2dec,
                                   self.pad_id, self.cond_dim)
            trg_mask = trg_mask.repeat(self.k, 1, 1).to(self.device)

            out_mol = self.predictor.predict(
                outputs[:, :i], e_outputs, conds, src_mask, trg_mask)
            outputs, log_scores = self.k_best_outputs(
                outputs, out_mol, log_scores, i)  # len(outputs) == k

            # Occurrences of end symbols for all input sentences. (index)
            ones = (outputs == self.eos_id).nonzero()
            sentence_lengths = torch.zeros(
                len(outputs), dtype=torch.long).to(self.device)

            for vec in ones:
                i = vec[0]  # i-th
                if sentence_lengths[i] == 0:  # First end symbol has not been found yet
                    # Position of first end symbol
                    sentence_lengths[i] = vec[1]

            # if (outputs == self.eos_id).cpu().numpy().argmax(axis=1).nonzero()[0].shape[0] == self.k:
            #     alpha = 0.7
            #     div = 1 / \
            #         (torch.tensor(((outputs == self.eos_id).cpu().numpy().argmax(axis=1))).type_as(
            #             log_scores) ** alpha)
            #     _, ind = torch.max(log_scores * div, 1)
            #     ind = ind.data[0]
            #     break

            num_finished_sentences = len(
                [s for s in sentence_lengths if s > 0])

            if num_finished_sentences == self.k:
                alpha = 0.7
                div = 1/(sentence_lengths.type_as(log_scores)**alpha)
                _, ind = torch.max(log_scores * div, 1)
                ind = ind.data[0]
                break

        if ind is None:
            length = (outputs[0] == self.eos_id).nonzero()[0]
            outs = ' '.join([self.trg_itos[tok]
                            for tok in outputs[0][1:length]])
            return outs
        else:
            length = (outputs[ind] == self.eos_id).nonzero()[0]
            return ' '.join([self.trg_itos[tok] for tok in outputs[ind][1:length]])

    def sample_smiles(self, dconds, z=None, transform=True):
        # handle properties
        if transform:
            dconds = self.transform_props(dconds)
        if dconds.dtype != torch.float32:
            dconds = torch.Tensor(dconds)
        assert dconds.dim() == 2
        # handle z
        if z is None:
            z, toklen = self.sample_z_from_data(n=dconds.size(0))
        else:
            toklen = [z[i].size(1) for i in range(len(z))]
        # sample smiles
        smiles, toklen_gen = [], []

        t = -time()
        for i in range(len(z)):
            z_in = z[i].to(self.device)
            props_in = dconds[i].to(self.device)

            if z_in.dim() == 2:
                z_in = torch.unsqueeze(z_in, 0)
            if props_in.dim() == 1:
                props_in = torch.unsqueeze(props_in, 0)

            smi = self.beam_search(props_in, toklen[i], z_in)
            toklen_gen.append(smi.count(" ") + 1)
            smiles.append("".join(smi).replace(" ", ""))
            # print(f"({i}) {t + time():.2f}", smiles[-1])
        return smiles, toklen_gen, toklen


# https://kikaben.com/transformers-evaluation-details/
class NewBeamSearch(Sampling):
    def __init__(self, predictor, kwargs):
        super().__init__(predictor, kwargs)
        self.alpha = 0.6
        self.k     = 4

    def sample_smiles(self, dconds, z=None, transform=True):
        # handle properties
        if transform:
            dconds = self.transform_props(dconds)
        if dconds.dtype != torch.float32:
            dconds = torch.Tensor(dconds)
        assert dconds.dim() == 2
        # handle z
        if z is None:
            z, _ = self.sample_z_from_data(n=dconds.size(0))

        # sample smiles
        smiles, toklen_gen, toklen = [], [], []

        t = -time()
        for i in range(len(z)):
            z_in = z[i].to(self.device)
            props_in = dconds[i].to(self.device)
            
            if z_in.dim() == 2:
                z_in = torch.unsqueeze(z_in, 0)
            if props_in.dim() == 1:
                props_in = torch.unsqueeze(props_in, 0)

            toklen.append(z_in.size(1))
            smi = self.beam_search(props_in, toklen[-1], z_in)
            smiles.append(smi)
            toklen_gen.append(len(smi))
            # print(f"({i}) {t + time():.2f}", smi)
        return smiles, toklen_gen, toklen
    
    def sequence_length_penalty(self, length):
        return ((5 + length) / (5 + 1)) ** self.alpha

    def beam_search(self, conds, toklen, z):
        # A batch of one input for Encoder
        # encoder_input = torch.Tensor([input_tokens])

        # Generate encoded features
        # with torch.no_grad():
        #     encoder_output = model.encode(encoder_input)
        encoder_output = z

        # Start with SOS
        decoder_input = torch.Tensor([[self.sos_id]]).long()
        decoder_input = decoder_input.to(self.device)
    
        # Maximum output size
        # max_output_length = encoder_input.shape[-1] + 50 # give some extra length

        src_mask = (torch.ones(1, 1, toklen) != 0).to(self.device)

        scores = torch.Tensor([0.]).to(self.device)
        vocab_size = len(self.trg_itos)
        
        for i in range(self.max_strlen):
            trg_mask = nopeak_mask(i+1, self.use_cond2dec,
                                   self.pad_id, self.cond_dim)
            trg_mask = trg_mask.to(self.device)
            
            if i > 0:
                trg_mask = trg_mask.repeat(self.k, 1, 1)
            
            # Decoder prediction
            logits = self.predictor.predict(decoder_input, encoder_output,
                                            conds, src_mask, trg_mask)

            # Softmax
            log_probs = torch.log_softmax(logits[:, -1], dim=1)
            log_probs = log_probs / self.sequence_length_penalty(i+1)

            # Set score to zero where EOS has been reached
            log_probs[decoder_input[:, -1]==self.eos_id, :] = 0
                                                
            # scores [beam_size, 1], log_probs [beam_size, vocab_size]
            scores = scores.unsqueeze(1) + log_probs

            # Flatten scores from [beams, vocab_size]
            # to [beams * vocab_size] to get top k, 
            # and reconstruct beam indices and token indices
            scores, indices = torch.topk(scores.reshape(-1), self.k)
            beam_indices  = torch.divide(indices, vocab_size, 
                                         rounding_mode='floor') # indices // vocab_size
            token_indices = torch.remainder(indices, vocab_size) # indices % vocab_size

            # Build the next decoder input
            next_decoder_input = []
            
            for beam_index, token_index in zip(beam_indices, token_indices):
                prev_decoder_input = decoder_input[beam_index]
                
                if prev_decoder_input[-1] == self.eos_id:
                    token_index = self.eos_id # once EOS, always EOS
                token_index = torch.tensor([token_index], device=self.device)
                token_index = token_index.long()
                next_decoder_input.append(torch.cat([prev_decoder_input, token_index]))
            decoder_input = torch.vstack(next_decoder_input).to(self.device)

            # If all beams are finished, exit
            if (decoder_input[:, -1] == self.eos_id).sum() == self.k:
                break

            # Encoder output expansion from the second time step to the beam size
            if i == 0:
                conds = conds.repeat(self.k, 1)
                src_mask = src_mask.repeat(self.k, 1, 1)
                encoder_output = encoder_output.expand(self.k, *encoder_output.shape[1:])

        # convert the top scored sequence to a list of text tokens
        decoder_output, _ = max(zip(decoder_input, scores), key=lambda x: x[1])
        decoder_output = decoder_output[1:].cpu().numpy() # remove SOS

        output_text_tokens = [self.trg_itos[i] for i in
                              decoder_output if i != self.eos_id] # remove EOS if exists
        return "".join(output_text_tokens)


def get_generator(predictor, decode_algo, kwargs):
    if decode_algo in ("greedy", "multinomial"):
        return MultinomialSearch(predictor, kwargs)
    elif decode_algo == "beam":
        return BeamSearch(predictor, kwargs)
    elif decode_algo == "newbeam":
        return NewBeamSearch(predictor, kwargs)
    else:
        exit(f"No such decoding algorithm: {decode_algo}")