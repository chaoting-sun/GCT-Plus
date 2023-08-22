import math
import joblib
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn.functional as F



def get_sampled_element(myCDF):
    a = np.random.uniform(0, 1)
    return np.argmax(myCDF >= a)-1


def run_sampling(xc, dxc, myPDF, myCDF, nRuns):
    sample_list = []
    X = np.zeros_like(myPDF, dtype=int)
    for k in np.arange(nRuns):
        idx = get_sampled_element(myCDF)
        sample_list.append(xc[idx] + dxc * np.random.normal() / 2)
        X[idx] += 1
    return np.array(sample_list).reshape(nRuns, 1), X/np.sum(X)


def tokenlen_gen_from_data_distribution(data, nBins, size):
    # obtain the discrete distribution of all the token length
    # print('nbins:', nBins)
    count_c, bins_c = np.histogram(data, bins=nBins)
    # print('count_c:', count_c)
    # print('bins_c:', bins_c)

    myPDF = count_c / np.sum(count_c)
    # print('diff:', np.diff(bins_c))
    dxc = np.diff(bins_c)[0]
    xc = bins_c[0:-1] + 0.5 * dxc

    myCDF = np.zeros_like(bins_c)
    myCDF[1:] = np.cumsum(myPDF)

    tokenlen_list, X = run_sampling(xc, dxc, myPDF, myCDF, size)

    # plot_distrib1(xc, myPDF)
    # plot_line(bins_c, myCDF, xc, myPDF)
    # plot_distrib3(xc, myPDF, X)

    return tokenlen_list


def gen_mol(cond, model, opt, SRC, TRG, toklen, z, device, scaler=None):
    # model.eval()
    print('cond1:', cond)

    if scaler is None:
        scaler = joblib.load('molGCT/scaler.pkl')
    #robustScaler = pickle.load(open(f'{opt.load_weights}/scaler.pkl', "rb"))
    if opt.conds == 'm':
        cond = cond.reshape(1, -1)
    elif opt.conds == 's':
        cond = cond.reshape(1, -1)
    elif opt.conds == 'l':
        cond = cond.reshape(1, -1)
    else:
        cond = np.array(cond.split(',')[:-1]).reshape(1, -1)

    print('cond2:', cond)

    cond = scaler.transform(cond)
    cond = Variable(torch.Tensor(cond))

    sentence = beam_search(cond, model, SRC, TRG, toklen, opt, z, device)
    return sentence


def k_best_outputs(outputs, out, log_scores, i, k):
    probs, ix = out[:, -1].data.topk(k)
    log_probs = torch.Tensor(
        [math.log(p) for p in probs.data.view(-1)]).view(k, -1) + log_scores.transpose(0, 1)
    k_probs, k_ix = log_probs.view(-1).topk(k)

    row = torch.div(k_ix, k, rounding_mode='floor')
    # row = k_ix // k
    col = k_ix % k

    outputs[:, :i] = outputs[row, :i]
    outputs[:, i] = ix[row, col]

    log_scores = k_probs.unsqueeze(0)
    return outputs, log_scores


def init_vars(cond, model, SRC, TRG, toklen, opt, z, device=None):
    init_tok = TRG.vocab.stoi['<sos>']

    src_mask = (torch.ones(1, 1, toklen) != 0)
    trg_mask = nopeak_mask(1, opt.nconds, opt.use_cond2dec)

    trg_in = torch.LongTensor([[init_tok]])

    if device is not None:
        z = z.to(device)
        trg_in = trg_in.to(device)
        src_mask = src_mask.to(device)
        trg_mask = trg_mask.to(device)

    if opt.use_cond2dec == True:
        output_mol = model.out(model.decoder(trg_in, z, cond,
                                             src_mask, trg_mask))[:, 3:, :]
    else:
        output_mol = model.out(model.decoder(
            trg_in, z, cond, src_mask, trg_mask)[0])

    out_mol = F.softmax(output_mol, dim=-1)

    probs, ix = out_mol[:, -1].data.topk(opt.k)
    log_scores = torch.Tensor([math.log(prob)
                              for prob in probs.data[0]]).unsqueeze(0)

    outputs = torch.zeros(opt.k, opt.max_strlen).long()
    if device is not None:
        outputs = outputs.to(device)

    outputs[:, 0] = init_tok
    outputs[:, 1] = ix[0]

    e_outputs = torch.zeros(opt.k, z.size(-2), z.size(-1))
    if device is not None:
        e_outputs = e_outputs.to(device)
    e_outputs[:, :] = z[0]

    return outputs, e_outputs, log_scores


def beam_search(cond, model, SRC, TRG, toklen, opt, z, device=None):
    if device is not None:
        cond = cond.to(device)
    cond = cond.view(1, -1)

    # 维持三个变量，e_outputs,outputs,log_scores
    # outputs 维度(beam_size,max_len) e_outputs(beam_size,seq_len,d_model)
    outputs, e_outputs, log_scores = init_vars(
        cond, model, SRC, TRG, toklen, opt, z, device)

    cond = cond.repeat(opt.k, 1)
    src_mask = (torch.ones(1, 1, toklen) != 0)
    src_mask = src_mask.repeat(opt.k, 1, 1)
    if device is not None:
        src_mask = src_mask.to(device)
    eos_tok = TRG.vocab.stoi['<eos>']

    ind = None

    for i in range(2, opt.max_strlen):
        trg_mask = nopeak_mask(i, opt)
        trg_mask = trg_mask.repeat(opt.k, 1, 1)
        if opt.use_cond2dec == True:
            output_mol = model.out(model.decoder(
                outputs[:, :i], e_outputs, cond, src_mask, trg_mask)[0])[:, 3:, :]
        else:
            output_mol = model.out(model.decoder(
                outputs[:, :i], e_outputs, cond, src_mask, trg_mask)[0])

        out_mol = F.softmax(output_mol, dim=-1)

        outputs, log_scores = k_best_outputs(
            outputs, out_mol, log_scores, i, opt.k)
        # Occurrences of end symbols for all input sentences.
        ones = (outputs == eos_tok).nonzero()
        sentence_lengths = torch.zeros(len(outputs), dtype=torch.long)
        if device is not None:
            sentence_lengths = sentence_lengths.to(device)
        for vec in ones:
            i = vec[0]
            if sentence_lengths[i] == 0:  # First end symbol has not been found yet
                sentence_lengths[i] = vec[1]  # Position of first end symbol

        num_finished_sentences = len([s for s in sentence_lengths if s > 0])

        if num_finished_sentences == opt.k:
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


def sample_molecule(opt, toklen_data, model, SRC, TRG, scaler, device):
    toklen = int(tokenlen_gen_from_data_distribution(data=toklen_data, nBins=int(
        toklen_data.max() - toklen_data.min()), size=1)) + 3  # +3 due to cond2enc

    z = torch.Tensor(np.random.normal(size=(1, toklen, opt.latent_dim)))

    molecule = []
    for cond in opt.conds:
        molecule.append(gen_mol(cond + ',', model, opt,
                        SRC, TRG, toklen, z, device, scaler))
    toklen_gen = molecule[0].count(" ") + 1
    molecule = ''.join(molecule).replace(" ", "")
    return molecule, toklen_gen, toklen