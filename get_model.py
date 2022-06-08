import os
import math
import numpy as np
import random
import pandas as pd
import torch
import joblib
import dill as pickle

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, QED

# from torchtext.legacy import data
from torchtext import data
from torch.autograd import Variable
import torch.nn.functional as F

from Tokenize import moltokenize
import configuration.opts as opts
from models.VAETransformer import transformer
from Process import data_preparation as dp
import beam


def get_sampled_element(myCDF):
    a = np.random.uniform(0, 1)
    return np.argmax(myCDF>=a)-1


def run_sampling(xc, dxc, myPDF, myCDF, nRuns):
    sample_list = []
    X = np.zeros_like(myPDF, dtype=int)
    for k in np.arange(nRuns):
        idx = get_sampled_element(myCDF)
        sample_list.append(xc[idx] + dxc * np.random.normal() / 2)
        X[idx] += 1
    return np.array(sample_list).reshape(nRuns, 1), X/np.sum(X)


def tokenlen_gen_from_data_distribution(data, nBins, size):
    count_c, bins_c, = np.histogram(data, bins=nBins)
    myPDF = count_c / np.sum(count_c)
    dxc = np.diff(bins_c)[0]
    xc = bins_c[0:-1] + 0.5 * dxc

    myCDF = np.zeros_like(bins_c)
    myCDF[1:] = np.cumsum(myPDF)

    tokenlen_list, X = run_sampling(xc, dxc, myPDF, myCDF, size)

    # plot_distrib1(xc, myPDF)
    # plot_line(bins_c, myCDF, xc, myPDF)
    # plot_distrib3(xc, myPDF, X)

    return tokenlen_list


def gen_mol(cond, model, opt, SRC, TRG, toklen, z, scaler=None):
    # model.eval()
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

    cond = scaler.transform(cond)
    cond = Variable(torch.Tensor(cond))

    sentence = beam_search(cond, model, SRC, TRG, toklen, opt, z)
    return sentence


def k_best_outputs(outputs, out, log_scores, i, k):
    probs, ix = out[:, -1].data.topk(k)
    log_probs = torch.Tensor([math.log(p) for p in probs.data.view(-1)]).view(k, -1) + log_scores.transpose(0,1)
    k_probs, k_ix = log_probs.view(-1).topk(k)
    
    row = torch.div(k_ix, k, rounding_mode='floor')
    # row = k_ix // k
    col = k_ix % k

    outputs[:, :i] = outputs[row, :i]
    outputs[:, i] = ix[row, col]

    log_scores = k_probs.unsqueeze(0)
    return outputs, log_scores

    
def init_vars(cond, model, SRC, TRG, toklen, opt, z):
    init_tok = TRG.vocab.stoi['<sos>']

    src_mask = (torch.ones(1, 1, toklen) != 0)
    trg_mask = nopeak_mask(1, opt)

    trg_in = torch.LongTensor([[init_tok]])

    if opt.device == 0:
        trg_in, z, src_mask, trg_mask = trg_in.cuda(), z.cuda(), src_mask.cuda(), trg_mask.cuda()

    # print('z:', z)
    # print('trg_in:', trg_in)
    # print('cond:', cond)
    # print('src_mask:', src_mask)
    # print('trg_mask:', trg_mask)

    if opt.use_cond2dec == True:
        output_mol = model.out(model.decoder(trg_in, z, cond, src_mask, trg_mask))[:, 3:, :]
    else:
        output_mol = model.out(model.decoder(trg_in, z, cond, src_mask, trg_mask)[0])

    out_mol = F.softmax(output_mol, dim=-1)
    
    probs, ix = out_mol[:, -1].data.topk(opt.k)
    log_scores = torch.Tensor([math.log(prob) for prob in probs.data[0]]).unsqueeze(0)
    
    outputs = torch.zeros(opt.k, opt.max_strlen).long()
    if opt.device == 0:
        outputs = outputs.cuda()
    outputs[:, 0] = init_tok
    outputs[:, 1] = ix[0]

    e_outputs = torch.zeros(opt.k, z.size(-2), z.size(-1))
    if opt.device == 0:
        e_outputs = e_outputs.cuda()
    e_outputs[:, :] = z[0]

    return outputs, e_outputs, log_scores
    

def nopeak_mask(size, opt):
    np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    if opt.use_cond2dec == True:
        cond_mask = np.zeros((1, opt.cond_dim, opt.cond_dim))
        cond_mask_upperright = np.ones((1, opt.cond_dim, size))
        cond_mask_upperright[:, :, 0] = 0
        cond_mask_lowerleft = np.zeros((1, size, opt.cond_dim))
        upper_mask = np.concatenate([cond_mask, cond_mask_upperright], axis=2)
        lower_mask = np.concatenate([cond_mask_lowerleft, np_mask], axis=2)
        np_mask = np.concatenate([upper_mask, lower_mask], axis=1)
    np_mask = Variable(torch.from_numpy(np_mask) == 0)
    if opt.device == 0:
      np_mask = np_mask.cuda()
    return np_mask
    

def beam_search(cond, model, SRC, TRG, toklen, opt, z):
    if opt.device == 0:
        cond = cond.cuda()
    cond = cond.view(1, -1)

    outputs, e_outputs, log_scores = init_vars(cond, model, SRC, TRG, toklen, opt, z)

    cond = cond.repeat(opt.k, 1)
    src_mask = (torch.ones(1, 1, toklen) != 0)
    src_mask = src_mask.repeat(opt.k, 1, 1)
    if opt.device == 0:
        src_mask = src_mask.cuda()
    eos_tok = TRG.vocab.stoi['<eos>']

    ind = None

    for i in range(2, opt.max_strlen):
        trg_mask = nopeak_mask(i, opt)
        trg_mask = trg_mask.repeat(opt.k, 1, 1)
        # print("\n-----------------------------")
        # print("trg", np.shape(outputs[:,:i]), outputs[:,:i][0])
        # print("z", np.shape(e_outputs), e_outputs[0])
        # print("cond", np.shape(cond), cond[0])
        # print("src_mask", np.shape(src_mask), src_mask[0])
        # print("trg_mask", np.shape(trg_mask), trg_mask[0])
        # print("\n-----------------------------")
        if opt.use_cond2dec == True:
            output_mol = model.out(model.decoder(outputs[:,:i], e_outputs, cond, src_mask, trg_mask)[0])[:, 3:, :]
        else:
            output_mol = model.out(model.decoder(outputs[:,:i], e_outputs, cond, src_mask, trg_mask)[0])

        out_mol = F.softmax(output_mol, dim=-1)
    
        outputs, log_scores = k_best_outputs(outputs, out_mol, log_scores, i, opt.k)
        ones = (outputs==eos_tok).nonzero() # Occurrences of end symbols for all input sentences.
        sentence_lengths = torch.zeros(len(outputs), dtype=torch.long).cuda()
        for vec in ones:
            i = vec[0]
            if sentence_lengths[i]==0: # First end symbol has not been found yet
                sentence_lengths[i] = vec[1] # Position of first end symbol

        num_finished_sentences = len([s for s in sentence_lengths if s > 0])

        if num_finished_sentences == opt.k:
            alpha = 0.7
            div = 1/(sentence_lengths.type_as(log_scores)**alpha)
            _, ind = torch.max(log_scores * div, 1)
            ind = ind.data[0]
            break

    
    if ind is None:
        length = (outputs[0]==eos_tok).nonzero()[0]
        outs = ' '.join([TRG.vocab.itos[tok] for tok in outputs[0][1:length]])
        print(outs)
        return outs

    else:
        length = (outputs[ind]==eos_tok).nonzero()[0]
        return ' '.join([TRG.vocab.itos[tok] for tok in outputs[ind][1:length]])

##########################################################################

def fix_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model(opt, SRC, TRG):
    model = transformer.build_transformer(len(SRC.vocab), len(TRG.vocab),
                                          opt.N, opt.d_model, opt.d_ff, 
                                          opt.H, opt.latent_dim, opt.dropout, 
                                          opt.nconds, use_cond2dec=False, use_cond2lat=True)
    model.load_state_dict(torch.load(opt.molgct_model))
    model = model.to(opt.device)
    return model


def get_number_list(low, high, num=10):
    return np.linspace(low, high, num)


def get_rand_number(low, high):
    max_num = 10000
    return low + (random.randint(0, max_num) / float(max_num)) * (high - low)


def get_mol_prop(mol):
    logP_v, tPSA_v, QED_v = Descriptors.MolLogP(mol), Descriptors.TPSA(mol), QED.qed(mol)
    return logP_v, tPSA_v, QED_v



def sample_molecule(opt, toklen_data, model, SRC, TRG, scaler):
    toklen = int(tokenlen_gen_from_data_distribution(data=toklen_data, nBins=int(toklen_data.max() - toklen_data.min()), size=1)) + 3 # +3 due to cond2enc
    z = torch.Tensor(np.random.normal(size=(1, toklen, opt.latent_dim)))

    molecule = []
    for cond in opt.conds:
        molecule.append(gen_mol(cond + ',', model, opt, SRC, TRG, toklen, z, scaler))
    toklen_gen = molecule[0].count(" ") + 1
    molecule = ''.join(molecule).replace(" ", "")
    return molecule, toklen_gen, toklen




def inference():
    # fix_random_seed()

    """ Options """
    parser = opts.general_opts()
    opt = parser.parse_args()

    opt.device = 0
    opt.k = 4
    opt.molgct_model = 'molGCT/molgct.pt'
    opt.toklen_list = 'data/moses/toklen_list.csv'
    
    os.makedirs('molGCT/inference', exist_ok=True)

    total_num_each = 100 # 100
    num_samples_each = 5
    tol_failure_num = 500 # 300 # tolerated failure number

    logp_lb = 0.03
    logp_ub = 4.97
    tpsa_lb = 17.92
    tpsa_ub = 112.83
    qed_lb = 0.58
    qed_ub = 0.95

    logp_values = np.linspace(logp_lb, logp_ub, num=num_samples_each)
    tpsa_values = np.linspace(tpsa_lb, tpsa_ub, num=num_samples_each)
    qed_values = np.linspace(qed_lb, qed_ub, num=num_samples_each)

    robustScaler = joblib.load('molGCT/scaler.pkl')

    """ Tools """
    SRC, TRG = dp.create_fields(weights_path='molGCT')
    model = get_model(opt, SRC, TRG)
    toklen_data = pd.read_csv(opt.toklen_list)

    model.eval()
    RDLogger.DisableLog('rdApp.*') # disable error from RDlLo
    
    for logp in logp_values:
        for tpsa in tpsa_values:
            for qed in qed_values:
                prediction_file = "molGCT/inference/{:.2f}_{:.2f}_{:.2f}.txt".format(logp, tpsa, qed)
                with open(prediction_file, 'w', buffering=5) as sample_file:
                    sample_file.write("logp_t\ttpsa_t\tqed_t\tlogp_p\ttpsa_p\tqed_p\tsmiles\n")
                    print("\n[TARGET]>>> logp: {}\ttpsa: {}\tqed: {}".format(logp, tpsa, qed))
                    opt.conds = ['{}, {}, {}'.format(logp, tpsa, qed)]

                    sample_num = 0
                    valid_num = 0
                    valid_logp = np.zeros(total_num_each)
                    valid_tpsa = np.zeros(total_num_each)
                    valid_qed = np.zeros(total_num_each)
                    valid_smi = []

                    while valid_num < total_num_each:
                        sample_num += 1
                        smiles, _, _ = sample_molecule(opt, toklen_data, model, SRC, TRG, robustScaler)
                        molecule = Chem.MolFromSmiles(smiles)

                        if molecule is not None:
                            smiles = Chem.MolToSmiles(molecule)

                            logp_p, tpsa_p, qed_p = get_mol_prop(molecule)
                            valid_logp[valid_num] = logp_p
                            valid_tpsa[valid_num] = tpsa_p
                            valid_qed[valid_num] = qed_p
                            valid_smi.append(smiles)
                            valid_num += 1

                            line = "{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{}".format(
                                   logp, tpsa, qed, logp_p, tpsa_p, qed_p, smiles)
                            print(f'[sample{sample_num}]>>> '+line)
                            sample_file.write(line+'\n')
                        else:
                            print(f'[sample{sample_num}]>>> sample failed')

                        if sample_num == tol_failure_num:
                            break

                    mean_file = "molGCT/inference/mean_{:.2f}_{:.2f}_{:.2f}.txt".format(logp, tpsa, qed)
                    with open(mean_file, 'w') as mean_file:
                        logp_m = np.average(valid_logp[:valid_num])
                        tpsa_m = np.average(valid_tpsa[:valid_num])
                        qed_m = np.average(valid_qed[:valid_num])
                        
                        valid_smi = np.array(valid_smi)
                        unique_smi = np.unique(valid_smi)

                        if sample_num < tol_failure_num:
                            success_rate = valid_num / sample_num * 100.0
                        else:
                            success_rate = 0

                        header = "logp_t\ttpsa_t\tqed_t\tlogp_p\ttpsa_p\tqed_p\tvalid\tunique\tr_success"
                        line = "{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:d}\t{:d}\t{:.2f}".format(
                            logp, tpsa, qed, logp_m, tpsa_m, qed_m, len(valid_smi), len(unique_smi), success_rate)
                        
                        print(header)
                        print(line)
                        mean_file.write(header+'\n')
                        mean_file.write(line+'\n')


if __name__ == "__main__":
    inference()