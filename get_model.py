import os
import math
import random
import joblib
import argparse
import pandas as pd
import numpy as np
import pandas as pd
import dill as pickle

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, QED
from moses.metrics import metrics

import torch
# from torchtext.legacy import data
from torch.autograd import Variable
import torch.nn.functional as F

from Configuration.config import options
from Utils.field import smiles_fields
from Utils import allocate_gpu
# import beam
from Utils.seed import set_seed
from Inference.generate_mols import sample_molecule
from Model.build_model import build_mlpencoder, build_mlptransformer, build_transformer


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
    count_c, bins_c = np.histogram(data, bins=nBins)

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


def nopeak_mask(size, cond_dim, use_cond2dec=True):
    np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    if use_cond2dec == True:
        cond_mask = np.zeros((1, cond_dim, cond_dim))
        cond_mask_upperright = np.ones((1, cond_dim, size))
        cond_mask_upperright[:, :, 0] = 0
        cond_mask_lowerleft = np.zeros((1, size, cond_dim))
        upper_mask = np.concatenate([cond_mask, cond_mask_upperright], axis=2)
        lower_mask = np.concatenate([cond_mask_lowerleft, np_mask], axis=2)
        np_mask = np.concatenate([upper_mask, lower_mask], axis=1)
    np_mask = Variable(torch.from_numpy(np_mask) == 0)
    return np_mask


class ModelPrediction(object):
    def __init__(self, decoder, use_cond2dec):
        self.decoder = decoder
        self.use_cond2dec = use_cond2dec
    
    def predict(self, trg, e_outputs, conds, src_mask, trg_mask):
        if self.use_cond2dec == True:
            output_mol = self.decoder(trg, e_outputs, conds,
                                      src_mask, trg_mask)[:, 3:, :]
        else:
            output_mol = self.decoder(trg, e_outputs, conds, src_mask, trg_mask)
        return F.softmax(output_mol, dim=-1)


class ModelInference(object):
    def __init__(self, k, cond_dim, latent_dim, max_strlen, model, use_cond2dec):
        self.k = k
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
        # row = k_ix // k
        col = k_ix % self.k

        # dim=(k,max_str)
        outputs[:, :i] = outputs[row, :i]
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

        # z    (dim=(1,toklen,latent_dim))
        # z[0] (dim=(toklen,latent_dim))
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
        
        print((outputs[0] == eos_tok).nonzero())

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

        dconds = torch.Tensor(scaler.transform(conds)).to(device)
        rand_conds = torch.normal(mean=0, std=0.2, size=(1, self.cond_dim)).to(device)
        mconds = torch.cat((dconds - rand_conds, rand_conds), axis=1)

        molecule = []

        for i in range(len(dconds)):
            molecule.append(self.beam_search([mconds, dconds],
                            predictor, TRG, toklen, z, device))
        toklen_gen = molecule[0].count(" ") + 1
        molecule = ''.join(molecule).replace(" ", "")
        return molecule, toklen_gen, toklen


    def sample_molecule_from_decoder(self, conds, toklen_data, predictor, TRG, scaler, device):
        toklen = int(tokenlen_gen_from_data_distribution(data=toklen_data, nBins=int(
            toklen_data.max() - toklen_data.min()), size=1)) + 3  # +3 due to cond2enc

        z = torch.Tensor(np.random.normal(size=(1, toklen, self.latent_dim))).to(device)

        dconds = torch.Tensor(scaler.transform(conds)).to(device)

        molecule = []

        for i in range(len(dconds)):
            molecule.append(self.beam_search(dconds, predictor, TRG, toklen, z, device))
        toklen_gen = molecule[0].count(" ") + 1
        molecule = ''.join(molecule).replace(" ", "")
        return molecule, toklen_gen, toklen


##########################################################################

def get_model(args, SRC, TRG, model_type):
    model_path = os.path.join(args.save_directory, f'model_{args.epoch}.pt')

    if model_type == 'transformer':
        return build_transformer(len(SRC.vocab), len(TRG.vocab), args.N, args.d_model,
                                 args.d_ff, args.H, args.latent_dim, args.dropout, args.nconds, 
                                 args.use_cond2dec, args.use_cond2lat, file_path=model_path)
    elif model_type == 'mlp_transformer':
        return build_mlptransformer(len(SRC.vocab), len(TRG.vocab), args.N, args.d_model, 
                                    args.d_ff, args.H, args.latent_dim, args.dropout, args.nconds, 
                                    args.use_cond2dec, args.use_cond2lat, args.variational, 
                                    args.molgct_path, file_path=model_path)
    elif model_type == 'mlp_encoder':
        return build_mlpencoder(len(SRC.vocab), len(TRG.vocab), args.N, args.d_model, 
                                args.d_ff, args.H, args.latent_dim, args.dropout, args.nconds, 
                                args.use_cond2dec, args.use_cond2lat, args.variational, 
                                args.molgct_path, file_path=model_path)
    else:
        exit('ERROR - No Model Type:', model_type)


def get_number_list(low, high, num=10):
    return np.linspace(low, high, num)


def get_rand_number(low, high):
    max_num = 10000
    return low + (random.randint(0, max_num) / float(max_num)) * (high - low)


def get_mol_prop(mol):
    logP_v, tPSA_v, QED_v = Descriptors.MolLogP(
        mol), Descriptors.TPSA(mol), QED.qed(mol)
    return logP_v, tPSA_v, QED_v





total_num_each = 100  # 100
num_samples_each = 5
tol_failure_num = 500  # 300 # tolerated failure number

logp_lb = 0.03
logp_ub = 4.97
tpsa_lb = 17.92
tpsa_ub = 112.83
qed_lb = 0.58
qed_ub = 0.95

logp_values = np.linspace(logp_lb, logp_ub, num=num_samples_each)
tpsa_values = np.linspace(tpsa_lb, tpsa_ub, num=num_samples_each)
qed_values = np.linspace(qed_lb, qed_ub, num=num_samples_each)


def mlptf_test():
    set_seed(seed=21)
    """ Options """
    parser = argparse.ArgumentParser()
    parser = options(parser)
    opt = parser.parse_args()

    opt.save_directory = f'/fileserver-gamma/chaoting/ML/cvae-transformer/Experiment/mlptf_train_stage2_sim0.70_1'
    device = allocate_gpu()
    SRC, TRG = smiles_fields(smiles_field_path='molGCT')

    mlptf_path = os.path.join(
        opt.save_directory, f'model_{opt.starting_epoch-1}.pt')
    model = build_mlpencoder(len(SRC.vocab), len(TRG.vocab), opt.N, opt.d_model, opt.d_ff,
                             opt.H, opt.latent_dim, opt.dropout, opt.nconds, opt.use_cond2dec,
                             opt.use_cond2lat, opt.variational, opt.transferring_model_path, mlptf_path)
    model = model.to(device)

    logp = 2
    tpsa = 20
    qed = 0.6


def inference_test():
    set_seed(seed=21)
    """ Options """
    parser = argparse.ArgumentParser()
    parser = options(parser)
    args = parser.parse_args()

    device = allocate_gpu()

    k = 4 # beam search with k

    os.makedirs('molGCT/inference', exist_ok=True)

    scaler = joblib.load('molGCT/scaler.pkl')

    """ Tools """
    print('>>> GET SRC/TRG FEILDS')
    SRC, TRG = smiles_fields(smiles_field_path='molGCT')

    print('>>> GET MODEL')
    model = get_model(args, SRC, TRG, model_type='mlp_encoder').to(device)
    toklen_data = pd.read_csv(args.toklen_path)

    model.eval()
    RDLogger.DisableLog('rdApp.*') # disable error from RDlLo

    print('>>> SAMPLE MOLECULES')
    # logp = float(input('Enter logP:'))
    # tpsa = float(input('Enter tPSA:'))
    # qed = float(input('Enter QED:'))

    logp = 2.2
    tpsa = 85.3
    qed = 0.8

    conds = np.array([[logp, tpsa, qed]])

    # opt.conds = [f'{logp}, {tpsa}, {qed}']

    print('>>> BUILD MODELINFERENCE')
    inference = ModelInference(k, args.nconds, args.latent_dim,
                               args.max_strlen, model, args.use_cond2dec)

    print('>>> SAMPLE MOLECULES')
    predictor = ModelPrediction(model.decode, args.use_cond2dec)
    smiles, _, _ = inference.sample_molecule_from_decoder(conds, toklen_data, predictor, 
                                                          TRG, scaler, device)
    
    # predictor = ModelPrediction(model.mlp_decode, args.use_cond2dec)
    # smiles, _, _ = inference.sample_molecule(conds, toklen_data, predictor, 
    #                                          TRG, scaler, device)
    print('>>> CONVERT SMILES TO MOL')
    molecule = Chem.MolFromSmiles(smiles)
    print(smiles, molecule)


def inference():
    set_seed(seed=21)
    """ Options """
    parser = opts.general_opts()
    opt = parser.parse_args()

    device = allocate_gpu()
    opt.k = 4
    opt.molgct_model = 'molGCT/molgct.pt'
    opt.toklen_list = 'data/moses/toklen_list.csv'

    os.makedirs('molGCT/inference', exist_ok=True)

    robustScaler = joblib.load('molGCT/scaler.pkl')

    """ Tools """
    SRC, TRG = smiles_fields(weights_path='molGCT')
    model = get_model(opt, SRC, TRG)
    toklen_data = pd.read_csv(opt.toklen_list)

    print("successful in getting the model.")
    tf_name = [n for n, p in model.named_parameters()]
    print("transformer:\n", tf_name)

    model.eval()
    RDLogger.DisableLog('rdApp.*')  # disable error from RDlLo

    for logp in logp_values:
        for tpsa in tpsa_values:
            for qed in qed_values:

                sample_p = os.path.join('molGCT', 'inference',
                                        '{:.2f}_{:.2f}_{:.2f}.txt'.format(logp, tpsa, qed))

                """
                Sample molecules
                """

                sample_num = 0
                if not os.path.exists(sample_p):
                    with open(sample_p, 'w', buffering=5) as sample_file:
                        sample_file.write(
                            "logp_t\ttpsa_t\tqed_t\tlogp_p\ttpsa_p\tqed_p\tsmiles\n")
                        print("\n[TARGET]>>> logp: {:.2f}\ttpsa: {:.2f}\tqed: {:.2f}".format(
                            logp, tpsa, qed))
                        opt.conds = [f'{logp}, {tpsa}, {qed}']

                        valid_num = 0
                        valid_smi = []

                        while valid_num < total_num_each:
                            sample_num += 1
                            smiles, _, _ = sample_molecule(opt, toklen_data, model,
                                                           SRC, TRG, robustScaler, device)
                            molecule = Chem.MolFromSmiles(smiles)

                            if molecule is not None:
                                smiles = Chem.MolToSmiles(molecule)

                                logp_p, tpsa_p, qed_p = get_mol_prop(molecule)
                                valid_smi.append(smiles)
                                valid_num += 1

                                line = "{:.2f}\t{:.2f}\t{:.2f}\t"\
                                       "{:.2f}\t{:.2f}\t{:.2f}\t{}".format(
                                           logp, tpsa, qed, logp_p, tpsa_p, qed_p, smiles)
                                print(f'[sample{sample_num}]>>> '+line)
                                sample_file.write(line+'\n')
                            else:
                                print(f'[sample{sample_num}]>>> sample failed')

                            if sample_num == tol_failure_num:
                                break

                """
                Compute metrics
                """
                pred_df = pd.read_csv(sample_p, sep='\t')
                valid_num = len(pred_df)

                mean_p = os.path.join('molGCT', 'inference',
                                      'mean_{:.2f}_{:.2f}_{:.2f}.txt'.format(logp, tpsa, qed))
                if os.path.exists(mean_p):
                    continue
                print("- Compute metrics:", mean_p)

                with open(mean_p, 'w') as mean_f:
                    if len(pred_df) == total_num_each:
                        success_rate = len(pred_df) / total_num_each * 100.0
                    else:
                        success_rate = 0
                    intDiv = metrics.internal_diversity(pred_df['smiles'])
                    print("internal diversity:", intDiv)

                    logp_err = pred_df['logp_p'] - pred_df['logp_t']
                    tpsa_err = pred_df['tpsa_p'] - pred_df['tpsa_t']
                    qed_err = pred_df['qed_p'] - pred_df['qed_t']

                    header = 'logp_mae\ttpsa_mae\tqed_mae\t'\
                             'logp_mse\ttpsa_mse\tqed_mse\t'\
                             'logp_max\ttpsa_max\tqed_max\t'\
                             'logp_min\ttpsa_min\tqed_min\t'\
                             'valid\tunique\tsuccess\tdiversity\n'\

                    line = '{:.3f}\t{:.3f}\t{:.3f}\t'\
                           '{:.3f}\t{:.3f}\t{:.3f}\t'\
                           '{:.3f}\t{:.3f}\t{:.3f}\t'\
                           '{:.3f}\t{:.3f}\t{:.3f}\t'\
                           '{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n'.format(
                               # mean absolute error
                               logp_err.apply(np.abs).mean(),
                               tpsa_err.apply(np.abs).mean(),
                               qed_err.apply(np.abs).mean(),
                               # mean signed error
                               logp_err.mean(),
                               tpsa_err.mean(),
                               qed_err.mean(),
                               # maximum error
                               logp_err.max(),
                               tpsa_err.max(),
                               qed_err.max(),
                               # minimum error
                               logp_err.min(),
                               tpsa_err.min(),
                               qed_err.min(),
                               # others
                               len(pred_df),
                               len(np.unique(pred_df['smiles'])),
                               success_rate,
                               intDiv
                           )

                    mean_f.write(header)
                    mean_f.write(line)

    header = 'logp\ttpsa\tqed\t'\
             'logp_mae\ttpsa_mae\tqed_mae\t'\
             'logp_mse\ttpsa_mse\tqed_mse\t'\
             'logp_max\ttpsa_max\tqed_max\t'\
             'valid\tunique\tsucess\tdiversity\n'

    header_dict = {
        'logp': [],
        'tpsa': [],
        'qed': [],
        'logp_mae': [],
        'tpsa_mae': [],
        'qed_mae': [],
        'logp_mse': [],
        'tpsa_mse': [],
        'qed_mse': [],
        'logp_max': [],
        'tpsa_max': [],
        'qed_max': [],
        'logp_min': [],
        'tpsa_min': [],
        'qed_min': [],
        'valid': [],
        'unique': [],
        'success': [],
        'diversity': []
    }

    for logp in logp_values:
        for tpsa in tpsa_values:
            for qed in qed_values:
                print("- Metrics file:", mean_p)
                mean_p = os.path.join('molGCT', 'inference',
                                      'mean_{:.2f}_{:.2f}_{:.2f}.txt'.format(logp, tpsa, qed))
                mean_df = pd.read_csv(mean_p, sep='\t')

                header_dict['logp'].append(logp)
                header_dict['tpsa'].append(tpsa)
                header_dict['qed'].append(qed)

                header_dict['logp_mae'].append(mean_df['logp_mae'].tolist()[0])
                header_dict['tpsa_mae'].append(mean_df['tpsa_mae'].tolist()[0])
                header_dict['qed_mae'].append(mean_df['qed_mae'].tolist()[0])

                header_dict['logp_mse'].append(mean_df['logp_mse'].tolist()[0])
                header_dict['tpsa_mse'].append(mean_df['tpsa_mse'].tolist()[0])
                header_dict['qed_mse'].append(mean_df['qed_mse'].tolist()[0])

                header_dict['logp_max'].append(mean_df['logp_max'].tolist()[0])
                header_dict['tpsa_max'].append(mean_df['tpsa_max'].tolist()[0])
                header_dict['qed_max'].append(mean_df['qed_max'].tolist()[0])

                header_dict['logp_min'].append(mean_df['logp_min'].tolist()[0])
                header_dict['tpsa_min'].append(mean_df['tpsa_min'].tolist()[0])
                header_dict['qed_min'].append(mean_df['qed_min'].tolist()[0])

                header_dict['valid'].append(mean_df['valid'].tolist()[0])
                header_dict['unique'].append(mean_df['unique'].tolist()[0])
                header_dict['success'].append(mean_df['success'].tolist()[0])
                header_dict['diversity'].append(
                    mean_df['diversity'].tolist()[0])

    data_df = pd.DataFrame.from_dict(header_dict)
    data_df.to_csv('molGCT/inference/output.csv')


def intdiv():
    smiles_list = []
    for logp in logp_values:
        for tpsa in tpsa_values:
            for qed in qed_values:

                sample_p = os.path.join('molGCT', 'inference',
                                        '{:.2f}_{:.2f}_{:.2f}.txt'.format(logp, tpsa, qed))
                print("File path:", sample_p)
                """
                Sample molecules
                """
                df = pd.read_csv(sample_p, sep='\t')
                smiles_list.extend(df['smiles'].tolist())

    div = metrics.internal_diversity(smiles_list)
    print(div)


if __name__ == "__main__":
    # mlptf_test()
    # inference()
    # intdiv()
    inference_test()
