import os
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from moses.metrics import SNNMetric

from Utils.dataset import get_dataloader


def rand_z(toklen, latent_dim, n=1):
    return torch.Tensor(np.random.normal(size=(n, toklen, latent_dim)))


def snnOf2MolGroups(molList1, molList2):
    return SNNMetric()(gen=molList1, ref=molList2)


def digits2smiles(vocab, digit_type='src'):
    def translate(digits):
        smi_list = [vocab.itos[d] for d in digits if d != vocab.stoi["<pad>"]]
        if digit_type == 'src':
            return ''.join(smi_list)
        elif digit_type == 'trg':
            return ''.join(smi_list[1:-1])
        else:
            exit(f'No digit_type: {digit_type}')
    return translate


def get_z_from_data(args, LOG, smiles_generator, fields, SRC, device, data_type="train", n=2):
    infile_path = os.path.join(
        args.data_path, 'aug', 'data_sim1.00', f"{data_type}_tiny.csv")
    oufile_path = os.path.join(
        args.data_path, 'aug', 'data_sim1.00', f"{data_type}_{n}.csv")

    LOG.info(f"Create new data path: {oufile_path}")
    df = pd.read_csv(infile_path)
    df = df.sample(n=n)
    df.to_csv(oufile_path, index=False)

    LOG.info(f"Get dataloader, #SMILES: {n}")
    dataloader, nbatches = get_dataloader(args.conditions, oufile_path,
                                          fields, SRC.vocab.stoi['<pad>'],
                                          args.max_strlen, device, batch_size=1)

    SRCs = []
    Zs = torch.empty((n, args.max_strlen, args.latent_dim),
                     dtype=torch.float32).to(device)
    Cs = torch.empty((n, 3), dtype=torch.float32).to(device)

    LOG.info(f"Generate Z from {n} source SMILES.")
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch.src.requires_grad, batch.econds.requires_grad = False, False
            z = smiles_generator.get_z_from_src(batch.src, batch.econds)[0]

            SRCs.append(batch.src[0])
            Zs[i], Cs[i] = z, batch.econds[0]

    LOG.info(f"Remove data path: {oufile_path}")
    # os.remove(oufile_path)
    return SRCs, Zs, Cs


def predict_smi_with_varying_z(args, LOG, smiles_generator, all_Zs, Cs, TRG, n_eachZ):
    mol_prev = []
    mol_curr = []
    mol_start = []
    
    for i, z in enumerate(all_Zs):
        c = Cs[0].view(1, -1)
        z = z.view(1, z.size(0), z.size(1))

        print(f"----- {i} -----")
        mol_list = []
        mol_prev = [m for m in mol_curr]

        for _ in range(n_eachZ):
            smiles = smiles_generator.sample_smiles(c, z, transform=False)[0]

            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                print(f"{smiles} -> O")
                mol_list.append(mol)
            else:
                print(f"{smiles} -> X")
            # smiles_list.append(smiles)
        
        mol_curr = [m for m in mol_list]

        if len(mol_start) == 0:
            mol_start = [m for m in mol_list]

        if len(mol_prev) == 0:
            mol_prev = [m for m in mol_list]
        else:
            snn_start = snnOf2MolGroups(mol_start, mol_curr)
            snn_prev = snnOf2MolGroups(mol_prev, mol_curr)
            print('snn_start:', snn_start)
            print('snn_prev:', snn_prev)




def varying_z_generate(args, smiles_generator, fields, device, logger, SRC, TRG):
    np.random.seed(0)
    LOG = logger(name="varying_z_generate", log_path="./test.log")

    #################### SETTINGS ####################
    n_splits = 6
    n_eachZ = 20
    z_dist_min, z_dist_max = 140, 155
    toklen = 40
    # 30 35 40
    ##################################################
    print("decode algo:", args.decode_algo)
    print("toklen:", toklen)

    def zs_dist(z1, z2): return torch.sqrt(torch.sum((z2-z1)**2)).item()

    if False:
        SRCs, Zs, Cs = get_z_from_data(
            args, LOG, smiles_generator, fields, SRC, device, n=2)
        # d2smi = digits2smiles(SRC.vocab, 'src')
        # SRCs = [d2smi(s) for s in SRCs]

        all_Zs = torch.empty((n_splits, Zs.size(1), Zs.size(2)),
                             dtype=torch.float32).to(device)
        del_z_each = (Zs[1] - Zs[0]) / (n_splits - 1)
        all_Zs[0] = Zs[0]

    else:
        z_begin = rand_z(toklen, args.latent_dim)
        z_end = rand_z(toklen, args.latent_dim)
        # while True:
        #     z_begin = rand_z(toklen, args.latent_dim)
        #     z_end = rand_z(toklen, args.latent_dim)
        #     dist = zs_dist(z_begin, z_end)
        #     print(dist)
        #     if z_dist_min < dist < z_dist_max:
        #         break
        dist = zs_dist(z_begin, z_end)
        print("dist:", dist)
        all_Zs = torch.empty(
            (n_splits, toklen, args.latent_dim)).to(device)
        del_z_each = ((z_end - z_begin) / (n_splits - 1)).to(device)
        all_Zs[0] = z_begin

        Cs = torch.FloatTensor([[0.4193,  1.2082, -1.7988],
                                [-0.0591, -0.4020, -0.6224]]).to(device)
        pass

    for i in range(1, n_splits):
        all_Zs[i] = all_Zs[i-1] + del_z_each



    predict_smi_with_varying_z(
        args, LOG, smiles_generator, all_Zs, Cs, TRG, n_eachZ)


