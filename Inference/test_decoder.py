import os
import numpy as np
import pandas as pd
from Inference.utils import prepare_generator
import torch
import random
from Model.build_model import get_generator
from Utils import DataloaderPreparation
from Utils.properties import get_property_fn, mols_to_props
from Utils.field import smi_to_id
from Utils.smiles import (
    get_mol,
    murcko_scaffold,
    tanimoto_similarity,
    plot_smiles_group,
    plot_smiles,
    murcko_scaffold_similarity
)

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from Utils.analysis import interpolation, dimension_reduction


property_peaks = [3.075,93.411,0.609]


def get_distance(v1, v2):
    return torch.sqrt(torch.sum((v2 - v1)**2)).item()


def create_dataset(smiles_list, property_list):
    property_fn = get_property_fn(property_list)
    mols = list(map(get_mol, smiles_list))
    murcko_sca = list(map(murcko_scaffold, mols))
    smi = pd.DataFrame({ 'src': smiles_list,
                         'src_scaffold': murcko_sca })
    props = pd.DataFrame({ f'src_{p}': list(map(fn, mols))
                           for p, fn in property_fn.items() })
    return pd.concat([smi, props], axis=1)


def sample_encoder_outputs(generator, dataloader, transform=True, n_samples=2000):
    for i, batch in enumerate(dataloader):
        print(f'batch: {i}')
        if i == n_samples:
            break

        _, mu, _ = generator.encode(batch, transform)
        
        if i == 0:
            dim = mu.size(-1)
            out_list = [[] for _ in range(dim)]
        
        for d in range(dim):
            out_list[d].append(mu[:,0,d].view(-1).cpu().numpy())

    sampled_list = []
    for d in range(dim):
        concatenated_out = np.concatenate(out_list[d], axis=0)
        if concatenated_out.shape[0] < n_samples:
            sampled_list.append(concatenated_out)
        else:
            sampled_out = np.random.choice(concatenated_out, n_samples,
                                           replace=False)
            sampled_list.append(sampled_out)
    return np.stack(sampled_list, axis=0).T


"""overall workflow

1. sample two specific molecules and their properties from a dataset
2. project the two molecules into latent spaces by the encoder
3. interpolate the latent spaces (and their corresponding properties)
4. decode each of the interpolated latent points into molecules
"""


"""parameters

1. number of interpolations
2. property constraints
"""


"""output folder x 1000

1. src1.png
2. src2.png
3. src.csv - src1 smiles/properties; src2 smiles/properties
4. prediction.csv - smiles/properties
5. prediction.png
"""

"""
同 src1, src2 下找各種 z 降低 metric
"""


# def sample_molecule_pair(df_data, property_list, same_scaffold=True,
#                          constraints=None):
#     if same_scaffold or constraints is not None:
#         while True:
#             filtered = df_data.copy()
#             if same_scaffold:
#                 rid = np.random.choice(len(df_data))
#                 sca = df_data.loc[rid, 'scaffold']
#                 filtered = filtered[df_data.scaffold == sca].reset_index(drop=True)
#                 if len(filtered) < 2:
#                     continue
                
#             if constraints is not None:
#                 is_valid = np.zeros((len(property_list),))
#                 filtered = filtered.sample(n=2).reset_index(drop=True)
#                 for i, (p, diff) in enumerate(constraints.items()):
#                     if abs(filtered.loc[0,p] - filtered.loc[1,p]) <= diff:
#                         is_valid[i] = 1
#                 if is_valid.sum() == len(is_valid):
#                     break
#             else:
#                 break
    
#     filtered = filtered.sample(n=2).reset_index(drop=True)
    
#     def process(idx):
#         row = filtered.iloc[[idx]]
#         row = row[['smiles', 'scaffold']+property_list]
#         row = row.rename(columns={ 'smiles': f'src_{idx}' })
#         row = row.rename(columns={ 'scaffold': f'scaffold_{idx}' })
#         row = row.rename(columns={ p: f'{p}_{idx}' for p in property_list })
#         return row.reset_index(drop=True)
#     row1 = process(0)
#     row2 = process(1)
    
#     return pd.concat([row1, row2], axis=1)
    
    # smiles = [filtered.loc[i, 'smiles'] for i in range(2)]
    # scaffolds = [filtered.loc[i, 'scaffold'] for i in range(2)]
    # conds = [filtered.loc[i, property_list] for i in range(2)]
    # return smiles, scaffolds, conds


def sample_molecule_pairs(df, n_pairs, property_list, same_scaffold=True,
                          property_constraint=None, similarity_threshold=0.5):
    gathered_set = set()

    while len(gathered_set) < n_pairs:
        dfcopy = df.copy()

        # scaffold constraint

        if same_scaffold:
            rand_id = np.random.choice(len(dfcopy))
            sca = dfcopy.loc[rand_id, 'scaffold']
            dfcopy = dfcopy[dfcopy.scaffold == sca]
        
        if len(dfcopy) < 2:
            continue
        
        found = False
        n_test_limit = 20

        for i in range(n_test_limit):
            pair = dfcopy.sample(n=2)

            # similarity constraint

            if tanimoto_similarity(pair['smiles'].iloc[0], pair['smiles'].iloc[1]) > similarity_threshold:
                continue

            # property constraint

            if len(property_list) > 0 and property_constraint is not None:
                is_valid = np.zeros((len(property_list),), dtype=bool)
                for i, (p, delp) in enumerate(property_constraint.items()):
                    if abs(pair[p].iloc[0] - pair[p].iloc[1]) <= delp:
                        is_valid[i] = True
                if np.all(is_valid):
                    found = True
            else:
                found = True

            if found:
                gathered_set.add(tuple(sorted((pair.index[0], pair.index[1]))))
                break

    final_pairs = []
    for pair_no in list(gathered_set):
        row_pair = []
        random.shuffle(list(pair_no))
        for i, no in enumerate(pair_no):
            pair = df.loc[[no], ['smiles', 'scaffold']+property_list]
            row = pair.rename(columns={**{ 'smiles': f'src_{i}',
                                           'scaffold': f'scaffold_{i}'},
                                       **{ p: f'{p}_{i}' for p in property_list }})
            row = row.reset_index(drop=True)
            row_pair.append(row)
        row_pair = pd.concat(row_pair, axis=1)
        final_pairs.append(row_pair)

    final_pairs = pd.concat(final_pairs, axis=0).reset_index(drop=True)
    print(final_pairs)
    return final_pairs
   

# def sample_molecule_pair(df, property_list, same_scaffold=True, constraint=None):
#     dfcopy = df.copy()

#     if same_scaffold:
#         rand_id = np.random.choice(len(dfcopy))
#         sca = dfcopy.loc[rand_id, 'scaffold']
#         dfcopy = dfcopy[dfcopy.scaffold == sca].reset_index(drop=True)

#     if len(dfcopy) < 2:
#         return sample_molecule_pair(df, property_list, same_scaffold, constraint)

#     while True:
#         pair = dfcopy.sample(n=2).reset_index(drop=True)
#         if constraint is not None:
#             is_valid = np.zeros((len(property_list),), dtype=bool)
#             for i, (p, delp) in enumerate(constraint.items()):
#                 if abs(pair.loc[0,p] - pair.loc[1,p]) <= delp:
#                     is_valid[i] = True
#             if np.all(is_valid):
#                 break
#         else:
#             break

#     pair = pair[['smiles', 'scaffold']+property_list]
#     row1 = pair.iloc[[0]].rename(
#         columns={**{ 'smiles': 'src_0',
#                     'scaffold': 'scaffold_0'},
#                  **{ p: f'{p}_0' for p 
#                     in property_list }}).reset_index(drop=True)
#     row2 = pair.iloc[[1]].rename(
#         columns={**{ 'smiles': 'src_1',
#                     'scaffold': 'scaffold_1'},
#                  **{ p: f'{p}_1' for p
#                     in property_list }}).reset_index(drop=True)
    
#     return pd.concat([row1, row2], axis=1)


def interpolate_latent_spaces(lat1, lat2, cond1, cond2, n, method='slerp'):
    interpolated_z = []
    interpolated_conds = []
    
    for alpha in np.linspace(0, 1, n, endpoint=True):
        toklen = int(interpolation['lerp'](lat1.size(1), lat2.size(1), alpha))

        z_info = interpolation[method](lat1[0,0,:], lat2[0,0,:], alpha)
        interpolated_z.append(z_info.unsqueeze(0).repeat(1, toklen, 1))
        interpolated_conds.append(interpolation['lerp'](cond1, cond2, alpha=0.5))
    return interpolated_z, interpolated_conds


def line_plot(df, save_path, y_ticks):
    df.index = np.arange(1, len(df)+1)
    ax = df.plot(figsize=(6.5, 5.5), use_index=True,
                 kind='line', color='black',
                 legend=False, alpha=0.5)
    ax.set_xlabel('Interpolation step', fontsize=18)
    ax.set_ylabel(y_ticks, fontsize=18)
    ax.set_yticks(np.arange(0, 1+0.2, 0.2))
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    plt.savefig(save_path)


def approximate_mu_or_logvar(v, toklen):
    """
    based on the 2 dimensional mu or logvar
    (v_toklen, lat_dim) and approximate a
    new mu or logvar of given toklen
    """
    mu = v.mean(dim=0)
    std = v.std(dim=0)
    app = torch.empty((toklen, v.size(1)))

    for d in range(v.size(1)):
        app[:, d] = app[:, d].normal_(mean=mu[d], std=std[d])
    return app
    

def interpolate_encoder_output(v1, v2, toklen, alpha,
                               interpolate_fn):
    app1 = approximate_mu_or_logvar(v1[0], toklen)
    app2 = approximate_mu_or_logvar(v2[0], toklen)
    
    new_app = torch.empty((1, toklen, v1.size(2)))
    for i in range(toklen):
        new_app[0, i, :] = interpolate_fn(app1[i, :],
                                          app2[i, :],
                                          alpha)
    return new_app


def topk(arr, k):
    arr = np.array(arr)
    ids = np.argsort(arr)[::-1][:k]
    return ids


def get_smoothness_by_simprev(tasim_prev, threshold=0.50):
    return len([t for t in tasim_prev
                if t >= threshold]) / len(tasim_prev)


import itertools

def get_trg_prop(property_list):
    trg_prop = {
        'logP': [1.0, 2.0, 3.0],
        'tPSA': [30.0, 60.0, 90.0],
        'QED' : [0.6, 0.725, 0.85],
    }
    prop_set = (trg_prop[p] for p in property_list)
    prop_comb = list(itertools.product(*prop_set))
    return [list(c) for c in prop_comb]


def get_smoothness_by_simstart(tasim_start_from_begin, tasim_start_from_end):
    """get smoothness from a list of TaSim(current, previous)"""
    delv_begin = np.array([tasim_start_from_begin[i] - tasim_start_from_begin[i-1]
                          for i in range(1, len(tasim_start_from_begin))])
    delv_end = np.array([tasim_start_from_end[i] - tasim_start_from_end[i-1]
                        for i in range(1, len(tasim_start_from_end))])
    return 1- (delv_begin.std(ddof=1)*delv_end.std(ddof=1))**0.5



class SmilesInterpolation:
    def __init__(self, args, generator, data_src='test_scaffolds'):
        self.n_jobs = args.n_jobs
        self.model_type = args.model_type
        self.use_scaffold = args.use_scaffold
        self.property_list = args.property_list
        self.property_fn = get_property_fn(args.property_list)
        
        self.generator = generator
        self.interpolate_fn = interpolation['slerp']
        self.data_src = data_src
        self.n_interpolates = 8

    def wrap_encoder_input(self, smiles_list, transform,
                           scaffold_list, econds):
        kwargs = { 'smiles_list': smiles_list }
        if self.use_scaffold:
            kwargs['scaffold_list'] = scaffold_list
        if len(self.property_list) > 0:
            kwargs['econds'] = econds
            kwargs['transform'] = transform
        return kwargs

    def wrap_decoder_input(self, zs, transform,
                           sca_smi=None, dconds=None):
        kwargs = { 'zs': zs }
        if self.use_scaffold:
            kwargs['sca_smi'] = sca_smi
        if len(self.property_list) > 0:
            kwargs['dconds'] = [dconds]
            kwargs['transform'] = [transform]
        return kwargs
    
    
    def generate_with_different_scaffold(self, pairs, save_folder,
                                         transform=True):
        rec = { 'esmi': [], 'dsmi': [], 'gsmi': [] }
        
        for p in range(len(pairs)):
            print('p:', p)
            
            esmi = pairs.loc[p, 'src_0']
            dsmi = pairs.loc[p, 'src_1']            
            encoder_scaffold = pairs.loc[p, 'scaffold_0']
            decoder_scaffold = pairs.loc[p, 'scaffold_1']
            econds = pairs.loc[p, [f'{prop}_0' for prop in self.property_list]].to_numpy()
            dconds = pairs.loc[p, [f'{prop}_1' for prop in self.property_list]].to_numpy()

            kwargs = self.wrap_encoder_input(smiles_list=[esmi],
                                             transform=transform,
                                             scaffold_list=[encoder_scaffold],
                                             econds=[econds])
            _, mu, logvar = self.generator.encode_smiles(**kwargs)

            std_curr = 0
            
            while True:
                print(std_curr)
                std = torch.exp(0.5*logvar)
                eps = torch.randn_like(mu).normal_(mean=0, std=std_curr)
                new_z = eps.mul(std).add(mu)
    
                kwargs = self.wrap_decoder_input(zs=new_z,
                                                transform=transform,
                                                sca_smi=decoder_scaffold,
                                                dconds=dconds)
                gsmi, *_ = self.generator.sample_smiles(**kwargs)
                gsmi = gsmi[0]
                
                mol = get_mol(gsmi)
                if mol:
                    break
                else:
                    std_curr += 0.02

            rec['esmi'].append(esmi)
            rec['dsmi'].append(dsmi)
            rec['gsmi'].append(gsmi)

            plot_smiles_group([esmi, dsmi, gsmi], os.path.join(save_folder, f'{p}.png'))
            if p == 10:
                break


  
        
            
        
    
    def generate_interpolated_smiles(self, pairs, save_folder,
                                     trg_conds=None, transform=True):
        suffix = '' if trg_conds is None else '_'+'-'.join(map(str, list(trg_conds)))

        for p in range(len(pairs)):
            print(f'#pair = {p}')

            smiles_path = os.path.join(save_folder, f'prediction{p}{suffix}.csv')
            figure_path = os.path.join(save_folder, f'prediction{p}{suffix}.png')
            print(smiles_path)

            if os.path.exists(smiles_path):
                continue
            
            # input information

            src0 = pairs.loc[p, 'src_0']
            src1 = pairs.loc[p, 'src_1']
            scaffold0 = pairs.loc[p, 'scaffold_0']
            scaffold1 = pairs.loc[p, 'scaffold_1']
            cond0 = pairs.loc[p, [f'{prop}_0' for prop in self.property_list]].to_numpy()
            cond1 = pairs.loc[p, [f'{prop}_1' for prop in self.property_list]].to_numpy()

            # encode smiles of both molecules

            kwargs0 = self.wrap_encoder_input(smiles_list=[src0],
                                              transform=transform,
                                              scaffold_list=[scaffold0],
                                              econds=[cond0])
            kwargs1 = self.wrap_encoder_input(smiles_list=[src1],
                                              transform=transform,
                                              scaffold_list=[scaffold1],
                                              econds=[cond1])
            
            _, mu0, logvar0 = self.generator.encode_smiles(**kwargs0)
            _, mu1, logvar1 = self.generator.encode_smiles(**kwargs1)

            toklen0 = mu0.size(1)
            toklen1 = mu1.size(1)

            smi_records = []
            mol_records = []

            for alpha in np.linspace(0, 1, self.n_interpolates+2,
                                     endpoint=True):
                # only interpolate the space between the two

                if alpha == 0 or alpha == 1:
                    continue

                # interpolate the properties and token length by lerp

                if trg_conds is None and len(self.property_list) > 0:
                    trg_conds = cond0*(1-alpha) + cond1*alpha

                # decode the interpolated latent space into smiles

                n_counts = 0
                std_current = 0.000
                is_valid = False

                # interpolate mean and logvar from encoder

                while not is_valid:
                    if toklen0 > toklen1:
                        toklen0, toklen1 = toklen1, toklen0
                    toklen = np.random.choice(np.arange(toklen0, toklen1+1))
                    # toklen =  int(toklen0*(1-alpha) + toklen1*alpha)

                    ip_mu = interpolate_encoder_output(
                        mu0, mu1, toklen, alpha,
                        self.interpolate_fn)
                    ip_logvar = interpolate_encoder_output(
                        logvar0, logvar1, toklen, alpha,
                        self.interpolate_fn)

                    if self.model_type == 'ctf':
                        ip_std = torch.empty_like(ip_logvar).normal_(mean=0, std=std_current)
                    else:
                        ip_std = torch.exp(0.5*ip_logvar)

                    eps = torch.randn_like(ip_mu).normal_(mean=0, std=std_current)
                    new_z = eps.mul(ip_std).add(ip_mu)

                    # sample smiles

                    kwargs = self.wrap_decoder_input(zs=new_z,
                                                     transform=transform,
                                                     sca_smi=scaffold0,
                                                     dconds=trg_conds)
                    smiles, *_ = self.generator.sample_smiles(**kwargs)

                    mol = get_mol(smiles[0])

                    if mol:
                        print(smiles[0])
                        smi_records.append(smiles[0])
                        mol_records.append(mol)
                        is_valid = True
                                    
                    if (n_counts + 1) % 2 == 0:
                        std_current += 0.005
                    print(std_current)

                    if self.model_type == 'ctf':
                        if std_current >= 2.0:
                            exit('cannot sample valid smiles!')
                    else:
                        if std_current >= 1.0:
                            exit('cannot sample valid smiles!')

                    n_counts += 1
            
            smi_records = [src0] + smi_records + [src1]
            valid_prop = mols_to_props(mol_records, self.property_fn, n_jobs=self.n_jobs)
            df = pd.concat([pd.DataFrame({ 'smiles': smi_records }), valid_prop], axis=1)
            df.to_csv(smiles_path)

            plot_smiles_group(smi_records, figure_path, n_per_mol=5)

        # compute and plot tanimoto similarity

        tasim_start_rec = {}
        tasim_prev_rec = {}
        smoothness_rec = { 'start': [], 'prev': [] }

        best_smoothness_start = 0
        best_tasim_start = None
        best_tasim_prev = None
        best_p = 0

        for p in range(len(pairs)):
            tasim_prev = []
            tasim_start_from_begin = []
            tasim_start_from_end = []

            df = pd.read_csv(os.path.join(save_folder, f'prediction{p}{suffix}.csv'))
            # df = pd.read_csv(os.path.join(save_folder, f'{data_src}_prediction{p}_c{ci}.csv'))

            for i in range(len(df)):
                first_smi = df.loc[0, 'smiles']
                last_smi = df.loc[len(df)-1, 'smiles']
                curr_smi = df.loc[i, 'smiles']

                tasim_start_from_begin.append(tanimoto_similarity(first_smi, curr_smi))
                tasim_start_from_end.append(tanimoto_similarity(last_smi, curr_smi))
                if i > 0:
                    prev_smi = df.loc[i-1, 'smiles']
                    tasim_prev.append(tanimoto_similarity(prev_smi, curr_smi))
                else:
                    tasim_prev.append(None)
            
            tasim_start_rec[p] = tasim_start_from_begin
            tasim_prev_rec[p] = tasim_prev

            smoothness_start = get_smoothness_by_simstart(tasim_start_from_begin, tasim_start_from_end)
            smoothness_prev = get_smoothness_by_simprev(tasim_prev[1:])

            smoothness_rec['start'].append(smoothness_start)
            smoothness_rec['prev'].append(smoothness_prev)

            if smoothness_start > best_smoothness_start:
                best_tasim_start = tasim_start_from_begin
                best_tasim_prev = tasim_prev
                best_p = p
        
        smoothness_rec = pd.DataFrame(smoothness_rec)
        smoothness_rec.to_csv(os.path.join(save_folder, f'smoothness{suffix}.csv'))

        print('plot the k best smiles list of interpolation...')    

        best_ids = topk(smoothness_rec['start'], k=8)
        best_smiles_list = []
        for p in best_ids:
            df = pd.read_csv(os.path.join(save_folder, f'prediction{p}{suffix}.csv'))
            best_smiles_list.append(df['smiles'].tolist())
        best_smiles_list = list(zip(*best_smiles_list))
        best_smiles_list = [item for sub_rec in best_smiles_list for item in sub_rec]
        plot_smiles_group(best_smiles_list, os.path.join(save_folder, f'prediction{suffix}.png'),
                        n_per_mol=8)

        print('plot the best smiles list of interpolation...')

        best_tasim_start = pd.DataFrame({ 'TaSim(start, current)': best_tasim_start })
        best_tasim_prev = pd.DataFrame({ 'TaSim(start, previous)': best_tasim_prev })

        line_plot(best_tasim_start, y_ticks='TanSim(start, current)',
                  save_path=os.path.join(save_folder, f'best-tasim-start_{best_p}.png'),
                  )    
        line_plot(best_tasim_prev, y_ticks='TanSim(start, previous)',
                  save_path=os.path.join(save_folder, f'best-tasim-previous_{best_p}.png'),
                  )    

        print('plot the line plot TaSim(start, current) and TaSim(previous, current)...')

        tasim_start_rec = pd.DataFrame(tasim_start_rec)
        tasim_prev_rec = pd.DataFrame(tasim_prev_rec)

        line_plot(tasim_start_rec, y_ticks='TanSim(start, current)',
                  save_path=os.path.join(save_folder, f'tasim-start{suffix}.png')
                  )
        line_plot(tasim_prev_rec, y_ticks='TanSim(start, previous)',
                  save_path=os.path.join(save_folder, f'tasim-prev{suffix}.png')
                  )


def generate_smiles_by_interpolation(args, generator, df_dataset, SRC, TRG,
                                     device, n_tests=100, n_interpolates=8,
                                     n_plots=8):
    ## TASK1: interpolation of random two molecules

    # Description:

    # As a Variational Autoencoder (VAE), the latent space should be smooth, allowing us
    # to interpolate between two latent spaces from two different molecules to search
    # for intermediate molecules.

    # We randomly sample 100 pairs of molecules from the dataset, encode them to obtain
    # their latent spaces, interpolate them by several equal-spacing points, and decode
    # the interpolated points to generate new molecules. We expect the Tanimoto similarity
    # between the first molecule in each pair and the generated molecules to decrease as
    # the latent space moves farther from the original molecule.
    
    # settings

    data_src = 'test_scaffolds' # the data source
    n_pairs = 100               # number of the test pairs
    slerp_or_lerp = 'slerp'     # interpolation method
    # new_conds = np.array([2, 58, 0.85])

    # constraints = { 'logP': 3, 'tPSA': 20, 'QED': 0.4 } # the property constraints of pairs

    # file path

    main_folder = os.path.join(args.infer_path, args.benchmark, f'test_decoder_ip13')
    os.makedirs(main_folder, exist_ok=True)

    data_path = os.path.join(main_folder, f"{data_src}_pair{'-scaffold' if args.use_scaffold else ''}.csv")
    save_folder = os.path.join(main_folder, args.model_name, slerp_or_lerp)
    os.makedirs(save_folder, exist_ok=True)

    # prepare and save pair data

    df_data = pd.read_csv(f'/fileserver-gamma/chaoting/ML/dataset/moses/raw/{data_src}.csv')
    if not os.path.exists(data_path):
        pairs = sample_molecule_pairs(df_data, n_pairs, args.property_list,
                                        args.use_scaffold, property_constraint=None,
                                        similarity_threshold=0.5)
        pairs.to_csv(data_path)
    pairs = pd.read_csv(data_path, index_col=[0])

    # for ci, new_conds in enumerate(trg_props):
    #     new_conds = np.array(new_conds)

    for p in range(n_pairs):
        smiles_path = os.path.join(save_folder, f'{data_src}_prediction{p}_c{ci}.csv')
        figure_path = os.path.join(save_folder, f'{data_src}_prediction{p}_c{ci}.png')
        # smiles_path = os.path.join(save_folder, f'{data_src}_prediction{p}.csv')
        # figure_path = os.path.join(save_folder, f'{data_src}_prediction{p}.png')

        if os.path.exists(smiles_path):
            continue

        print(f'#pair = {p}')

        src0 = pairs.loc[p, 'src_0']
        src1 = pairs.loc[p, 'src_1']
        scaffold0 = pairs.loc[p, 'scaffold_0']
        scaffold1 = pairs.loc[p, 'scaffold_1']

        if len(args.property_list) > 0:
            cond0 = pairs.loc[p, [f'{prop}_0' for prop in args.property_list]].to_numpy()
            cond1 = pairs.loc[p, [f'{prop}_1' for prop in args.property_list]].to_numpy()

        if len(args.property_list) > 0:
            if args.use_scaffold:
                _, mu0, logvar0 = generator.encode_smiles(smiles_list=[src0],
                                                        scaffold_list=[scaffold0],
                                                        econds=[cond0],
                                                        transform=False
                                                        )
                _, mu1, logvar1 = generator.encode_smiles(smiles_list=[src1],
                                                        scaffold_list=[scaffold1],
                                                        econds=[cond1],
                                                        transform=False
                                                        )

            else:
                _, mu0, logvar0 = generator.encode_smiles(smiles_list=[src0],
                                                        econds=[cond0],
                                                        transform=False
                                                        )
                _, mu1, logvar1 = generator.encode_smiles(smiles_list=[src1],
                                                        econds=[cond1],
                                                        transform=False
                                                        )
        else:
            _, mu0, logvar0 = generator.encode_smiles(smiles_list=[src0])
            _, mu1, logvar1 = generator.encode_smiles(smiles_list=[src1])

        toklen0 = mu0.size(1)
        toklen1 = mu1.size(1)

        smi_records = []
        mol_records = []
        interpolate_fn = interpolation[slerp_or_lerp]
        interpolate_precision = 'high'

        property_fn = get_property_fn(args.property_list)

        if args.use_scaffold:
            src_ids = [TRG.vocab.stoi[e] for e in TRG.tokenize(scaffold0)]

        for alpha in np.linspace(0, 1, n_interpolates+2,
                                endpoint=True):
            # only interpolate the space between the two

            if alpha == 0 or alpha == 1:
                continue

            # interpolate the properties and token length by lerp

            if new_conds is None and len(args.property_list) > 0:
                new_conds = cond0*(1-alpha) + cond1*alpha

            # decode the interpolated latent space into smiles

            n_counts = 0
            std_current = 0.000

            # interpolate mean and logvar from encoder

            while True:
                toklen = np.random.choice(np.arange(toklen0, toklen1+1))
                # toklen =  int(toklen0*(1-alpha) + toklen1*alpha)

                if interpolate_precision == 'high':
                    ip_mu = interpolate_encoder_output(mu0, mu1, toklen, alpha,
                                                        interpolate_fn)
                    ip_logvar = interpolate_encoder_output(logvar0, logvar1, toklen, alpha,
                                                            interpolate_fn)

                elif interpolate_precision == 'low':
                    ip_mu = interpolate_fn(mu0[0,0,:], mu1[0,0,:],
                                        alpha).repeat(1, toklen, 1)
                    ip_logvar = interpolate_fn(logvar0[0,0,:], logvar1[0,0,:],
                                            alpha).repeat(1, toklen, 1)

                if args.model_type == 'ctf':
                    ip_std = torch.empty_like(ip_logvar).normal_(mean=0, std=std_current)
                else:
                    ip_std = torch.exp(0.5*ip_logvar)

                eps = torch.randn_like(ip_mu).normal_(mean=0, std=std_current)
                new_z = eps.mul(ip_std).add(ip_mu)

                # sample smiles

                if len(args.property_list) > 0:
                    if args.use_scaffold:
                        smiles, *_ = generator.sample_smiles(prop=[new_conds],
                                                            src_ids=src_ids,
                                                            zs=new_z,
                                                            transform=True)
                    else:
                        smiles, *_ = generator.sample_smiles(prop=[new_conds],
                                                            zs=new_z,
                                                            transform=True)
                else:
                    smiles, *_ = generator.sample_smiles(zs=new_z)


                mol = get_mol(smiles[0])

                if mol:
                    print(smiles[0])
                    smi_records.append(smiles[0])
                    mol_records.append(mol)
                    break
                                
                if (n_counts + 1) % 2 == 0:
                    std_current += 0.005
                print(std_current)

                if args.model_type == 'ctf':
                    if std_current >= 2.0:
                        exit('cannot sample valid smiles!')
                else:
                    if std_current >= 1.0:
                        exit('cannot sample valid smiles!')

                n_counts += 1
        
        smi_records = [src0] + smi_records + [src1]
        valid_prop = mols_to_props(mol_records, property_fn, n_jobs=args.n_jobs)
        df = pd.concat([pd.DataFrame({ 'smiles': smi_records }), valid_prop], axis=1)
        df.to_csv(smiles_path)

        plot_smiles_group(smi_records, figure_path, n_per_mol=5)

    # compute and plot tanimoto similarity

    tasim_start_rec = {}
    tasim_prev_rec = {}
    smoothness_rec = { 'start': [], 'prev': [] }

    best_smoothness_start = 0
    best_tasim_start = None
    best_tasim_prev = None
    best_p = 0

    for p in range(n_pairs):
        tasim_prev = []
        tasim_start_from_begin = []
        tasim_start_from_end = []

        df = pd.read_csv(os.path.join(save_folder, f'{data_src}_prediction{p}.csv'))
        # df = pd.read_csv(os.path.join(save_folder, f'{data_src}_prediction{p}_c{ci}.csv'))

        for i in range(len(df)):
            tasim_start_from_begin.append(tanimoto_similarity(df.loc[0, 'smiles'], df.loc[i, 'smiles']))
            tasim_start_from_end.append(tanimoto_similarity(df.loc[len(df)-1, 'smiles'], df.loc[i, 'smiles']))
            if i > 0:
                tasim_prev.append(tanimoto_similarity(df.loc[i-1, 'smiles'], df.loc[i, 'smiles']))
            else:
                tasim_prev.append(None)
        
        tasim_start_rec[p] = tasim_start_from_begin
        tasim_prev_rec[p] = tasim_prev

        smoothness_start = get_smoothness_by_simstart(tasim_start_from_begin, tasim_start_from_end)
        smoothness_prev = get_smoothness_by_simprev(tasim_prev[1:])

        smoothness_rec['start'].append(smoothness_start)
        smoothness_rec['prev'].append(smoothness_prev)

        if smoothness_start > best_smoothness_start:
            best_tasim_start = tasim_start_from_begin
            best_tasim_prev = tasim_prev
            best_p = p
    
    smoothness_rec = pd.DataFrame(smoothness_rec)
    # smoothness_rec.to_csv(os.path.join(save_folder, 'smoothness.csv'))
    smoothness_rec.to_csv(os.path.join(save_folder, f'smoothness_c{ci}.csv'))

    print('plot the k best smiles list of interpolation...')    

    best_ids = topk(smoothness_rec['start'], k=n_plots)
    best_smiles_list = []
    for p in best_ids:
        df = pd.read_csv(os.path.join(save_folder, f'{data_src}_prediction{p}.csv'))
        best_smiles_list.append(df['smiles'].tolist())
    best_smiles_list = list(zip(*best_smiles_list))
    best_smiles_list = [item for sub_rec in best_smiles_list for item in sub_rec]
    plot_smiles_group(best_smiles_list, os.path.join(save_folder, f'{data_src}_prediction.png'),
                    n_per_mol=n_plots)

    print('plot the best smiles list of interpolation...')

    best_tasim_start = pd.DataFrame({ 'TaSim(start, current)': best_tasim_start })
    best_tasim_prev = pd.DataFrame({ 'TaSim(start, previous)': best_tasim_prev })

    line_plot(best_tasim_start, os.path.join(save_folder, f'{data_src}_best-tasim-start_{best_p}.png'), y_ticks='TanSim(start, current)')    
    line_plot(best_tasim_prev, os.path.join(save_folder, f'{data_src}_best-tasim-previous_{best_p}.png'), y_ticks='TanSim(start, previous)')    

    print('plot the line plot TaSim(start, current) and TaSim(previous, current)...')

    tasim_start_rec = pd.DataFrame(tasim_start_rec)
    tasim_prev_rec = pd.DataFrame(tasim_prev_rec)

    line_plot(tasim_start_rec, os.path.join(save_folder, f'{data_src}_tasim-start.png'), y_ticks='TanSim(start, current)')
    line_plot(tasim_prev_rec, os.path.join(save_folder, f'{data_src}_tasim-prev.png'), y_ticks='TanSim(start, previous)')


@torch.no_grad()
def test_decoder(args, toklen_data, df_train, scaler,
                 SRC, TRG, COND, device):
    # save_folder = os.path.join(args.infer_path, args.benchmark, 'test_decoder-formal',
    #                            args.model_name, slerp_or_lerp)
    # os.makedirs(save_folder, exist_ok=True)
    
    args.model_path = os.path.join(args.train_path, args.benchmark,
                                   args.model_name, f'model_{args.epoch}.pt')
    generator = get_generator(args, SRC, TRG, toklen_data, scaler, device)

    dp = DataloaderPreparation(device, SRC, TRG,
                               model_type=args.model_type,
                               property_list=args.property_list,
                               use_scaffold=args.use_scaffold)

    # file path

    data_src = 'test_scaffolds' # the data source
    main_folder = os.path.join(args.infer_path, args.benchmark, 'test_decoder-debug')
    data_path = os.path.join(main_folder, f"{data_src}_pair{'-scaffold' if args.use_scaffold else ''}.csv")
    save_folder = os.path.join(main_folder, args.model_name)

    os.makedirs(main_folder, exist_ok=True)
    os.makedirs(save_folder, exist_ok=True)

    n_pairs = 100 # number of the test pairs

    df_data = pd.read_csv(f'/fileserver-gamma/chaoting/ML/dataset/moses/raw/{data_src}.csv')
    if not os.path.exists(data_path):
        pairs = sample_molecule_pairs(df_data, n_pairs, args.property_list,
                                        args.use_scaffold, property_constraint=None,
                                        similarity_threshold=0.5)
        pairs.to_csv(data_path)

    if False:
        SIP = SmilesInterpolation(args, generator)

        # constraints = { 'logP': 3, 'tPSA': 20, 'QED': 0.4 } # the property constraints of pairs

        pairs = pd.read_csv(data_path, index_col=[0])
        SIP.generate_interpolated_smiles(pairs, save_folder, transform=True)

        # prepare and save pair data

    if False:
        SIP = SmilesInterpolation(args, generator)

        pairs = pd.read_csv(data_path, index_col=[0])
        trg_conds_list = get_trg_prop(args.property_list)

        for trg_conds in trg_conds_list:
            SIP.generate_interpolated_smiles(pairs, save_folder, trg_conds, transform=True)
    
    
    if True:
        data_path = os.path.join(main_folder, f"{data_src}_pair.csv")
        pairs = pd.read_csv(data_path, index_col=[0])
        
        SIP = SmilesInterpolation(args, generator)
        SIP.generate_with_different_scaffold(pairs, save_folder=save_folder)
    
    # property_fn = get_property_fn(args.property_list)
    
    # tasim_start_records = {}
    # tasim_prev_records = {}
    # # scasim_start_records = {}
    # # scasim_prev_records = {}
    # overall_smi_records = []
    
    # for testi in range(n_tests):
    #     current_path = os.path.join(save_folder, f'{testi}')
    #     os.makedirs(current_path, exist_ok=True)

    #     print('sample molecules from dataset...')
        
        
    #     src_smiles, src_scaffolds, src_conds = sample_molecule_pair(df_train, args.property_list,
    #                                                                 same_scaffold=args.use_scaffold,
    #                                                                 constraints=constraints)
        
    #     plot_smiles(src_smiles[0], os.path.join(current_path, 'src1.png'))
    #     plot_smiles(src_smiles[1], os.path.join(current_path, 'src2.png'))
    #     s = pd.DataFrame({ 'smiles': src_smiles })
    #     p = pd.DataFrame(np.array(src_conds), columns=args.property_list)
    #     smiles_prop = pd.concat([s, p], axis=1)
    #     smiles_prop.to_csv(os.path.join(current_path, 'src.csv'))

    #     print('encode molecules...')
        
    #     if args.use_scaffold:
    #         _, mus, logvars = encode_molecules(generator, src_smiles, src_conds, SRC,
    #                                            device, src_scaffolds, transform=True)
    #     else:
    #         _, mus, logvars = encode_molecules(generator, src_smiles, src_conds, SRC,
    #                                            device, transform=True)

    #     print('interpolate latent spaces...')
    #     print('sample molecules by decoder...')

    #     smi_records = []
    #     mol_records = []
        
    #     if args.use_scaffold:
    #         src_ids = [TRG.vocab.stoi[e] for e in TRG.tokenize(src_scaffolds[0])]

    #     # interpolation between two molecules
    #     for alpha in np.linspace(0, 1, n_interpolates+2, endpoint=True):
    #         if alpha == 0 or alpha == 1:
    #             continue
    #         # new_conds = interpolation['lerp'](src_conds[0], src_conds[0], alpha)
    #         new_conds = interpolation['lerp'](src_conds[0], src_conds[1], alpha)                
    #         toklen = int(interpolation['lerp'](mus[0].size(1), mus[1].size(1), alpha))

    #         mu_info = interpolation[slerp_or_lerp](mus[0][0,0,:],
    #                                                mus[1][0,0,:],
    #                                                alpha).repeat(1, toklen, 1)
    #         logvar_info = interpolation[slerp_or_lerp](logvars[0][0,0,:],
    #                                                    logvars[1][0,0,:],
    #                                                    alpha).repeat(1, toklen, 1)
    #         std_info = torch.exp(0.5*logvar_info)

    #         n_counts = 0
    #         std_current = 0.005
            
    #         while True:
    #             eps = torch.randn_like(mu_info).normal_(mean=0, std=std_current)
    #             new_z = eps.mul(std_info).add(mu_info)
                
    #             if args.use_scaffold:
    #                 smiles, *_ = generator.sample_smiles(prop=[new_conds],
    #                                                      src_ids=src_ids,
    #                                                      zs=new_z,
    #                                                      transform=True)
    #             else:
    #                 smiles, *_ = generator.sample_smiles(prop=[new_conds],
    #                                                      zs=new_z,
    #                                                      transform=True)
    #             mol = get_mol(smiles[0])
    #             if mol:
    #                 print(smiles[0])
    #                 smi_records.append(smiles[0])
    #                 mol_records.append(mol)
    #                 break
                                
    #             if (n_counts + 1) % 5 == 0:
    #                 std_current += 0.005
    #             print(std_current)
    #             if std_current >= 0.20:
    #                 exit('cannot sample valid smiles!')

    #             n_counts += 1
        
    #     smi_records.insert(0, src_smiles[0])
    #     smi_records.append(src_smiles[1])
    #     mol_records.insert(0, get_mol(src_smiles[0]))
    #     mol_records.append(get_mol(src_smiles[1]))
        
    #     if testi < n_plots:
    #         overall_smi_records.append(smi_records)
        
    #     plot_smiles_group(smi_records, os.path.join(current_path, 'prediction.png'))
    #     valid_prop = mols_to_props(mol_records, property_fn, n_jobs=args.n_jobs)
        
    #     df = pd.concat([pd.DataFrame({ 'smiles': smi_records }),
    #                     valid_prop], axis=1)
    #     df.to_csv(os.path.join(current_path, 'prediction.csv'))
        
    #     print('plot similarity...')
        
    #     tasim_start = []
    #     tasim_prev = []
        
    #     for i in range(len(smi_records)):
    #         tasim_start.append(tanimoto_similarity(smi_records[0], smi_records[i]))
    #         if i == 0:
    #             tasim_prev.append(None)
    #         else:
    #             tasim_prev.append(tanimoto_similarity(smi_records[i-1], smi_records[i]))        
        
    #     tasim_start_records[testi] = tasim_start
    #     tasim_prev_records[testi] = tasim_prev
    
    # # plot the interpolated molecules
    # overall_smi_records = list(zip(*overall_smi_records))
    # overall_smi_records = [item for sub_rec in overall_smi_records for item in sub_rec]
    # plot_smiles_group(overall_smi_records, os.path.join(save_folder, 'prediction.png'),
    #                   n_per_mol=n_plots)
  
    # tasim_start_records = pd.DataFrame(tasim_start_records)
    # tasim_prev_records = pd.DataFrame(tasim_prev_records)
    
    # line_plot(tasim_start_records, os.path.join(save_folder, 'tasim_start.png'))
    # line_plot(tasim_prev_records, os.path.join(save_folder, 'tasim_prev.png'))

    exit()
    
    dataset = create_dataset(interpolated_smiles, args.property_list)
    
    test_outputs = []
    for i in range(len(dataset)):
        dataloader = dp.get_dataloader(dataset.iloc[[i]], batch_size=1, is_train=False)
        test_outputs.append(sample_encoder_outputs(generator, dataloader))
    test_outputs = np.concatenate(test_outputs, axis=0)

    train = pd.read_csv("/fileserver-gamma/chaoting/ML/dataset/moses/prepared/train_v0_logP-tPSA-QED.csv")
    dataloader = dp.get_dataloader(train, batch_size=1, is_train=False)
    train_outputs = sample_encoder_outputs(generator, dataloader,
                                           transform=False,
                                           n_samples=1000)

    outputs = np.concatenate((train_outputs, test_outputs), axis=0)
    transformed_outputs = dimension_reduction['t-sne'](outputs)
    
    # plot projections on the latent space

    n_classes = []
    n_classes.extend([0]*train_outputs.shape[0])
    for i in range(len(test_outputs)):
        n_classes.append(i+1)

    plt.figure()
    plt.scatter(transformed_outputs[-len(test_outputs):, 0],
                transformed_outputs[-len(test_outputs):, 1],
                c=n_classes[-len(test_outputs):],
                s=3)
    plt.colorbar()
    plt.savefig('3-lat.png', dpi=300)
    
    
    
    
    
    
def test_decoder1(args, toklen_data, scaler, SRC, TRG, device):



    for epoch in args.epoch_list:
        args.use_model_path = os.path.join(args.train_path,
                                        args.model_name,
                                        f'model_{epoch}.pt')

        generator = prepare_generator(args, SRC, TRG,
                                      toklen_data, scaler, device)

        max_length = 80
        similarity_list = []
        distance_list = []
        cnt = 0
        n_samples = 1000

        props_t = np.array([property_peaks])
        props_t = np.tile(props_t, (2, 1))

        while cnt < n_samples:
            zs, _ = generator.sample_z_from_data(n=2)
            
            smiles, *_ = generator.sample_smiles(props_t, zs)
            sim = similarity_fcn(smiles[0], smiles[1])
            if sim == None:
                continue
            print(cnt, smiles)

            pad = torch.zeros((1,abs(max_length-zs[0].size(1)), zs[0].size(2)), dtype=torch.long)
            zs[0] = torch.concat([zs[0], pad], axis=1)
            pad = torch.zeros((1,abs(max_length-zs[1].size(1)), zs[1].size(2)), dtype=torch.long)
            zs[1] = torch.concat([zs[1], pad], axis=1)

            similarity_list.append(sim)
            distance_list.append(get_distance(zs[0], zs[1]) / max_length)
            cnt += 1
        
    df = pd.DataFrame({ 'similarity': similarity_list, 'distance': distance_list })
    df['distance'] = df['distance'] / df['distance'].max()
    print(df)
    df.to_csv('./4.csv')