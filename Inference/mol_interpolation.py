import os
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Model.build_model import get_sampler
from Utils.properties import get_property_fn, mols_to_props
from Utils.smiles import get_mol, murcko_scaffold, \
    tanimoto_similarity, plot_smiles_group


def lerp(v1, v2, alpha):
    return v1 * (1-alpha) + v2 * alpha


def slerp(v1, v2, alpha):
    is_torch = isinstance(v1, torch.Tensor) and isinstance(v2, torch.Tensor)
    norm = torch.norm if is_torch else np.linalg.norm
    acos = torch.acos if is_torch else np.arccos
    dot = torch.dot if is_torch else np.dot
    sin = torch.sin if is_torch else np.sin

    z1_normalized = v1 / norm(v1)
    z2_normalized = v2 / norm(v2)
    omega = acos(dot(z1_normalized, z2_normalized))
    
    return (sin((1 - alpha) * omega) * v1 + sin(alpha * omega) * v2) / sin(omega)


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


def sample_encoder_outputs(sampler, dataloader, transform=True, n_samples=2000):
    for i, batch in enumerate(dataloader):
        print(f'batch: {i}')
        if i == n_samples:
            break

        _, mu, _ = sampler.encode(batch, transform)
        
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


def sample_molecular_pairs(dataset, n, property_list, same_scaffold=True,
                           prop_threshold=None, sim_threshold=0.5):
    valid_pair_set = set()

    while len(valid_pair_set) < n:
        left = dataset.copy()

        # scaffold constraint

        if same_scaffold:
            rand_id = np.random.choice(len(left))
            left = left[left.scaffold == left.loc[rand_id, 'scaffold']]
        
        if len(left) < 2:
            continue
        
        is_valid = True
        n_test_limit = 20

        for i in range(n_test_limit):
            pair = left.sample(n=2)

            # similarity constraint

            if tanimoto_similarity(pair['smiles'].iloc[0], pair['smiles'].iloc[1]) > sim_threshold:
                continue

            # property constraint

            if len(property_list) > 0 and prop_threshold is not None:
                for i, (p, delp) in enumerate(prop_threshold.items()):
                    if abs(pair[p].iloc[0] - pair[p].iloc[1]) > delp:
                        is_valid = False
            
            if is_valid:
                valid_pair_set.add(tuple(sorted((pair.index[0], pair.index[1]))))
                break

    final_pairs = []
    for pair_no in list(valid_pair_set):
        row_pair = []
        random.shuffle(list(pair_no))

        for i, no in enumerate(pair_no):
            pair = dataset.loc[[no], ['smiles', 'scaffold'] + property_list]
            row = pair.rename(columns={**{ 'smiles': f'src_{i}', 'scaffold': f'scaffold_{i}'},
                                       **{ p: f'{p}_{i}' for p in property_list }})
            row = row.reset_index(drop=True)
            row_pair.append(row)
        row_pair = pd.concat(row_pair, axis=1)
        final_pairs.append(row_pair)

    final_pairs = pd.concat(final_pairs, axis=0).reset_index(drop=True)
    return final_pairs


def line_plot(df, save_path, y_label):
    df.index = np.arange(1, len(df)+1)
    
    fig = plt.figure(dpi=150)
    ax = df.plot(figsize=(5, 4.6), use_index=True,
                 kind='line', color='black', legend=False,
                 alpha=0.5, ax=plt.gca())
    ax.set_xlabel('Interpolation step', fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)
    ax.set_yticks(np.arange(0, 1+0.2, 0.2))
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    plt.tight_layout()
    plt.savefig(save_path)


def topk(arr, k):
    arr = np.array(arr)
    ids = np.argsort(arr)[::-1][:k]
    return ids


class SmilesInterpolation:
    def __init__(self, args, sampler, LOG):
        self.LOG = LOG
        self.sampler = sampler

        self.n_jobs = args.n_jobs
        self.model_type = args.model_type
        self.save_folder = args.save_folder
        self.use_scaffold = args.use_scaffold
        self.property_list = args.property_list
        self.n_interpolations = args.n_interpolations
        self.property_fn = get_property_fn(args.property_list)
    
    def approximate_z(self, v, toklen):
        """ approximate mu or logvar"""
        mu = v.mean(dim=0)
        std = v.std(dim=0)
        app = torch.empty((toklen, v.size(1)))
        for d in range(v.size(1)):
            app[:, d] = app[:, d].normal_(mean=mu[d], std=std[d])
        return app

    def interpolate_z_pair(self, z1, z2, alpha, interpolate_fn=slerp):
        toklen = int(lerp(z1.size(0), z1.size(1), alpha))

        z1_approx = self.approximate_z(z1, toklen)
        z2_approx = self.approximate_z(z2, toklen)
        new_app = torch.empty((1, toklen, z1.size(1)))
        for i in range(toklen):
            new_app[0, i, :] = interpolate_fn(z1_approx[i, :], z2_approx[i, :], alpha)
        return new_app
    
    def compute_smoothness_prev(sim_prev, threshold=0.50):
        return sum([1 for s in sim_prev if s >= threshold]) / len(sim_prev)
    
    def compute_smoothness_start(sim_start_forward, sim_start_reverse):
        del_begin = np.array([sim_start_forward[i] - sim_start_forward[i-1]
                              for i in range(1, len(sim_start_forward))])
        del_end = np.array([sim_start_reverse[i] - sim_start_reverse[i-1]
                            for i in range(1, len(sim_start_reverse))])
        return 1- (del_begin.std(ddof=1)*del_end.std(ddof=1))**0.5
    
    def interpolate_smiles(self, pairs):
        # define save path
        
        smoothness_path = os.path.join(self.save_folder, 'smoothness.csv')
        sim_start_path = os.path.join(self.save_folder, 'sim-start.png')
        sim_prev_path = os.path.join(self.save_folder, 'sim-prev.png')
        best_pred_path = os.path.join(self.save_folder, f'prediction.png')

        for p, row in pairs.iterrows():
            self.LOG.info(f'# Pair = {p}')

            smiles_path = os.path.join(self.save_folder, f'prediction{p}.csv')
            figure_path = os.path.join(self.save_folder, f'prediction{p}.png')

            if os.path.exists(smiles_path):
                continue
            
            # input information

            src0 = row['src_0']
            src1 = row['src_1']
            scaffold0 = row['scaffold_0']
            scaffold1 = row['scaffold_1']
            prop0 = row[[f'{prop}_0' for prop in self.property_list]].to_numpy()
            prop1 = row[[f'{prop}_1' for prop in self.property_list]].to_numpy()

            # encode molecules
                        
            kwargs0 = { 'smiles_list': [src0] }
            kwargs1 = { 'smiles_list': [src1] }

            if self.use_scaffold:
                kwargs0['scaffold_list'] = [scaffold0]
                kwargs1['scaffold_list'] = [scaffold1]

            if len(self.property_list) > 0:
                kwargs0['econds'] = [prop0]
                kwargs1['econds'] = [prop1]

            _, mu0, logvar0 = self.sampler.encode_smiles(**kwargs0)
            _, mu1, logvar1 = self.sampler.encode_smiles(**kwargs1)

            # interpolate latent space

            gen_smi = []
            gen_mol = []

            for alpha in np.linspace(0, 1, self.n_interpolations+2, endpoint=True):
                trg_prop = prop0 * (1 - alpha) + prop1 * alpha

                # only interpolate the space between the two
                
                if alpha == 0 or alpha == 1:
                    continue
                
                # decode the interpolated latent space into smiles

                n_counts = 0
                curr_std = 0

                while True:
                    # create interpolated z

                    mu_ip = self.interpolate_z_pair(mu0[0], mu1[0], alpha)
                    logvar_ip = self.interpolate_z_pair(logvar0[0], logvar1[0], alpha)
                    if self.model_type == 'ctf':
                        std_ip = torch.empty_like(logvar_ip).normal_(mean=0, std=curr_std)
                    else:
                        std_ip = torch.exp(0.5*logvar_ip)
                    eps = torch.randn_like(mu_ip).normal_(mean=0, std=curr_std)
                    new_z = eps.mul(std_ip).add(mu_ip)

                    # sample smiles
                    
                    kwargs = { 'zs': new_z }
                    
                    if self.use_scaffold:
                        kwargs['scaffold'] = scaffold0
                    if len(self.property_list) > 0:
                        kwargs['dconds'] = [trg_prop]
                    else:
                        kwargs['n'] = new_z.size(0)

                    smiles, *_ = self.sampler.sample_smiles(**kwargs)

                    mol = get_mol(smiles[0])

                    if mol:
                        self.LOG.info(f'alpha = {alpha}: SMILES = {smiles[0]}')
                        gen_smi.append(smiles[0])
                        gen_mol.append(mol)
                        break
                                    
                    if (n_counts + 1) % 2 == 0:
                        curr_std += 0.005
                    
                    self.LOG.info(f'current std: {curr_std}')

                    if self.model_type == 'ctf':
                        if curr_std >= 2.0:
                            exit('cannot sample valid smiles!')
                    else:
                        if curr_std >= 1.0:
                            exit('cannot sample valid smiles!')

                    n_counts += 1
            
            gen_smi = [src0] + gen_smi + [src1]
            valid_prop = mols_to_props(gen_mol, self.property_fn, n_jobs=self.n_jobs)
            df = pd.concat([pd.DataFrame({ 'smiles': gen_smi }), valid_prop], axis=1)
            df.to_csv(smiles_path)

            plot_smiles_group(gen_smi, figure_path, n_per_mol=5)

        # compute and plot tanimoto similarity

        smoothness_record = { 'start': [], 'prev': [] }

        for p in range(len(self.pairs)):
            sim_prev = []
            sim_start_forward = []
            sim_start_reverse = []

            pred = pd.read_csv(os.path.join(self.save_folder, f'prediction{p}.csv'))

            for i in range(len(pred)):
                first_smi = pred.loc[0, 'smiles']
                last_smi = pred.loc[len(pred)-1, 'smiles']
                curr_smi = pred.loc[i, 'smiles']

                sim_start_forward.append(tanimoto_similarity(first_smi, curr_smi))
                sim_start_reverse.append(tanimoto_similarity(last_smi, curr_smi))
                if i > 0:
                    prev_smi = df.loc[i-1, 'smiles']
                    sim_prev.append(tanimoto_similarity(prev_smi, curr_smi))
                else:
                    sim_prev.append(None)
            
            # compute smoothness

            smoothness_record['start'].append(self.compute_smoothness_start(sim_start_forward,
                                                                            sim_start_reverse))
            smoothness_record['prev'].append(self.compute_smoothness_prev(sim_prev[1:]))

        smoothness_record = pd.DataFrame(smoothness_record)
        smoothness_record.to_csv(smoothness_path)

        # plot k best predictions

        best_smiles_list = []
        best_ids = topk(smoothness_record['start'], k=6)

        for p in best_ids:
            pred = pd.read_csv(os.path.join(self.save_folder, f'prediction{p}.csv'))
            best_smiles_list.append(pred['smiles'].tolist())

        best_smiles_list = list(zip(*best_smiles_list))
        best_smiles_list = [item for sub_rec in best_smiles_list for item in sub_rec]
        plot_smiles_group(best_smiles_list, best_pred_path, n_per_mol=6, img_size=(530, 300))

        # plot line plot of all smoothnesses

        tasim_start_rec = pd.DataFrame(tasim_start_rec)
        tasim_prev_rec = pd.DataFrame(tasim_prev_rec)

        line_plot(tasim_start_rec, y_label=r'$SIM_{start}$',
                  save_path=sim_start_path)
        line_plot(tasim_prev_rec, y_label=r'$SIM_{prev}$',
                  save_path=sim_prev_path)


@torch.no_grad()
def mol_interpolation(
        args,
        toklen_data,
        train,
        test_scaffolds,
        scaler,
        SRC,
        TRG,
        device,
        logger
    ):
    # define save path

    os.makedirs(args.save_folder, exist_ok=True)

    LOG = logger(name='mol_interpolation', log_path=os.path.join(args.save_folder, 'record.log'))    
    if args.use_scaffold:
        pair_path = os.path.join(args.pair_folder, f"{args.pair_source}_samples-scaffold.csv")
    else:
        pair_path = os.path.join(args.pair_folder, f"{args.pair_source}_samples.csv")
    
    # get sampler

    args.model_path = os.path.join(args.model_folder, args.model_name)
    sampler = get_sampler(args, SRC, TRG, toklen_data, scaler, device)
    # sampler = None
    
    # sample molecular pairs
    
    if not os.path.exists(pair_path):
        LOG.info('Sample molecular pairs')
        
        if args.pair_source == 'train':
            pairs = sample_molecular_pairs(train, args.n_pairs, args.property_list,
                                           args.use_scaffold, sim_threshold=0.5)
        elif args.pair_source == 'test_scaffolds':
            pairs = sample_molecular_pairs(test_scaffolds, args.n_pairs, args.property_list,
                                           args.use_scaffold, sim_threshold=0.5)
        pairs.to_csv(pair_path)

    #  interpolate molecular pairs
    
    pairs = pd.read_csv(pair_path, index_col=[0])

    SIP = SmilesInterpolation(args, sampler, LOG)
    SIP.interpolate_smiles(pairs)
