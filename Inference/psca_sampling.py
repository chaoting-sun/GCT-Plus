import os
import torch
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
from functools import partial
import matplotlib.pyplot as plt
from moses.metrics import metrics
from collections import OrderedDict

from Model.build_model import get_sampler
from Configuration.config_default import prop_tolerance, \
    selected_target_prop, molgpt_selected_target_prop 
from Utils import get_mol, mapper, get_canonical, \
    mols_to_props, get_property_fn, plot_smiles, \
    murcko_scaffold, murcko_scaffold_similarity


def get_trg_prop_combination(property_list, scaffold_source=False):
    if scaffold_source == 'molgpt':
        prop_set = (molgpt_selected_target_prop[p] for p in property_list)
    else:
        prop_set = (selected_target_prop[p] for p in property_list)
    prop_comb = list(itertools.product(*prop_set))
    trg_prop_list = [list(c) for c in prop_comb]
    return np.array(trg_prop_list)


def get_sample(n, save_path, train, dataset):
    np.random.seed(0)

    if not os.path.exists(save_path):
        unique_scaffold = dataset.drop_duplicates(subset='scaffold', ignore_index=True)
        samples = unique_scaffold.sample(n=n, replace=False, ignore_index=True)
        samples['n_train'] = samples['scaffold'].apply(
            lambda sca: len(train[train.scaffold == sca]))
        samples.to_csv(save_path)

    samples = pd.read_csv(save_path, index_col=[0])
    return samples

    
def sample_smiles(sampler, n, scaffold, trg_prop,
                  batch_size, LOG):
    gen = []
    dconds = np.repeat([trg_prop], batch_size, axis=0)
    while n > 0:
        LOG.info(f'# Samples left: {n}')
        current_gen, *_ = sampler.sample_smiles(dconds[:min(n, batch_size)],
                                                scaffold=scaffold)
        gen.extend(current_gen)
        n -= len(gen)
    return gen


# def plot_highlighted_smiles(scaffold_sample, save_folder, property_list,
#                             train, n_jobs, n=25):
#     # test_scaffolds
#     sid = 76
#     trg_prop = ['3.0', '60.0', '0.85']

#     substructure = scaffold_sample.loc[sid, 'scaffold']
#     gen = pd.read_csv(os.path.join(save_folder, f's{sid}_p{"-".join(trg_prop)}_prop.csv'),
#                       index_col=[0])
#     # get smiles with correct substructure
#     gen['substructure'] = gen['smiles'].apply(murcko_scaffold)
#     mols = mapper(get_mol, gen['smiles'], n_jobs)
#     gen['canonical'] = mapper(get_canonical, mols, n_jobs)
#     gen = gen.drop_duplicates(subset=['canonical'])
#     gen = gen[gen.substructure == substructure]
#     # get smiles with low error
#     for i, p in enumerate(property_list):
#         gen[f'{p}-normalized_AE'] = gen[p].apply(lambda x: abs(x - float(trg_prop[i]))) / (train[p].max() - train[p].min())
    
#     gen = gen.sort_values(by=[f'{p}-normalized_AE' for p in property_list],
#                           ignore_index=True)
#     # gen = gen.round({ 'logP': 2, 'tPSA': 1, 'QED': 2 })
#     smiles_list = gen['smiles'].iloc[:n].tolist()
#     gen[property_list].iloc[:n].to_numpy()

#     descriptions = []
#     for i in range(n):
#         p = gen.loc[i, property_list].tolist()
#         p = f'logP: {p[0]:.2f}, tPSA: {p[1]:.1f}, QED: {p[2]:.2f}'
#         descriptions.append(p)
#     print(smiles_list)

#     plot_smiles(substructure, os.path.join(save_folder, f's{sid}.png'))
#     plot_highlighted_smiles_group(smiles_list, substructure,
#                                   img_size=(400, 260),
#                                   save_path=os.path.join(save_folder, 
#                                             f's{sid}_p{"-".join(trg_prop)}_gen.png'),
#                                   n_per_mol=5,
#                                   descriptions=descriptions)
#     exit()


@torch.no_grad()
def psca_sampling(
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

    LOG = logger(name='psca_sampling', log_path=os.path.join(args.save_folder, 'record.log'))
    cond_val_path = os.path.join(args.save_folder, f'condition_{"-".join(args.property_list)}.csv')
    prop_dist_path = os.path.join(args.save_folder, 'prop_distribution.png')
    avg_scaf_metric_path = os.path.join(args.save_folder, 'avg_scaf_metric.csv')
    avg_prop_metric_path = os.path.join(args.save_folder, 'avg_prop_metric.csv')

    # property conditions

    trg_prop_comb = get_trg_prop_combination(args.property_list, args.scaffold_source)
    pd.DataFrame(trg_prop_comb, columns=args.property_list).to_csv(cond_val_path)
    property_fn = get_property_fn(args.property_list)

    # get scaffold

    scaffold_path = os.path.join(args.scaffold_folder, f'{args.scaffold_source}.csv')

    if args.scaffold_source == 'train':
        scaffold_sample = get_sample(args.n_scaffolds, scaffold_path,
                                     train, dataset=train)

    elif args.scaffold_source == 'test_scaffolds':
        scaffold_sample = get_sample(args.n_scaffolds, scaffold_path,
                                     train, dataset=test_scaffolds)

    elif args.scaffold_source == 'molgpt':
        scaffold_sample = pd.read_csv(scaffold_path, index_col=[0])

    # get sampler

    sampler = get_sampler(args, SRC, TRG, toklen_data, scaler, device)

    # generate SMILES
    
    LOG.info('start generation')

    for sid in range(len(scaffold_sample)):
        continue
        
        scaffold = scaffold_sample.loc[sid, 'scaffold']
        metric_path = os.path.join(args.save_folder, f'metric_s{sid}.csv')            

        # generate SMILES

        for pid, trg_prop in enumerate(trg_prop_comb):
            suffix = '-'.join(map(str, trg_prop))
            # gen_path = os.path.join(save_folder, f's{sid}_p{suffix}_gen.csv')
            gen_path = os.path.join(args.save_folder, f'gen_s{sid}_p{pid}.csv')

            if os.path.exists(gen_path):
                continue
            
            LOG.info(f'Generate SMILES: sid = {sid}\tscaffold = {scaffold}\tpid = {pid}')
            
            gen = sample_smiles(sampler, args.n_samples, scaffold,
                                trg_prop, args.batch_size, LOG)
            gen = pd.DataFrame(gen, columns=['smiles'])
            gen.to_csv(gen_path)

        # define metrics

        metric = OrderedDict()

        for p in args.property_list:
            metric[p] = []
        for met in ('scaffold', 'valid', 'unique', 'novel', 'intDiv', 'sim', 'SSF', 'sim80'):
            metric[met] = []
        for p in args.property_list:
            metric[f'{p}-MSE'] = []
            metric[f'{p}-MAE'] = []
            metric[f'{p}-SD'] = []
        metric['valid_in_tolerance'] = []
        metric['unique_in_tolerance'] = []

        # compute properties and metrics

        for pid, trg_prop in enumerate(trg_prop_comb):
            LOG.info(f'Compute properties and metrics: sid = {sid}\tscaffold = {scaffold}\tpid = {pid}')

            gen_path = os.path.join(args.save_folder, f'gen_s{sid}_p{pid}.csv')
            prop_path = os.path.join(args.save_folder, f'prop_s{sid}_p{pid}.csv')
            
            gen = pd.read_csv(gen_path, index_col=[0])
            gen = gen.dropna(subset='smiles').reset_index(drop=True)
            gen['mol'] = mapper(get_mol, gen['smiles'], args.n_jobs)

            valid = gen.dropna(subset='mol').reset_index(drop=True).copy()
            valid['smiles'] =  mapper(get_canonical, valid['mol'], args.n_jobs)
            valid['scaffold'] = mapper(murcko_scaffold, valid['smiles'], args.n_jobs)

            # compute properties

            # if not os.path.exists(prop_path):
            prop = mols_to_props(valid['mol'], property_fn, n_jobs=args.n_jobs)
            prop.insert(0, 'scaffold', valid['scaffold'])
            prop.insert(0, 'smiles', valid['smiles'])
            prop.to_csv(prop_path)

            # compute metrics

            valid = pd.read_csv(prop_path, index_col=[0])
            similarity_fn = partial(murcko_scaffold_similarity, smi_or_mol2=scaffold)
            valid['scaffold_sim'] = mapper(similarity_fn, valid['smiles'], args.n_jobs)

            # Molgpt defined validity as the fraction of SMILES that
            # "satisfy chemical valencies and contain scaffolds that
            # have a Tanimoto similarity of at least 0.8 to the desired scaffold."

            if args.scaffold_source == 'molgpt':
                valid = valid.dropna(subset='scaffold_sim').reset_index(drop=True)
                valid = valid[valid.scaffold_sim >= 0.8]

            unique = valid.drop_duplicates(subset='smiles', ignore_index=True).copy()

            # compute metrics

            for i, p in enumerate(args.property_list):
                metric[p].append(trg_prop[i])
            metric['scaffold'].append(scaffold)

            if len(valid) > 0:
                _valid = len(valid) / args.n_samples
                _unique = len(unique) / len(valid)
                _novel = len(set(unique['smiles']) - set(train['smiles'])) / len(unique)
                _intDiv = metrics.internal_diversity(valid['smiles'], args.n_jobs)
                _sim = valid['scaffold_sim'].mean()
                _SSF = len(valid[valid.scaffold_sim == 1]) / len(valid)
                _sim80 = len(valid[valid.scaffold_sim >= 0.8]) / len(valid)
            else:
                _valid = _unique = _novel = _intDiv = sim = _SSF = _sim80 = np.nan

            metric['valid'].append(_valid)
            metric['unique'].append(_unique)
            metric['novel'].append(_novel)
            metric['intDiv'].append(_intDiv)
            metric['sim'].append(_sim)
            metric['SSF'].append(_SSF)
            metric['sim80'].append(_sim80)

            for i, p in enumerate(args.property_list):
                if len(valid) > 0:
                    delp = valid[p] - trg_prop[i]
                    mse = delp.mean()
                    mae = delp.abs().mean()
                    sd = delp.std()
                else:
                    mse = mae = sd = np.nan
                metric[f'{p}-MSE'].append(mse)
                metric[f'{p}-MAE'].append(mae)
                metric[f'{p}-SD'].append(sd)

            good_mol = valid.copy()

            good_mol = good_mol[good_mol.scaffold == scaffold] # meet scaffold condition
            for i, p in enumerate(args.property_list): # meet property condition
                good_mol = good_mol[(good_mol[p] - trg_prop[i]).abs() <= prop_tolerance[p]]
            
            metric['valid_in_tolerance'].append(len(good_mol) / args.n_samples)
            metric['unique_in_tolerance'].append(len(good_mol.drop_duplicates('smiles')) / args.n_samples)
                        
            current_metric = pd.DataFrame(metric)
            current_metric.to_csv(metric_path)
            LOG.info(current_metric)

    # compute average metric

    all_metric = []

    for sid in range(len(scaffold_sample)):
        metric = pd.read_csv(os.path.join(args.save_folder,
                             f'metric_s{sid}.csv'), index_col=[0])
        metric['scaffold'] = len(metric) * [scaffold_sample.loc[sid, 'scaffold']]                
        all_metric.append(metric)

    all_metric = pd.concat(all_metric, axis=0)

    scaf_avg_metric = all_metric.groupby('scaffold').mean()
    scaf_avg_metric = scaf_avg_metric.reindex(scaffold_sample['scaffold'])
    scaf_avg_metric = scaf_avg_metric.reset_index()
    scaf_avg_metric.to_csv(avg_scaf_metric_path)

    all_metric = all_metric.select_dtypes(include=['float', 'int'])
    prop_avg_metric = all_metric.groupby(args.property_list).mean()
    prop_avg_metric = prop_avg_metric.reset_index(drop=True)
    prop_avg_metric.to_csv(avg_prop_metric_path)

    # plot property distribution

    xlimit = {
        'logP': [-2,  6],
        'tPSA': [0, 120],
        'QED' : [0.2, 1],
        'SAS' : [1,  10],
    }
    
    LOG.info('Gather molecular properties')
    
    gen_prop = []
    for sid in range(args.n_scaffolds):
        for pid, prop in enumerate(trg_prop_comb):
            current_prop = pd.read_csv(os.path.join(args.save_folder, f'prop_s{sid}_p{pid}.csv'),
                                       index_col=[0])
            trg_prop = pd.DataFrame(np.tile(np.array(prop), (len(current_prop),1)),
                                    columns=[f'trg_{p}' for p in args.property_list])
            current_prop = pd.concat([trg_prop, current_prop], axis=1)
            gen_prop.append(current_prop)
    gen_prop = pd.concat(gen_prop, axis=0)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16.5, 4.5))

    for i, p in enumerate(args.property_list):
        LOG.info(f'Plot property distributions: {p}')

        ax = axes[i]
        if args.scaffold_source == 'molgpt':
            trg_prop = molgpt_selected_target_prop[p]
        else:
            trg_prop = selected_target_prop[p]

        for tp in trg_prop:
            sns.kdeplot(data=gen_prop[gen_prop[f'trg_{p}'] == tp].loc[:, p],
                        ax=ax, shade=True, linewidth=2.5, legend=False)
        sns.kdeplot(data=train.loc[:, p], ax=ax, shade=False,
                    linewidth=2.5, color='red', legend=False)
        
        ax.set_xlabel(xlabel=p, fontsize=22)
        if i == 0:
            ax.set_ylabel(ylabel='Density', fontsize=22)
        else:
            ax.set_ylabel(None)
        ax.set_xlim(xlimit[p][0], xlimit[p][1])
        ax.tick_params(axis="both", which="major", labelsize=20)
        ax.legend(trg_prop+['train'], fontsize=20)
       
        for tp in trg_prop:
            ax.axvline(x=tp, linestyle='--', color='gray')

    fig.savefig(prop_dist_path, bbox_inches="tight")