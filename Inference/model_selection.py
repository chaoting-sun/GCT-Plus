import os
import random
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import OrderedDict

from Utils.mapper import mapper
from Utils.smiles import get_mol, mol_to_smi, get_canonical
from Model.build_model import get_sampler
from Utils.properties import mols_to_props, get_property_fn
from guacamol.utils.chemistry import (
    calculate_pc_descriptors,
    calculate_internal_pairwise_similarities,
    continuous_kldiv,
    discrete_kldiv
)


# def sample_from_dataset(dataset, property_fn, n, n_jobs):
#     test_prop = random.sample(dataset, n)
#     mols = mapper(get_mol, test_prop, n_jobs)
#     smiles = pd.DataFrame({ 'SMILES': test_prop })
#     props = mols_to_props(mols, property_fn, n_jobs=n_jobs)
#     smiles_props = pd.concat([smiles, props], axis=1)
#     return smiles_props


def sample_smiles(sampler, n, batch_size, LOG):
    gen = []
    while n > 0:
        LOG.info(f'# Samples left: {n}')
        current_gen, *_ = sampler.sample_smiles(min(n, batch_size))
        gen.extend(current_gen)
        n -= len(current_gen)
    return gen


def kl_divergence(data1, data2):
    # compute the probability distributions
    density1 = stats.gaussian_kde(data1)
    density2 = stats.gaussian_kde(data2)
    # evaluate the distributions at a set of points
    x = np.linspace(min(data1.min(), data2.min()), max(data1.max(), data2.max()), 1000)
    p1 = density1(x)
    p2 = density2(x)
    # compute the KL divergence
    kl_div = stats.entropy(pk=p1, qk=p2)
    return kl_div


def compute_kldiv_score(unique_gen, reference):
    pc_descriptor_subset = [
        'BertzCT',
        'MolLogP',
        'MolWt',
        'TPSA',
        'NumHAcceptors',
        'NumHDonors',
        'NumRotatableBonds',
        'NumAliphaticRings',
        'NumAromaticRings'
    ]
    
    d_gen = calculate_pc_descriptors(unique_gen, pc_descriptor_subset)
    d_ref = calculate_pc_descriptors(reference, pc_descriptor_subset)

    kldivs = {}
    
    for i in range(4):
        kldiv = continuous_kldiv(X_baseline=d_ref[:, i], X_sampled=d_gen[:, i])
        kldivs[pc_descriptor_subset[i]] = kldiv
        print(pc_descriptor_subset[i], kldiv)

    for i in range(4, 9):
        kldiv = discrete_kldiv(X_baseline=d_ref[:, i], X_sampled=d_gen[:, i])
        kldivs[pc_descriptor_subset[i]] = kldiv
        print(pc_descriptor_subset[i], kldiv)

    ref_sim = calculate_internal_pairwise_similarities(reference)
    ref_sim = ref_sim.max(axis=1)

    sampled_sim = calculate_internal_pairwise_similarities(unique_gen)
    sampled_sim = sampled_sim.max(axis=1)

    kldiv_int_int = continuous_kldiv(X_baseline=ref_sim, X_sampled=sampled_sim)
    kldivs['internal_similarity'] = kldiv_int_int
    partial_scores = [np.exp(-score) for score in kldivs.values()]
    kldivs['score'] = sum(partial_scores) / len(partial_scores)

    return kldivs


def model_selection(
        args,
        test,
        toklen_data,
        scaler,
        SRC,
        TRG,
        device,
        logger
    ):
    
    os.makedirs(args.save_folder, exist_ok=True)

    LOG = logger(name='model_selection', log_path=os.path.join(args.save_folder, 'record.log'))

    test_path = os.path.join(args.save_folder, 'test_samples.csv')
    ref_prop_folder = os.path.join(args.save_folder, 'test_samples_prop.csv')
    kl_path = os.path.join(args.save_folder, 'kl.csv')
    
    interested_properties = ['logP', 'tPSA', 'QED', 'MW', 'SAS', 'NP',
                             'HAC', 'HBA', 'HBD', 'RBN', 'AIRN', 'ARRN']
    property_fn = get_property_fn(interested_properties)

    # sample molecules from test dataset as the reference set

    if not os.path.exists(test_path):
        test_samples = test.sample(n=args.n_samples).reset_index(drop=True)
        test_samples.to_csv(test_path)
    
    test_samples = pd.read_csv(test_path, index_col=[0])
    
    kldivs_dict = OrderedDict()

    for i, epoch in enumerate(args.epoch_list):
        gen_path = os.path.join(args.save_folder, f'gen{epoch}.csv')
        
        args.model_path = os.path.join(args.model_folder, f'model_{epoch}.pt')
        sampler = get_sampler(args, SRC, TRG, toklen_data, scaler, device)

        if not os.path.exists(gen_path):
            gen = sample_smiles(sampler, args.n_samples, args.batch_size, LOG)
            gen = pd.DataFrame(gen, columns=['smiles'])
            gen.to_csv(gen_path)

        # compute kldiv scores
        
        gen = pd.read_csv(gen_path, index_col=[0])
        gen = gen.dropna(subset='smiles')

        LOG.info('compute kl divergence score...')

        gen['mol'] = mapper(get_mol, gen['smiles'], args.n_jobs)
        gen = gen.dropna(subset='mol').reset_index(drop=True)
        gen['canonical'] = mapper(get_canonical, gen['mol'], args.n_jobs)
        gen = gen.drop_duplicates('canonical')

        kldivs = compute_kldiv_score(set(gen['canonical']), test_samples['smiles'])
        
        if i == 0:
            for k, v in kldivs.items():
                kldivs_dict[k] = [v]
        else:
            for k, v in kldivs.items():
                kldivs_dict[k].append(v)

        cumm_kldivs = pd.DataFrame(kldivs_dict, index=args.epoch_list[:i+1])
        cumm_kldivs.to_csv(kl_path)

    cumm_kldivs = pd.read_csv(kl_path, index_col=[0])
    best_epoch = cumm_kldivs['score'].idxmax()
    LOG.info(f'Best epoch: {best_epoch}')    

    for epoch in args.epoch_list:
        gen_path = os.path.join(args.save_folder, f'gen{epoch}.csv')
        prop_path = os.path.join(args.save_folder, f'prop{epoch}.csv')
        
        if not os.path.exists(gen_path):
            continue
        
        gen = pd.read_csv(gen_path, index_col=[0])
        gen['mol'] = mapper(get_mol, gen['smiles'], args.n_jobs)
        gen = gen.dropna(subset='mol', ignore_index=True)
        gen_prop = mols_to_props(gen['mol'], property_fn, args.n_jobs)
        gen_prop.to_csv(prop_path)
        
    LOG.info('compute gen prop...')

    xlimit = {
        'logP': [-5, 10],
        'tPSA': [0, 150],
        'QED' : [0,   1],
        'MW'  : [0, 600],
        'SAS' : [1,  10],
        'NP'  : [-5,  5],
    }

    LOG.info('compute test prop...')

    if not os.path.exists(ref_prop_folder):
        ref = pd.read_csv(test_path, index_col=[0])
        ref['mol'] = mapper(get_mol, ref['smiles'], args.n_jobs)
        ref = ref.dropna(subset=['mol'], ignore_index=True)
        ref_prop = mols_to_props(ref['mol'], property_fn, args.n_jobs)
        ref_prop.to_csv(ref_prop_folder)

    ref_prop = pd.read_csv(ref_prop_folder, index_col=[0])

    LOG.info('Plot prop...')

    for epoch in args.epoch_list:
        prop_path = os.path.join(args.save_folder, f'prop{epoch}.csv')
        fig_dist_path = os.path.join(args.save_folder, f'dist{epoch}.png')
        fig_num_path = os.path.join(args.save_folder, f'num{epoch}.png')

        LOG.info(f'Epoch: {epoch}')
        
        if not os.path.exists(prop_path):
            continue

        gen_prop = pd.read_csv(prop_path, index_col=[0])

        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 8.5))

        for i, prop in enumerate(['logP', 'tPSA', 'QED', 'MW', 'SAS', 'NP']):
            rowi = i // 3
            coli = i % 3
            ax = axes[rowi, coli]

            df = pd.DataFrame({ 'gen': gen_prop[prop], 'test': ref_prop[prop] })
            sns.kdeplot(data=df, ax=ax, shade=True, linewidth=2.5, legend=False)
            
            ax.set_xlim(left=xlimit[prop][0], right=xlimit[prop][1])
            ax.set_xlabel(xlabel=prop, fontsize=17)
            if coli == 0:
                ax.set_ylabel(ylabel='Density', fontsize=17)
            else:
                ax.set_ylabel(None)
            ax.tick_params(axis="both", which="major", labelsize=13)
        
        fig.legend(labels=['gen', 'test'], fontsize=16, loc='lower center',
                   ncol=2, bbox_to_anchor=(0.5, -0.05))
        fig.savefig(fig_dist_path, bbox_inches="tight")

        LOG.info(f'Save figure: {fig_num_path}')

        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 8.5))

        for i, prop in enumerate(['HAC', 'HBA', 'HBD', 'RBN', 'AIRN', 'ARRN']):
            rowi = i // 3
            coli = i % 3

            ax = axes[rowi, coli]

            gen_cnt = gen_prop[prop].value_counts(sort=False).sort_index()
            test_cnt = ref_prop[prop].value_counts(sort=False).sort_index()

            gen_cnt = gen_cnt / gen_cnt.sum()
            test_cnt = test_cnt / test_cnt.sum()

            gen_cnt.plot(kind='bar', ax=ax, color='blue', alpha=0.7, rot=0)
            test_cnt.plot(kind='bar', ax=ax, color='orange', alpha=0.7, rot=0)

            ax.set_xlabel(xlabel=prop, fontsize=17)
            if coli == 0:
                ax.set_ylabel(ylabel='Density', fontsize=17)
            else:
                ax.set_ylabel(None)
            ax.tick_params(axis="both", which="major", labelsize=13)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        fig.legend(labels=['gen', 'test'], fontsize=16, loc='lower center',
                   ncol=2, bbox_to_anchor=(0.5, -0.05))
        fig.savefig(fig_num_path, bbox_inches="tight")
