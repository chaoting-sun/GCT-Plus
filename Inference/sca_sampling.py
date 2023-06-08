import os
import torch
import numpy as np
import pandas as pd
from functools import partial
from moses.metrics import metrics
from collections import OrderedDict
from pathos.multiprocessing import ProcessingPool as Pool
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

from Model.build_model import get_generator
from Utils.smiles import (
    get_mol,
    murcko_scaffold,
    murcko_scaffold_similarity,
    plot_smiles_group,
    plot_smiles,
    mol_to_smi
)
from Utils.properties import mols_to_props, get_property_fn
from Utils.metric import get_error_fn
from Utils.mapper import mapper
from Utils.seed import set_seed
import matplotlib.pyplot as plt
import seaborn as sns
from Utils.smiles import plot_highlighted_smiles_group
from rdkit import Chem
import random


def get_sample(df_dataset, df_train, data_folder, data_name, n):
    np.random.seed(0)
    save_path = os.path.join(data_folder, f'{data_name}_sample.csv')

    if not os.path.exists(save_path):
        data_sample = df_dataset.sample(frac=1).reset_index(drop=True)    
        data_sample = data_sample.drop_duplicates(subset='scaffold', ignore_index=True)
        data_sample[:n].to_csv(save_path)
        data_sample['n_train'] = data_sample['scaffold'].apply(
            lambda sca: len(df_train[df_train.scaffold == sca]))
        
    return pd.read_csv(save_path, index_col=[0])


def generate_scaffold(smiles):
    if smiles is not None and isinstance(smiles, str) and len(smiles) > 0:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return MurckoScaffoldSmiles(mol=mol)
    else:
        return None
    
    
def plot_highlighted_smiles(scaffold_sample, save_folder, n_jobs=1, gen_id=85, n=36):
    # row : col = 4 : 5 = 300 : 140
    
    substructure = scaffold_sample.loc[gen_id, 'scaffold']
    gen = pd.read_csv(os.path.join(save_folder, f's{gen_id}_gen.csv'),
                      index_col=[0])
    gen['substructure'] = gen['smiles'].apply(generate_scaffold)
    mols = mapper(get_mol, gen['smiles'], n_jobs)
    gen['canonical'] = mol_to_smi(mols, n_jobs)
    gen = gen.drop_duplicates(subset=['canonical'])
    gen = gen[gen.substructure == substructure]
    smiles_list = gen.sample(n=n)['canonical'].tolist()
    print(smiles_list)

    plot_smiles(substructure, os.path.join(save_folder, f's{gen_id}.png'))
    print(os.path.join(save_folder, f's{gen_id}.png'))
    print(os.path.join(save_folder, f's{gen_id}_gen.png'))
    plot_highlighted_smiles_group(smiles_list, substructure, 
                                  img_size=(250, 140),
                                  save_path=os.path.join(save_folder, f's{gen_id}_gen.png'),
                                  n_per_mol=6)
    exit()


def sca_sampling(
        args,
        toklen_data,
        df_train,
        df_test_scaffolds,
        scaler,
        SRC,
        TRG,
        device,
        logger
    ):    
    train_set = set(df_train['smiles'])

    task_path = os.path.join(args.infer_path, args.benchmark, 'sca_sampling')
    os.makedirs(task_path, exist_ok=True)
    LOG = logger(name='sca_sampling', log_path=os.path.join(task_path, 'record.log'))

    # get sampler

    args.model_path = os.path.join(args.train_path, args.benchmark,
                                   args.model_name, f'model_{args.epoch}.pt')
    sampler = get_generator(args, SRC, TRG, toklen_data, scaler, device)

    # get scaffold

    LOG.info('get scaffold...')

    if args.sample_from == 'train':
        scaffold_sample = get_sample(df_train, df_train, task_path,
                                     'train', n=args.n_scaffolds)
    elif args.sample_from == 'test_scaffolds':
        scaffold_sample = get_sample(df_test_scaffolds, df_train, task_path,
                                     'test_scaffolds', n=args.n_scaffolds)
    
    # sample conditioned by a scaffold

    if args.molgpt:
        save_folder = os.path.join(task_path, f'{args.model_name}-{args.epoch}',
                                   f'{args.sample_from}-molgpt')
    else:
        save_folder = os.path.join(task_path, f'{args.model_name}-{args.epoch}',
                                   args.sample_from)
    os.makedirs(save_folder, exist_ok=True)
    
    # plot_highlighted_smiles(scaffold_sample, save_folder, args.n_jobs)
    
    metric = OrderedDict()
    
    metric_path = os.path.join(save_folder, 'metric.csv')

    if os.path.exists(metric_path):
       _metric = pd.read_csv(metric_path, index_col=[0])
       metric = _metric.to_dict(orient='list') 
    else:
        for met in ('scaffold', 'valid', 'unique', 'novel', 'intDiv', 'sim', 'SSF', 'sim80',
                    'n_valid', 'n_unique', 'n_novel', 'n_SSF', 'n_sim80'):
            metric[met] = []

    for sid in range(len(scaffold_sample)):
        if sid < len(metric['valid']):
            continue

        gen_path = os.path.join(save_folder, f's{sid}_gen.csv')
        sim_fig_path = os.path.join(save_folder, f's{sid}_sim.png')

        n = args.n_samples
        trg_scaffold = scaffold_sample.loc[sid, 'scaffold']
        metric['scaffold'].append(trg_scaffold)

        LOG.info(f'sid: {sid}')
        LOG.info('sample smiles from scaffold: %s', trg_scaffold)

        if not os.path.exists(gen_path):
            samples = []

            while n > 0:
                print(f'n samples left: {n}')
                smiles, *_ = sampler.sample_smiles(n=min(n, args.batch_size),
                                                   scaffold=trg_scaffold)
                samples.extend(smiles)
                n -= len(smiles)

            samples = pd.DataFrame(samples, columns=['smiles'])
            samples.to_csv(gen_path)

        samples = pd.read_csv(gen_path, index_col=[0])['smiles']

        LOG.info('compute metrics...')

        mols = mapper(get_mol, samples, args.n_jobs)
        mols = [m for m in mols if m is not None and isinstance(m, float) is False]
        valid_smi = mol_to_smi(mols, args.n_jobs)
        unique_smi = set(valid_smi)

        similarity_fn = partial(murcko_scaffold_similarity, smi_or_mol2=trg_scaffold)
        scaffold_similarity = mapper(similarity_fn, valid_smi, args.n_jobs)
        scaffold_similarity = [s for s in scaffold_similarity if s is not None]

        metric['valid'].append(len(valid_smi) / len(samples))
        metric['unique'].append(len(unique_smi) / len(valid_smi))
        metric['novel'].append(len(unique_smi - train_set) / len(unique_smi))
        metric['intDiv'].append(metrics.internal_diversity(list(unique_smi), args.n_jobs))
        metric['sim'].append(sum(scaffold_similarity) / len(scaffold_similarity))
        metric['SSF'].append(len([1 for s in scaffold_similarity
                             if s == 1]) / len(scaffold_similarity))
        metric['sim80'].append(len([1 for s in scaffold_similarity
                               if s >= 0.80]) / len(scaffold_similarity))

        metric['n_valid'].append(len(valid_smi))
        metric['n_unique'].append(len(unique_smi))
        metric['n_novel'].append(len(unique_smi - train_set))
        metric['n_SSF'].append(len([1 for s in scaffold_similarity if s == 1]))
        metric['n_sim80'].append(len([1 for s in scaffold_similarity if s >= 0.80]))

        LOG.info('save metrics...')

        _metric = pd.DataFrame(metric)
        _metric.to_csv(metric_path)

        LOG.info('plot Murcko scaffold similarity...')

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5.5, 5.5))

        sns.kdeplot(data=scaffold_similarity, ax=ax,
                    shade=True, linewidth=2, legend=False)
        ax.set_xlabel(xlabel='Murcko scaffold similarity', fontsize=17)
        ax.set_ylabel(ylabel='Density', fontsize=17)
        ax.tick_params(axis="both", which="major", labelsize=14)
        fig.savefig(sim_fig_path, bbox_inches="tight")

    # plot murcko scaffold similarity

    LOG.info('plot all Murcko scaffold similarity...')

    gathered_sim = OrderedDict()

    for sid in range(len(scaffold_sample)):
        # if sid == 20:
        #     break
 
        LOG.info(f'gather data from sid: {sid}')
    
        gen_path = os.path.join(save_folder, f's{sid}_gen.csv')

        trg_scaffold = scaffold_sample.loc[sid, 'scaffold']
        samples = pd.read_csv(gen_path, index_col=[0])['smiles']
        mols = mapper(get_mol, samples, args.n_jobs)
        mols = [m for m in mols if m is not None and isinstance(m, float) is False]
        valid_smi = mol_to_smi(mols, args.n_jobs)

        similarity_fn = partial(murcko_scaffold_similarity, smi_or_mol2=trg_scaffold)
        scaffold_similarity = mapper(similarity_fn, valid_smi, args.n_jobs)
        scaffold_similarity = [s for s in scaffold_similarity if s is not None]
        scaffold_similarity = random.sample(scaffold_similarity, 1000)

        gathered_sim[sid] = scaffold_similarity

    gathered_sim = pd.DataFrame.from_dict(gathered_sim,
                                          orient="index").transpose()
    
    LOG.info('plot...')

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5.5, 5.5))

    for sid in range(len(scaffold_sample)):
        sns.kdeplot(data=gathered_sim[sid], ax=ax, shade=True, linewidth=2,
                    legend=False)
        ax.set_xlabel(xlabel='Murcko scaffold similarity', fontsize=17)
        ax.set_ylabel(ylabel='Density', fontsize=17)
        ax.tick_params(axis="both", which="major", labelsize=14)
    ax.set_xlim(0, 1.00)
    fig.savefig(os.path.join(save_folder, 'sim.png'), bbox_inches="tight")

    LOG.info('plot...')

    metric = pd.read_csv(metric_path, index_col=[0])
    n_metric = scaffold_sample[['n_train']].copy()
    n_metric[['n_unique', 'n_novel']] = metric[['n_unique', 'n_novel']].copy()
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5.5, 5.5))
    fig.set_dpi(300)

    # for i in range(len(n_metric)):
    plt.scatter(n_metric['n_train'] / 1000, n_metric['n_unique'] / 1000,
                c='blue', s=15, marker='o', label='# unique')
    plt.scatter(n_metric['n_train'] / 1000, n_metric['n_novel'] / 1000,
                c='orange', s=15, marker='^', label='# novel')
    plt.plot(np.arange(0, 11), np.arange(0, 11), c='black', linewidth=1)
    plt.xlabel('# seen in train (x$10^3$)', fontsize=17)
    plt.ylabel('# samples (x$10^3$)', fontsize=17)
    plt.tick_params(axis="both", which="major", labelsize=14)
    plt.legend(fontsize='14')
    plt.xlim(0, 10)
    plt.ylim(0, 10)

    plt.savefig(os.path.join(save_folder, 'diversity.png'))

