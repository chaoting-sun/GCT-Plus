import os
import csv
import torch
import moses
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from Model.build_model import get_sampler
from Utils import mapper, get_mol, get_canonical, \
    get_property_fn, mols_to_props


def sample_smiles(sampler, n, batch_size, LOG):
    gen = []
    while n > 0:
        LOG.info(f'# Samples left: {n}')
        current_gen, *_ = sampler.sample_smiles(min(n, batch_size))
        gen.extend(current_gen)
        n -= len(current_gen)
    return gen


def compute_metrics(gen, train, test, test_scaffolds, n_jobs):
    ptest = moses.dataset.get_statistics('test')
    ptest_scaffolds = moses.dataset.get_statistics('test_scaffolds')    

    metrics = moses.get_all_metrics(
        gen=gen,
        n_jobs=n_jobs,
        train=train,
        test=test,
        ptest=ptest,
        test_scaffolds=test_scaffolds,
        ptest_scaffolds=ptest_scaffolds
    )
    return metrics


def plot_descriptor_dist(gen_prop, test_prop, save_path):
    prop_list = ['logP', 'tPSA', 'QED', 'MW', 'SAS', 'NP']
    
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 8.5))

    for i, prop in enumerate(prop_list):
        rowi = i // 3
        coli = i % 3
        ax = axes[rowi, coli]

        df = pd.DataFrame({ 'gen' : gen_prop[prop], 'test': test_prop[prop] })
        sns.kdeplot(data=df, ax=ax, shade=True, linewidth=2.5, legend=False)

        ax.set_xlabel(xlabel=prop, fontsize=17)
        if coli == 0:
            ax.set_ylabel(ylabel='Density', fontsize=17)
        else:
            ax.set_ylabel(None)
        ax.tick_params(axis="both", which="major", labelsize=13)
    
    fig.legend(labels=['gen', 'test'], fontsize=16, loc='lower center',
               ncol=2, bbox_to_anchor=(0.5, -0.05))
    fig.savefig(save_path, bbox_inches="tight") 


def plot_descriptor_num(gen_prop, test_prop, save_path):
    feat_list = ['HAC', 'HBA', 'HBD', 'RBN', 'AIRN', 'ARRN']
    
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 8.5))

    for i, prop in enumerate(feat_list):
        rowi = i // 3
        coli = i % 3

        ax = axes[rowi, coli]

        gen_cnt = gen_prop[prop].value_counts(sort=False).sort_index()
        test_cnt = test_prop[prop].value_counts(sort=False).sort_index()

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
    fig.savefig(save_path, bbox_inches="tight")


@torch.no_grad()
def uc_sampling(
        args,
        train,
        test,
        test_scaffolds,
        toklen_data,
        scaler,
        SRC,
        TRG,
        device,
        logger
    ):    
    # define save path

    os.makedirs(args.save_folder, exist_ok=True)    
    
    LOG = logger(name='uc_sampling', log_path=os.path.join(args.save_folder, 'record.log'))
    sample_path = os.path.join(args.save_folder, 'gen.csv')
    prop_path = os.path.join(args.save_folder, 'prop.csv')
    metric_path = os.path.join(args.save_folder, 'metric.csv')
    test_sample_path = os.path.join(args.save_folder, 'test_samples.csv')
    descriptor_dist_path = os.path.join(args.save_folder, 'descriptor_dist.png')
    descriptor_num_path = os.path.join(args.save_folder, 'descriptor_num.png')
    
    # get sampler / property function
    
    args.model_path = os.path.join(args.model_folder, args.model_name)
    sampler = get_sampler(args, SRC, TRG, toklen_data, scaler, device)
    property_fn = get_property_fn(args.descriptor)

    # sample SMILES

    if not os.path.exists(sample_path):
        LOG.info('Sample SMILES')
        gen = sample_smiles(sampler, args.n_samples,
                            args.batch_size, LOG)
        gen = pd.DataFrame(gen, columns=['SMILES'])
        gen.to_csv(sample_path)
    
    gen = pd.read_csv(sample_path, index_col=[0])
    
    # compute metrics
    
    if not os.path.exists(metric_path):
        LOG.info('Compute metrics')
        
        metrics = compute_metrics(gen['SMILES'], 
                                  train['smiles'],
                                  test['smiles'],
                                  test_scaffolds['smiles'],
                                  args.n_jobs)
        LOG.info(metrics)
        
        with open(metric_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(list(metrics.keys()))
            writer.writerow(list(metrics.values()))

    # compute properties of gen

    if not os.path.exists(prop_path):
        LOG.info('Compute properties for gen')
        
        gen = gen.dropna(subset=['SMILES'])
        mols = mapper(get_mol, gen['SMILES'], args.n_jobs)
        mols = [m for m in mols if m is not None]
        smiles = mapper(get_canonical, mols, args.n_jobs)
        smiles = pd.DataFrame(smiles, columns=['SMILES'])
        gen_prop = mols_to_props(mols, property_fn, n_jobs=args.n_jobs)
        gen_prop = pd.concat([smiles, gen_prop], axis=1)
        gen_prop.to_csv(prop_path)

    # compute properties of test (30000)
    
    if not os.path.exists(test_sample_path):
        LOG.info('Compute properties for test')
        
        test_samples = random.sample(test['smiles'], 30000)
        mols = mapper(get_mol, test_samples, args.n_jobs)
        smiles = pd.DataFrame({ 'SMILES': test_samples })
        props = mols_to_props(mols, property_fn, n_jobs=args.n_jobs)
        test_prop = pd.concat([smiles, props], axis=1)
        test_prop.to_csv(test_sample_path)
    
    # plot distributions

    LOG.info('Plot molecular properties and structural features')

    test_prop = pd.read_csv(test_sample_path)
    gen_prop = pd.read_csv(prop_path)

    plot_descriptor_dist(gen_prop, test_prop, descriptor_dist_path)
    plot_descriptor_num(gen_prop, test_prop, descriptor_num_path)