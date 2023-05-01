import os
import numpy as np
import pandas as pd
from Inference.utils import prepare_generator
import torch
import moses
import csv
from time import time
from Model.build_model import get_generator
from Utils import DataloaderPreparation
from Utils.properties import get_property_fn, property_type, mols_to_props
from Utils.field import smi_to_id
from collections import OrderedDict
import random
from Utils.smiles import (
    get_mol,
    mol_to_smi,
    murcko_scaffold,
    tanimoto_similarity,
    plot_smiles_group,
    plot_smiles,
    murcko_scaffold_similarity
)
from Utils.mapper import mapper
from Utils.seed import set_seed
from matplotlib.ticker import MaxNLocator

import matplotlib.pyplot as plt
import seaborn as sns


def kde_plot(df, save_path, xlabel, ylabel, xlimit=None,
             figsize=(6.5, 5)):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    
    for c in df.columns:
        sns.kdeplot(df[c], ax=ax, shade=True, label=c, linewidth=3)
    
    # df.plot.kde(ax=ax, legend=True, xlim=xlimit)
    ax.legend(fontsize=14)
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    fig.savefig(save_path, bbox_inches="tight") 


import scipy.stats as stats

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


# def sampling_to_search_best_model(args, SRC, TRG, toklen_data,
#                                     scaler, test, device):
#     n_samples = 10000
#     batch_size = 512
#     epoch_list = [5, 10, 15, 20, 25, 30, 31, 32, 33, 34, 35]
#     # epoch_list = [5, 10, 15, 20, 25, 30, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45]
#     # epoch_list = []

#     interested_properties = ['logP', 'tPSA', 'QED', 'MW', 'SAS']
#     property_fn = get_property_fn(interested_properties)

#     save_folder = os.path.join(args.infer_path, args.benchmark,
#                                'uc-sampling_search-best-model')
#     os.makedirs(save_folder, exist_ok=True)

#     kl_path = os.path.join(save_folder, f'{args.model_name}_kl.csv')
#     test_sample_path = os.path.join(save_folder, 'test_samples.csv')        

#     if not os.path.exists(test_sample_path):
#         test_prop = random.sample(test, n)
#         mols = mapper(get_mol, test_prop, args.n_jobs)
#         smiles = pd.DataFrame({ 'SMILES': test_prop })
#         props = mols_to_props(mols, property_fn, n_jobs=args.n_jobs)
#         smiles_props = pd.concat([smiles, props], axis=1)
#         smiles_props.to_csv(test_sample_path)

#     test_prop = pd.read_csv(test_sample_path)

#     kl_val = OrderedDict()

#     for i, epoch in enumerate(epoch_list):
#         print('model epoch:', epoch)

#         args.model_path = os.path.join(args.train_path,
#                                        args.benchmark,
#                                        args.model_name,
#                                        f'model_{epoch}.pt')
#         generator = get_generator(args, SRC, TRG, toklen_data,
#                                   scaler, device)

#         gen_path = os.path.join(save_folder, f'{args.model_name}-{epoch}_gen.csv')
#         property_path = os.path.join(save_folder, f'{args.model_name}-{epoch}_prop.csv')

#         print('sample molecules')

#         if not os.path.exists(gen_path):
#             gen = []
#             n = n_samples
            
#             while n > 0:
#                 print('n samples left:', n)
                
#                 current_gen, *_ = generator.sample_smiles(n=min(n, batch_size))
#                 gen.extend(current_gen)
#                 n -= len(current_gen)

#             gen = pd.DataFrame(gen, columns=['SMILES'])
#             gen.to_csv(gen_path)

#         gen = pd.read_csv(gen_path, index_col=[0])
#         gen = gen.dropna(subset=['SMILES'])

#         print('evaluate: compute properties')

#         if not os.path.exists(property_path):
#             mols = mapper(get_mol, gen['SMILES'], args.n_jobs)
#             mols = [m for m in mols if m is not None]
#             smiles = pd.DataFrame(mol_to_smi(mols, args.n_jobs), columns=['SMILES'])
            
#             props = mols_to_props(mols, property_fn, n_jobs=args.n_jobs)
#             smiles_props = pd.concat([smiles, props], axis=1)
#             smiles_props.to_csv(property_path)

#         gen_prop = pd.read_csv(property_path)

#         for prop in interested_properties:
#             if prop not in kl_val:
#                 kl_val[prop] = []
#             kl_val[prop].append(kl_divergence(test_prop[prop], gen_prop[prop]))

#         if i == len(epoch_list)-1:
#             kl_val = pd.DataFrame(kl_val, index=epoch_list)
#             kl_val.to_csv(kl_path)
    
#     kl_val = pd.read_csv(kl_path, index_col=[0])
#     kl_average = kl_val.mean(axis=1)
#     best_epoch = kl_average.idxmin()

#     gen_prop = pd.read_csv(os.path.join(save_folder, f'{args.model_name}-{best_epoch}_prop.csv'),
#                            index_col=[0])
    
#     fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(24, 4.2))

#     for i, prop in enumerate(interested_properties):
#         ax = axes[i]
#         sns.kdeplot(gen_prop[prop], ax=ax, shade=True, label='gen', linewidth=3)
#         sns.kdeplot(test_prop[prop], ax=ax, shade=True, label='test', linewidth=3)

#         ax.legend(fontsize=14, loc='best')
#         ax.set_xlabel(xlabel=prop, fontsize=17)
#         if i == 0:
#             ax.set_ylabel(ylabel='Density', fontsize=17)
#         else:
#             ax.set_ylabel(None)
#         ax.tick_params(axis="both", which="major", labelsize=13)
    
#     fig.savefig(os.path.join(save_folder, f'{args.model_name}-{best_epoch}_prop.png'), bbox_inches="tight") 
    

@torch.no_grad()
def unconditioned_sampling(
        args,
        train,
        test,
        test_scaffolds,
        toklen_data,
        scaler,
        SRC,
        TRG,
        device
    ):
        
    # if True:
    #     sampling_to_search_best_model(args, SRC, TRG, toklen_data,
    #                                   scaler, test, device)
    #     exit()

    print('create a generator')

    args.model_path = os.path.join(args.train_path, args.benchmark,
                                   args.model_name, f'model_{args.epoch}.pt')
    # generator = get_generator(args, SRC, TRG, toklen_data, scaler, device)
    generator = None

    print('generate molecules')

    batch_size = 512
    interested_properties = ['logP', 'tPSA', 'QED', 'MW', 'SAS', 'NP',
                             'HAC', 'HBA', 'HBD', 'RBN', 'AIRN', 'ARRN']
    
    property_fn = get_property_fn(interested_properties)
    ptest = moses.dataset.get_statistics('test')
    ptest_scaffolds = moses.dataset.get_statistics('test_scaffolds')
    
    if False:
        # required_toklen = 25
        required_toklen = None

        print('create a save folder')

        suffix = 'random' if required_toklen is None else required_toklen
        save_folder = os.path.join(args.infer_path, args.benchmark,
                                   f'uc-sampling_{suffix}')
        os.makedirs(save_folder, exist_ok=True)

        sample_path = os.path.join(save_folder, f'{args.model_name}-{args.epoch}_gen.csv')
        property_path = os.path.join(save_folder, f'{args.model_name}-{args.epoch}_prop.csv')
        metric_path = os.path.join(save_folder, f'{args.model_name}-{args.epoch}_metric.csv')
        
        print('Files to save:', sample_path, property_path, metric_path)

        print('sample molecules')

        n = 10000
        samples = []
        toklens = []
        toklen_gens = []

        while n > 0:
            print(f'n samples left: {n}')

            k = min(n, batch_size)

            if required_toklen is None:
                toklen = generator.sample_toklen(k)
            else:
                toklen = [required_toklen] * k
            
            current_samples, current_toklen, current_toklen_gen = generator.sample_smiles(n=k, toklen=toklen)
            # current_samples, current_toklen, current_toklen_gen = generator.sample_smiles(n=k, toklen=toklen)

            # current_samples, *_ = generator.sample_smiles(min(n, batch_size))
            samples.extend(current_samples)
            toklens.extend(current_toklen)
            toklen_gens.extend(current_toklen_gen)
            n -= len(current_samples)

        samples = pd.DataFrame(samples, columns=['SMILES'])
        samples.to_csv(sample_path)

        print('evaluate: compute properties')

        gen = pd.read_csv(sample_path, index_col=[0])
        gen = gen.dropna(subset=['SMILES'])

        # if not os.path.exists(property_path):
        mols = mapper(get_mol, gen['SMILES'], args.n_jobs)
        mols = [m for m in mols if m is not None]
        smiles = pd.DataFrame(mol_to_smi(mols, args.n_jobs), columns=['SMILES'])
        props = mols_to_props(mols, property_fn, n_jobs=args.n_jobs)
        props = pd.concat([smiles, props], axis=1)
        props.to_csv(property_path)
        
        print('property saved')

        print('evaluate: compute metrics')

        print('sample train smiles')
        
        test_sample_path = os.path.join(save_folder, 'test_samples.csv')
        
        if not os.path.exists(test_sample_path):
            test_samples = random.sample(test, n)
            mols = mapper(get_mol, test_samples, args.n_jobs)
            smiles = pd.DataFrame({ 'SMILES': test_samples })
            props = mols_to_props(mols, property_fn, n_jobs=args.n_jobs)
            smiles = pd.concat([smiles, props], axis=1)
            smiles.to_csv(test_sample_path)
        
        train_samples = pd.read_csv(test_sample_path)

        print('sample generated smiles')

        gen_samples = pd.read_csv(property_path)

        for prop in interested_properties:
            stat_plot = {}
            
            stat_plot['gen'] = gen_samples[prop]
            stat_plot['train'] = train_samples[prop][:len(stat_plot['gen'])]
            stat_plot = pd.DataFrame(stat_plot)
            
            kde_plot(stat_plot, os.path.join(save_folder, f'{prop}.png'),
                    xlabel=prop, ylabel='Density', xlimit=None)

    else:
        print('create a save folder')

        save_folder = os.path.join(args.infer_path, args.benchmark, 'uc-sampling')
        os.makedirs(save_folder, exist_ok=True)

        n = args.n_samples

        sample_path = os.path.join(save_folder, f'{args.model_name}-{args.epoch}_gen.csv')
        property_path = os.path.join(save_folder, f'{args.model_name}-{args.epoch}_prop.csv')
        metric_path = os.path.join(save_folder, f'{args.model_name}-{args.epoch}_metric.csv')

        print('Files to save:', sample_path, property_path, metric_path)

        print('sample molecules')

        if not os.path.exists(sample_path):
            samples = []

            while n > 0:
                print(f'n samples left: {n}')

                current_samples, *_ = generator.sample_smiles(min(n, batch_size))
                samples.extend(current_samples)
                n -= len(current_samples)

            samples = pd.DataFrame(samples, columns=['SMILES'])
            samples.to_csv(sample_path)

        print('evaluate: compute properties')

        gen = pd.read_csv(sample_path, index_col=[0])
        gen = gen.dropna(subset=['SMILES'])

        if not os.path.exists(property_path):
            mols = mapper(get_mol, gen['SMILES'], args.n_jobs)
            mols = [m for m in mols if m is not None]
            smiles = pd.DataFrame(mol_to_smi(mols, args.n_jobs), columns=['SMILES'])
            props = mols_to_props(mols, property_fn, n_jobs=args.n_jobs)
            props = pd.concat([smiles, props], axis=1)
            props.to_csv(property_path)
        
        print('property saved')

        print('evaluate: compute metrics')

        if not os.path.exists(metric_path):
            metrics = moses.get_all_metrics(
                gen=gen['SMILES'],
                n_jobs=args.n_jobs,
                train=train,
                test=test,
                ptest=ptest,
                test_scaffolds=test_scaffolds,
                ptest_scaffolds=ptest_scaffolds
            )

            print(metrics)

            with open(metric_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(list(metrics.keys()))
                writer.writerow(list(metrics.values()))
        
        print('metric saved')
        
        print('sample test smiles')
        
        test_sample_path = os.path.join(save_folder, 'test_samples.csv')
        
        if not os.path.exists(test_sample_path):
            test_samples = random.sample(test, 30000)
            mols = mapper(get_mol, test_samples, args.n_jobs)
            smiles = pd.DataFrame({ 'SMILES': test_samples })
            props = mols_to_props(mols, property_fn, n_jobs=args.n_jobs)
            test_prop = pd.concat([smiles, props], axis=1)
            test_prop.to_csv(test_sample_path)
        
        test_prop = pd.read_csv(test_sample_path)
        gen_prop = pd.read_csv(property_path)

        print('plot')

        def plot_distributions(gen_prop, test_prop, save_path):
            fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 8.5))

            for i, prop in enumerate(['logP', 'tPSA', 'QED', 'MW', 'SAS', 'NP']):
                rowi = i // 3
                coli = i % 3
                ax = axes[rowi, coli]

                df = pd.DataFrame({ 'gen': gen_prop[prop],
                                    'test': test_prop[prop]
                                })
                sns.kdeplot(data=df, ax=ax, shade=True, linewidth=2.5, legend=False)
                
                # print(df)

                # sns.kdeplot(gen_prop[prop], ax=ax, shade=True, linewidth=0, fill=True, color='orange')
                # sns.kdeplot(test_prop[prop], ax=ax, shade=True, linewidth=0, fill=True, color='blue')

                ax.set_xlabel(xlabel=prop, fontsize=17)
                if coli == 0:
                    ax.set_ylabel(ylabel='Density', fontsize=17)
                else:
                    ax.set_ylabel(None)
                ax.tick_params(axis="both", which="major", labelsize=13)
            
            fig.legend(labels=['gen', 'test'], fontsize=16, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.05))
            fig.savefig(save_path, bbox_inches="tight") 

        def plot_numbers(gen_prop, test_prop, save_path):
            fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 8.5))

            for i, prop in enumerate(['HAC', 'HBA', 'HBD', 'RBN', 'AIRN', 'ARRN']):
                rowi = i // 3
                coli = i % 3
                
                print(rowi, coli)

                ax = axes[rowi, coli]

                gen_cnt = gen_prop[prop].value_counts(sort=False).sort_index()
                test_cnt = test_prop[prop].value_counts(sort=False).sort_index()

                gen_cnt = gen_cnt / gen_cnt.sum()
                test_cnt = test_cnt / test_cnt.sum()

                gen_cnt.plot(kind='bar', ax=ax, color='blue', alpha=0.7, rot=0)
                test_cnt.plot(kind='bar', ax=ax, color='orange', alpha=0.7, rot=0)

                # sns.histplot(x=gen_cnt.index, y=gen_cnt, ax=ax, alpha=0.8, edgecolor='blue', color='blue', )
                # sns.histplot(x=test_cnt.index, y=test_cnt, ax=ax, alpha=0.8, edgecolor='orange', color='orange')

                # sns.histplot(df, ax=ax, kde=True, alpha=0.8)

                # ax.hist(df, histtype='stepfilled', alpha=0.5, density=1, bins=10)

                # ax.hist(gen_prop[prop].astype('int'), label='gen', **kwargs)
                # ax.hist(test_prop[prop].astype('int'), label='test', **kwargs)

                # sns.histplot(gen_prop[prop], ax=ax, label='gen', kde=True, color='blue', edgecolor='white', binwidth=1, alpha=0.7)
                # sns.histplot(test_prop[prop], ax=ax, label='test', kde=True, color='orange', edgecolor='white', binwidth=1, alpha=0.7)

                ax.set_xlabel(xlabel=prop, fontsize=17)
                if coli == 0:
                    ax.set_ylabel(ylabel='Density', fontsize=17)
                else:
                    ax.set_ylabel(None)
                ax.tick_params(axis="both", which="major", labelsize=13)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            
            fig.legend(labels=['gen', 'test'], fontsize=16, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.05))
            fig.savefig(save_path, bbox_inches="tight")

        plot_distributions(gen_prop, test_prop, os.path.join(save_folder, f'{args.model_name}-{args.epoch}_dist.png'))
        plot_numbers(gen_prop, test_prop, os.path.join(save_folder, f'{args.model_name}-{args.epoch}_num.png'))

        # get average properties from 3 generated sets

        model_name = ['vaetf1', 'vaetf2', 'vaetf3']
        best_epochs = [37, 38, 37]

        for i in range(3):       
            curr_prop = pd.read_csv(os.path.join(save_folder, f'{model_name[i]}-{best_epochs[i]}_prop.csv'), index_col=[0])
            curr_prop = curr_prop.sample(n=10000)
            if i == 0:
                gen_prop = curr_prop.copy()
            else:
                gen_prop = pd.concat([gen_prop, curr_prop], axis=0)

        gen_prop = gen_prop.reset_index(drop=True)

        plot_distributions(gen_prop, test_prop, os.path.join(save_folder, f'{args.model_type}_dist.png'))
        plot_numbers(gen_prop, test_prop, os.path.join(save_folder, f'{args.model_type}_num.png'))

                


