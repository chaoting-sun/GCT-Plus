import os
import random
import numpy as np
import pandas as pd
from collections import OrderedDict
import scipy.stats as stats
from Utils.mapper import mapper
from Utils.smiles import get_mol, mol_to_smi
from Model.build_model import get_sampler
from Utils.properties import mols_to_props, get_property_fn
import matplotlib.pyplot as plt
import seaborn as sns
from guacamol.utils.chemistry import (
    calculate_pc_descriptors,
    calculate_internal_pairwise_similarities,
    continuous_kldiv,
    discrete_kldiv
)
from matplotlib.ticker import MaxNLocator
# from rdkit.ML.Descriptors import MoleculeDescriptors


# def sample_from_dataset(dataset, property_fn, n, n_jobs):
#     test_prop = random.sample(dataset, n)
#     mols = mapper(get_mol, test_prop, n_jobs)
#     smiles = pd.DataFrame({ 'SMILES': test_prop })
#     props = mols_to_props(mols, property_fn, n_jobs=n_jobs)
#     smiles_props = pd.concat([smiles, props], axis=1)
#     return smiles_props


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
    score = sum(partial_scores) / len(partial_scores)

    return score, kldivs


def model_selection(
        args,
        df_train,
        df_test,
        toklen_data,
        scaler,
        SRC,
        TRG,
        device
    ):
    
    md_task = {
        'vaetf'      : 'uc',
        'cvaetf'     : 'p',
        'scavaetf'   : 'sca',
        'scacvaetfv3': 'psca'
    }

    task = md_task[args.model_type]
    
    save_folder = os.path.join(args.infer_path,
                               args.benchmark,
                               f'model_selection-{task}',
                               f'{args.model_name}')
    os.makedirs(save_folder, exist_ok=True)

    n_batch = int(np.ceil(args.n_samples / args.batch_size))

    interested_properties = ['logP', 'tPSA', 'QED', 'MW', 'SAS', 'NP',
                             'HAC', 'HBA', 'HBD', 'RBN', 'AIRN', 'ARRN']
    property_fn = get_property_fn(interested_properties)

    property_path = os.path.join(save_folder, 'property_samples.csv')
    test_path = os.path.join(save_folder, 'test_samples.csv')
    kl_path = os.path.join(save_folder, f'{args.model_name}_kl.csv')

    # sample molecules from test dataset as the reference set

    # if not os.path.exists(test_path):
    test_samples = df_test.sample(n=args.n_samples, ignore_index=True)
    test_samples.to_csv(test_path)
        
    test_samples = pd.read_csv(test_path, index_col=[0])
    
    # sample properties from the property distribution of the training set
    
    if not os.path.exists(property_path):
        property_samples = []
        for p in args.property_list:
            property_samples.append(df_train[p].sample(n=args.n_samples, ignore_index=True))
        df_train = df_train.dropna(subset=['scaffold'])
        property_samples.append(df_train['scaffold'].sample(n=args.n_samples, ignore_index=True))

        property_samples = pd.concat(property_samples, axis=1)
        property_samples.to_csv(property_path)
    
    property_samples = pd.read_csv(property_path, index_col=[0])
    
    kldivs_dict = OrderedDict()

    for i, epoch in enumerate(args.epoch_list):
        gen_path = os.path.join(save_folder, f'{args.model_name}-{epoch}_gen.csv')
        args.model_path = os.path.join(args.train_path, args.benchmark,
                                       args.model_name, f'model_{epoch}.pt')
        generator = get_sampler(args, SRC, TRG, toklen_data,
                                  scaler, device)

        if not os.path.exists(gen_path):
            gen = []
            
            for b in range(n_batch):
                sid = args.batch_size * b
                eid = args.batch_size * (b+1)
                if eid > args.n_samples:
                    eid = args.n_samples
                print(f'sample: {sid} - {eid}')

                kwargs = {}
                
                if args.use_scaffold:
                    kwargs['scaffolds'] = property_samples['scaffold'].iloc[sid:eid]
                                
                if len(args.property_list) > 0:
                    kwargs['dconds'] = property_samples[args.property_list].iloc[sid:eid]
                else:
                    kwargs['n'] = eid - sid
                
                if args.model_type == 'scacvaetfv3':
                    current_gen, *_ = generator.sample_multiple_smiles(**kwargs)
                else:
                    current_gen, *_ = generator.sample_smiles(**kwargs)
                gen.extend(current_gen)

            gen = pd.DataFrame(gen, columns=['smiles'])
            gen.to_csv(gen_path)

        # compute kldiv scores
        
        gen = pd.read_csv(gen_path, index_col=[0])
        gen = gen.dropna(subset=['smiles'])

        print('compute kl divergence score...')

        mols = mapper(get_mol, gen['smiles'], args.n_jobs)
        mols = [m for m in mols if m is not None]
        unique_gen = set(mol_to_smi(mols, args.n_jobs))

        score, kldivs = compute_kldiv_score(unique_gen, test_samples['smiles'])
        
        if i == 0:
            for k, v in kldivs.items():
                kldivs_dict[k] = [v]
            kldivs_dict['Score'] = [score]
        else:
            for k, v in kldivs.items():
                kldivs_dict[k].append(v)
            kldivs_dict['Score'].append(score)
        
        print(kldivs_dict)

        df_kldivs = pd.DataFrame(kldivs_dict, index=args.epoch_list[:i+1])
        df_kldivs.to_csv(kl_path)
    
    # kl_val = pd.read_csv(kl_path, index_col=[0])
    # kl_average = kl_val.mean(axis=1)
    # best_epoch = kl_average.idxmin()

    # gen_prop = pd.read_csv(os.path.join(save_folder, f'{args.model_name}-{best_epoch}_prop.csv'),
    #                        index_col=[0])
    
    # fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(24, 4.2))

    # for i, prop in enumerate(interested_properties):
    #     ax = axes[i]
    #     sns.kdeplot(gen_prop[prop], ax=ax, shade=True, label='gen', linewidth=3)
    #     sns.kdeplot(test_prop[prop], ax=ax, shade=True, label='test', linewidth=3)

    #     ax.legend(fontsize=14, loc='best')
    #     ax.set_xlabel(xlabel=prop, fontsize=17)
    #     if i == 0:
    #         ax.set_ylabel(ylabel='Density', fontsize=17)
    #     else:
    #         ax.set_ylabel(None)
    #     ax.tick_params(axis="both", which="major", labelsize=13)
    
    # fig.savefig(os.path.join(save_folder, f'{args.model_name}-{best_epoch}_prop.png'), bbox_inches="tight") 

    for epoch in range(1,51):
        gen_path = os.path.join(save_folder, f'{args.model_name}-{epoch}_gen.csv')
        prop_path = os.path.join(save_folder, f'{args.model_name}-{epoch}_prop.csv')
        
        if not os.path.exists(gen_path):
            continue

        print('processing epoch:', epoch)
        
        gen = pd.read_csv(gen_path, index_col=[0])
        try:
            gen = gen.dropna(subset=['smiles'])
            mols = mapper(get_mol, gen['smiles'], args.n_jobs)
        except:
            gen = gen.dropna(subset=['SMILES'])
            mols = mapper(get_mol, gen['SMILES'], args.n_jobs)
        mols = [m for m in mols if m is not None]
        gen_prop = mols_to_props(mols, property_fn, args.n_jobs)
        gen_prop.to_csv(prop_path)

    print('compute gen prop...')

    xlimit = {
        'logP': [-5, 10],
        'tPSA': [0, 150],
        'QED' : [0, 1],
        'MW'  : [0, 600],
        'SAS' : [1, 10],
        'NP'  : [-5, 5],
    }

    print('compute test prop...')

    if not os.path.exists(os.path.join(save_folder, 'test_prop.csv')):
        test_samples = pd.read_csv(test_path, index_col=[0])
        mols = mapper(get_mol, test_samples['smiles'], args.n_jobs)
        mols = [m for m in mols if m is not None]
        test_prop = mols_to_props(mols, property_fn, args.n_jobs)
        test_prop.to_csv(os.path.join(save_folder, 'test_prop.csv'))

        test_prop = pd.read_csv(os.path.join(save_folder, 'test_prop.csv'), index_col=[0])

    # kld = os.path.join(save_folder, f'{args.model_name}_kl.csv')
    # kld = pd.read_csv(kl_path, index_col=[0])
    # best_epoch = kld['Score'].idxmax()

    print('plot prop...')

    for epoch in range(1,51):
        prop_path = os.path.join(save_folder, f'{args.model_name}-{epoch}_prop.csv')
        fig_dist_path = os.path.join(save_folder, f'{args.model_name}-{epoch}_dist.png')
        fig_num_path = os.path.join(save_folder, f'{args.model_name}-{epoch}_num.png')

        if not os.path.exists(prop_path):
            continue

        gen_prop = pd.read_csv(prop_path, index_col=[0])

        print('save figure:', fig_dist_path)

        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 8.5))

        for i, prop in enumerate(['logP', 'tPSA', 'QED', 'MW', 'SAS', 'NP']):
            rowi = i // 3
            coli = i % 3
            ax = axes[rowi, coli]

            df = pd.DataFrame({ 'gen': gen_prop[prop],
                                'test': test_prop[prop]
                                })
            sns.kdeplot(data=df, ax=ax, shade=True, linewidth=2.5, legend=False)
            
            # sns.kdeplot(gen_prop[prop], ax=ax, shade=True, linewidth=0, fill=True, color='orange')
            # sns.kdeplot(test_prop[prop], ax=ax, shade=True, linewidth=0, fill=True, color='blue')

            ax.set_xlim(left=xlimit[prop][0], right=xlimit[prop][1])

            ax.set_xlabel(xlabel=prop, fontsize=17)
            if coli == 0:
                ax.set_ylabel(ylabel='Density', fontsize=17)
            else:
                ax.set_ylabel(None)
            ax.tick_params(axis="both", which="major", labelsize=13)
        
        fig.legend(labels=['gen', 'test'], fontsize=16, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.05))
        fig.savefig(fig_dist_path, bbox_inches="tight") 

        print('save figure:', fig_num_path)

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
        fig.savefig(fig_num_path, bbox_inches="tight")














def model_selection(
        args,
        SRC,
        TRG,
        toklen_data,
        scaler,
        test,
        device
    ):
    interested_properties = ['logP', 'tPSA', 'QED', 'MW', 'SAS']
    property_fn = get_property_fn(interested_properties)

    save_folder = os.path.join(args.infer_path, args.benchmark,
                               'uc-sampling_search-best-model')
    os.makedirs(save_folder, exist_ok=True)

    kl_path = os.path.join(save_folder, f'{args.model_name}_kl.csv')
    test_sample_path = os.path.join(save_folder, 'test_samples.csv')        

    if not os.path.exists(test_sample_path):
        test_prop = random.sample(test, n)
        mols = mapper(get_mol, test_prop, args.n_jobs)
        smiles = pd.DataFrame({ 'SMILES': test_prop })
        props = mols_to_props(mols, property_fn, n_jobs=args.n_jobs)
        smiles_props = pd.concat([smiles, props], axis=1)
        smiles_props.to_csv(test_sample_path)

    test_prop = pd.read_csv(test_sample_path)

    kl_val = OrderedDict()

    for i, epoch in enumerate(args.epoch_list):
        print('model epoch:', epoch)

        args.model_path = os.path.join(args.train_path,
                                       args.benchmark,
                                       args.model_name,
                                       f'model_{epoch}.pt')
        generator = get_sampler(args, SRC, TRG, toklen_data,
                                  scaler, device)

        gen_path = os.path.join(save_folder, f'{args.model_name}-{epoch}_gen.csv')
        prop_path = os.path.join(save_folder, f'{args.model_name}-{epoch}_prop.csv')

        print('sample molecules')

        if not os.path.exists(gen_path):
            gen = []
            n = args.n_samples
            
            while n > 0:
                print('n samples left:', n)
                
                current_gen, *_ = generator.sample_smiles(n=min(n, args.batch_size))
                gen.extend(current_gen)
                n -= len(current_gen)

            gen = pd.DataFrame(gen, columns=['SMILES'])
            gen.to_csv(gen_path)

        gen = pd.read_csv(gen_path, index_col=[0])
        gen = gen.dropna(subset=['SMILES'])

        print('evaluate: compute properties')

        if not os.path.exists(prop_path):
            mols = mapper(get_mol, gen['SMILES'], args.n_jobs)
            mols = [m for m in mols if m is not None]
            smiles = pd.DataFrame(mol_to_smi(mols, args.n_jobs), columns=['SMILES'])
            
            props = mols_to_props(mols, property_fn, n_jobs=args.n_jobs)
            smiles_props = pd.concat([smiles, props], axis=1)
            smiles_props.to_csv(prop_path)

        gen_prop = pd.read_csv(prop_path)

        for prop in interested_properties:
            if prop not in kl_val:
                kl_val[prop] = []
            kl_val[prop].append(kl_divergence(test_prop[prop], gen_prop[prop]))

        if i == len(args.epoch_list)-1:
            kl_val = pd.DataFrame(kl_val, index=args.epoch_list)
            kl_val.to_csv(kl_path)
    
    kl_val = pd.read_csv(kl_path, index_col=[0])
    kl_average = kl_val.mean(axis=1)
    best_epoch = kl_average.idxmin()

    gen_prop = pd.read_csv(os.path.join(save_folder, f'{args.model_name}-{best_epoch}_prop.csv'),
                           index_col=[0])
    
    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(24, 4.2))

    for i, prop in enumerate(interested_properties):
        ax = axes[i]
        sns.kdeplot(gen_prop[prop], ax=ax, shade=True, label='gen', linewidth=3)
        sns.kdeplot(test_prop[prop], ax=ax, shade=True, label='test', linewidth=3)

        ax.legend(fontsize=14, loc='best')
        ax.set_xlabel(xlabel=prop, fontsize=17)
        if i == 0:
            ax.set_ylabel(ylabel='Density', fontsize=17)
        else:
            ax.set_ylabel(None)
        ax.tick_params(axis="both", which="major", labelsize=13)
    
    fig.savefig(os.path.join(save_folder, f'{args.model_name}-{best_epoch}_prop.png'), bbox_inches="tight") 

