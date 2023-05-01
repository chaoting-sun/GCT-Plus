import os
import numpy as np
import pandas as pd
import random
import itertools
from Model.build_model import get_generator
from Utils.properties import mols_to_props, get_property_fn
from Utils.smiles import get_mol, mol_to_smi
from Utils.mapper import mapper
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict


def get_trg_prop(property_list):
    trg_prop = {
        'logP': [ 1.0,   2.0,  3.0],
        'tPSA': [30.0,  60.0, 90.0],
        'QED' : [ 0.6, 0.725, 0.85],
        'SAS' : [ 2.0,  2.75,  3.5],
    }
    prop_set = (trg_prop[p] for p in property_list)
    prop_comb = list(itertools.product(*prop_set))
    return [list(c) for c in prop_comb]


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


def sample_smiles(generator, trg_prop, n_samples,
                  batch_size=512):
    samples = []
    dconds = np.tile(trg_prop, (batch_size, 1))
    n = n_samples
    
    while n > 0:
        print(f'n samples left: {n}')
        
        k = min(n, batch_size)
        current_samples, *_ = generator.sample_smiles(dconds[:k])
        samples.extend(current_samples)
        n -= len(current_samples)
    return samples


def prop_sampling(
        args,
        df_train,
        df_test,
        toklen_data,
        scaler,
        SRC,
        TRG,
        device,
        logger
    ):    

    n_samples = 10000

    trg_prop_list = get_trg_prop(args.property_list)
    
    df_trg_prop = pd.DataFrame(trg_prop_list, columns=args.property_list)
    df_trg_prop.to_csv(os.path.join(args.infer_path, args.benchmark, 'prop_sampling',
                                    f'property_{"-".join(args.property_list)}.csv'))

    interested_properties = ['logP', 'tPSA', 'QED',
                              'SAS',   'NP',  'MW',
                              'HAC',  'HBA', 'HBD']
    property_fn = get_property_fn(interested_properties)

    # create file path and folder
    save_folder = os.path.join(args.infer_path, args.benchmark,
                               'prop_sampling', args.model_name)
    os.makedirs(save_folder, exist_ok=True)

    LOG = logger(name='scaffold sampling', log_path=os.path.join(save_folder, 'record.log'))

    LOG.info('create a generator...')
    
    args.model_path = os.path.join(args.train_path,
                                   args.benchmark,
                                   args.model_name,
                                   f'model_{args.epoch}.pt')
    generator = get_generator(args, SRC, TRG, toklen_data,
                              scaler, device)

    print('create a save folder')

    save_folder = os.path.join(args.infer_path, args.benchmark, 'prop_sampling',
                               f'{args.model_name}_{args.epoch}')
    os.makedirs(save_folder, exist_ok=True)

    property_path = os.path.join(save_folder, f'property.csv')
    metric_path = os.path.join(save_folder, f'metric.csv')
    
    # print('Files to save:', gen_path, property_path, metric_path)

    print('sample molecules')

    error_path = os.path.join(save_folder, f'{args.model_name}-{args.epoch}_error.csv')
    errors = OrderedDict()

    for i, trg_prop in enumerate(trg_prop_list):
        print('properties:', trg_prop)
        
        gen_path = os.path.join(save_folder, f'{args.model_name}-{args.epoch}_gen_{i}.csv')
        property_path = os.path.join(save_folder, f'{args.model_name}-{args.epoch}_prop_{i}.csv')
        
        print('generate: sample smiles')
        
        if not os.path.exists(gen_path):
            gen = sample_smiles(generator, trg_prop, n_samples)
            gen = pd.DataFrame(gen, columns=['SMILES'])
            gen.to_csv(gen_path)

        print('evaluate: compute properties')

        gen = pd.read_csv(gen_path, index_col=[0])
        gen = gen.dropna(subset=['SMILES'])

        if not os.path.exists(property_path):
            mols = mapper(get_mol, gen['SMILES'], args.n_jobs)
            mols = [m for m in mols if m is not None]
            smiles = pd.DataFrame(mol_to_smi(mols, args.n_jobs), columns=['SMILES'])
            props = mols_to_props(mols, property_fn, n_jobs=args.n_jobs)
            props = pd.concat([smiles, props], axis=1)
            props.to_csv(property_path)

        # print('sample generated smiles')
        # gen_samples = pd.read_csv(os.path.join(save_folder, f'property.csv'))

        # for prop in interested_properties:
        #     stat_plot = {}
            
        #     stat_plot['gen'] = gen_samples[prop]
        #     stat_plot['train'] = train_samples[prop][:len(stat_plot['gen'])]
        #     stat_plot = pd.DataFrame(stat_plot)
            
        #     kde_plot(stat_plot, os.path.join(save_folder, f'{prop}.png'),
        #             xlabel=prop, ylabel='Density', xlimit=None)

        print('compute errors')

        props = pd.read_csv(property_path, index_col=[0])
        
        for j, p in enumerate(args.property_list):
            delp = props[p] - trg_prop[j]
            mse = delp.mean()
            mae = delp.abs().mean()
            sd = delp.std()
            
            if i == 0:                
                errors[f'{p}-MSE'] = [mse]
                errors[f'{p}-MAE'] = [mae]
                errors[f'{p}-SD'] = [sd]
            else:
                errors[f'{p}-MSE'].append(mse)
                errors[f'{p}-MAE'].append(mae)
                errors[f'{p}-SD'].append(sd)

        df_errors = pd.DataFrame(errors)
        df_errors.to_csv(error_path)
        
        print(errors)


