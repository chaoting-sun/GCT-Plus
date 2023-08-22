import os
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from moses.metrics import metrics
from collections import OrderedDict

from Model.build_model import get_sampler
from Configuration.config_default import selected_target_prop, prop_tolerance
from Utils import get_mol, get_canonical, mols_to_props, \
    mapper, plot_smiles_group, get_property_fn
    

def get_trg_prop_combination(property_list):
    prop_set = (selected_target_prop[p] for p in property_list)
    prop_comb = list(itertools.product(*prop_set))
    trg_prop_list = [list(c) for c in prop_comb]
    return np.array(trg_prop_list)

def sample_smiles(sampler, n, trg_prop, batch_size, LOG):
    gen = []
    dconds = np.tile(trg_prop, (batch_size, 1))

    while n > 0:
        LOG.info(f'# Samples left: {n}')
        
        k = min(n, batch_size)
        current_gen, *_ = sampler.sample_smiles(dconds[:k])
        gen.extend(current_gen)
        n -= len(current_gen)
    return gen


def get_n_train_near_prop(trg_prop_comb, property_list,
                          tolerance, train):
    n_train = []
    for prop in trg_prop_comb:
        filtered = train.copy()
        for i, p in enumerate(property_list):
            filtered = filtered[(prop[i] - tolerance[p] <= filtered[p]) & 
                            (filtered[p] <= prop[i] + tolerance[p])]
        n_train.append(len(filtered))
    return n_train


def scatter_plot_3d(np_arr, labels, save_path):
    colors = ['r', 'b', 'g'] * 9

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')    
    ax.scatter(np_arr[:, 0], np_arr[:, 1], np_arr[:, 2],
               marker='o', s=40, c=colors)

    ax.set_xlabel(labels[0], fontsize=14, labelpad=7)
    ax.set_ylabel(labels[1], fontsize=14, labelpad=7)
    ax.set_zlabel(labels[2], fontsize=14, labelpad=7)
    
    ax.tick_params(axis='x', which='major', labelsize=12, width=4)
    ax.tick_params(axis='y', which='major', labelsize=12, width=4)
    ax.tick_params(axis='z', which='major', labelsize=12, width=4)

    plt.savefig(save_path)


# def kde_plot(df, save_path, xlabel, ylabel, xlimit=None,
#              figsize=(6.5, 5)):
#     fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    
#     for c in df.columns:
#         sns.kdeplot(df[c], ax=ax, shade=True, label=c, linewidth=3)
    
#     # df.plot.kde(ax=ax, legend=True, xlim=xlimit)
#     ax.legend(fontsize=14)
#     ax.set_xlabel(xlabel, fontsize=20)
#     ax.set_ylabel(ylabel, fontsize=20)
#     plt.xticks(fontsize=16)
#     plt.yticks(fontsize=16)
#     fig.savefig(save_path, bbox_inches="tight")


# def plot_3d_figure(x, y, z, save_path):
#     # Create a 3D scatter plot
#     fig = plt.figure(figsize=(10, 10))
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(x, y, z, c='b', marker='o', alpha=0.6)

#     # Compute the KDEs
#     kde_xy = kde.gaussian_kde(np.vstack([x, y]))
#     kde_yz = kde.gaussian_kde(np.vstack([y, z]))
#     kde_xz = kde.gaussian_kde(np.vstack([x, z]))

#     # Create a grid for the KDE projections
#     grid_size = 100
#     xy_space = np.linspace(-3, 3, grid_size)
#     yz_space = np.linspace(-3, 3, grid_size)
#     xz_space = np.linspace(-3, 3, grid_size)
#     xy_grid, yz_grid = np.meshgrid(xy_space, yz_space)
#     xz_grid, _ = np.meshgrid(xz_space, xy_space)

#     # Compute the KDE values at the grid points
#     xy_kde_vals = kde_xy(np.vstack([xy_grid.ravel(), yz_grid.ravel()])).reshape(grid_size, grid_size)
#     yz_kde_vals = kde_yz(np.vstack([yz_grid.ravel(), xz_grid.ravel()])).reshape(grid_size, grid_size)
#     xz_kde_vals = kde_xz(np.vstack([xy_grid.ravel(), xz_grid.ravel()])).reshape(grid_size, grid_size)

#     # Plot the KDE projections on the respective planes
#     ax.contourf(xy_grid, yz_grid, xy_kde_vals, zdir='z', offset=np.min(z), alpha=0.6, cmap='viridis')
#     ax.contourf(yz_grid, xz_grid, yz_kde_vals, zdir='x', offset=np.min(x), alpha=0.6, cmap='viridis')
#     ax.contourf(xy_grid, xz_grid, xz_kde_vals, zdir='y', offset=np.min(y), alpha=0.6, cmap='viridis')

#     # Set axis labels
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
    
#     plt.savefig(save_path)


# plot_good_smiles(args.model_name, args.epoch, save_folder, trg_prop_list,
#                  args.property_list, train)


# def plot_good_smiles(model_name, epoch, save_folder, trg_prop_list,
#                      property_list, train, n=25):
#     trg_prop = [3.0, 60.0, 0.725]
    
#     prop_no = -1
#     for i, p in enumerate(trg_prop_list):
#         if p == trg_prop:
#             prop_no = i
#     if prop_no == -1:
#         exit('find nothing')

#     gen = pd.read_csv(os.path.join(save_folder, f'{model_name}-{epoch}_prop_{prop_no}.csv'),
#                       index_col=[0])
#     for i, p in enumerate(property_list):
#         gen[f'{p}-normalized_AE'] = gen[p].apply(lambda x: abs(x - float(trg_prop[i]))) / (train[p].max() - train[p].min())
#     gen = gen.sort_values(by=[f'{p}-normalized_AE' for p in property_list],
#                           ignore_index=True)
#     smiles_list = gen['smiles'].iloc[:n].tolist()
#     gen[property_list].iloc[:n].to_numpy()

#     descriptions = []
#     for i in range(n):
#         p = gen.loc[i, property_list].tolist()
#         p = f'logP: {p[0]:.2f}, tPSA: {p[1]:.1f}, QED: {p[2]:.2f}'
#         descriptions.append(p)
#     print(smiles_list)

#     save_path = os.path.join(save_folder, f'good_smiles_{prop_no}.png')
#     plot_smiles_group(smiles_list, save_path, n_per_mol=5, descriptions=descriptions,
#                       img_size=(310, 220), n_jobs=1)
#     exit()


def p_sampling(
        args,
        train,
        toklen_data,
        scaler,
        SRC,
        TRG,
        device,
        logger
    ):
    # define save path
    
    os.makedirs(args.save_folder, exist_ok=True)
    LOG = logger(name='p_sampling', log_path=os.path.join(args.save_folder, 'record.log'))

    cond_png_path = os.path.join(args.save_folder, 'condition.png')
    cond_val_path = os.path.join(args.save_folder, f'condition_{"-".join(args.property_list)}.csv')
    metric_path = os.path.join(args.save_folder, 'metric.csv')
    prop_dist_path = os.path.join(args.save_folder, 'prop_distribution.png')
    
    # get sampler

    sampler = get_sampler(args, SRC, TRG, toklen_data, scaler, device)

    # property conditions
    
    trg_prop_comb = get_trg_prop_combination(args.property_list)
    scatter_plot_3d(trg_prop_comb, args.property_list, cond_png_path)
    pd.DataFrame(trg_prop_comb, columns=args.property_list).to_csv(cond_val_path)

    interested_properties = ['logP', 'tPSA', 'QED',
                              'SAS',   'NP',  'MW',
                              'HAC',  'HBA', 'HBD']
    property_fn = get_property_fn(interested_properties)

    # generate SMILES

    LOG.info('Sample molecules')

    for i, trg_prop in enumerate(trg_prop_comb):
        LOG.info('Property: %s', trg_prop)
        gen_path = os.path.join(args.save_folder, f'gen{i}.csv')

        if os.path.exists(gen_path):
            continue
        gen = sample_smiles(sampler, args.n_samples, trg_prop, args.batch_size, LOG)
        gen = pd.DataFrame(gen, columns=['smiles'])
        gen.to_csv(gen_path)

    # define metrics

    metric = OrderedDict()
    for met in ('valid', 'unique', 'novel', 'intDiv'):
        metric[met] = []
    for p in args.property_list:
        metric[f'{p}-MSE'] = []
        metric[f'{p}-MAE'] = []
        metric[f'{p}-SD'] = []
    for met in ('valid_in_tolerance', 'unique_in_tolerance'):
        metric[met] = []
    metric['n_train'] = []

    # compute properties and metrics

    if not os.path.exists(metric_path):
        n_train = get_n_train_near_prop(trg_prop_comb, args.property_list,
                                        prop_tolerance, train)

        for i, trg_prop in enumerate(trg_prop_comb):
            gen_path = os.path.join(args.save_folder, f'gen{i}.csv')
            prop_path = os.path.join(args.save_folder, f'prop{i}.csv')

            LOG.info(f'Compute properties and metrics: {i}')

            gen = pd.read_csv(gen_path, index_col=[0])
            gen = gen.dropna(subset=['smiles'])
            gen['mol'] = mapper(get_mol, gen['smiles'], args.n_jobs)

            valid = gen.dropna(subset='mol').copy()
            valid['canonical'] =  mapper(get_canonical, valid['mol'], args.n_jobs)
            unique = valid.drop_duplicates(subset='smiles').copy()

            metric['valid'].append(len(valid) / args.n_samples)
            metric['unique'].append(len(unique) / len(valid))
            metric['novel'].append(len(set(unique['smiles']) - set(train['smiles'])) / len(unique))
            metric['intDiv'].append(metrics.internal_diversity(unique['smiles'], args.n_jobs))

            if not os.path.exists(prop_path):
                mol_prop = mols_to_props(valid['mol'], property_fn, n_jobs=args.n_jobs)
                mol_prop.insert(0, 'smiles', valid['smiles'])
                mol_prop.to_csv(prop_path)
            
            mol_prop = pd.read_csv(prop_path, index_col=[0])

            for j, p in enumerate(args.property_list):
                delp = mol_prop[p] - trg_prop[j]
                mse = delp.mean()
                mae = delp.abs().mean()
                sd = delp.std()
                
                metric[f'{p}-MSE'].append(mse)
                metric[f'{p}-MAE'].append(mae)
                metric[f'{p}-SD'].append(sd)    

            good_prop = mol_prop.copy()
            for j, p in enumerate(args.property_list):
                good_prop = good_prop[(good_prop[p] - trg_prop[j]).abs() <= prop_tolerance[p]]
            metric['valid_in_tolerance'].append(len(good_prop) / args.n_samples)
            metric['unique_in_tolerance'].append(len(good_prop.drop_duplicates('smiles')) / args.n_samples)
            metric['n_train'].append(n_train[i])

            current_metric = pd.DataFrame(metric)
            current_metric.to_csv(metric_path)

    # plot property distributions

    xlimit = {
        'logP': [-2,  6],
        'tPSA': [0, 120],
        'QED' : [0.2, 1],
        'SAS' : [1,  10],
    }

    LOG.info('Gather molecular properties')

    gen_prop = []
    for i, prop in enumerate(trg_prop_comb):
        current_prop = pd.read_csv(os.path.join(args.save_folder,
                                   f'prop{i}.csv'), index_col=[0])
        trg_prop = pd.DataFrame(np.tile(np.array(prop), (len(current_prop),1)),
                                columns=[f'trg_{p}' for p in args.property_list])
        current_prop = pd.concat([trg_prop, current_prop], axis=1)
        gen_prop.append(current_prop)
    gen_prop = pd.concat(gen_prop, axis=0)

    LOG.info('Plot property distributions')

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16.5, 4.5))

    for i, p in enumerate(args.property_list):
        ax = axes[i]
        trg_prop = selected_target_prop[p]

        for tp in trg_prop:
            sns.kdeplot(data=gen_prop[gen_prop[f'trg_{p}'] == tp].loc[:, p],
                        ax=ax, shade=True, linewidth=2.5, legend=False)
        sns.kdeplot(data=train.loc[:, p], ax=ax, shade=False,
                    linewidth=2.5, color='red', legend=False)

        ax.set_xlabel(xlabel=p, fontsize=17)
        if i == 0:
            ax.set_ylabel(ylabel='Density', fontsize=17)
        else:
            ax.set_ylabel(None)
        ax.set_xlim(left=xlimit[p][0], right=xlimit[p][1])
        ax.tick_params(axis="both", which="major", labelsize=13)
        ax.legend(['train']+trg_prop, fontsize=16)
        
        for tp in trg_prop:
            ax.axvline(x=tp, linestyle='--', color='gray')        

    fig.savefig(prop_dist_path, bbox_inches="tight")