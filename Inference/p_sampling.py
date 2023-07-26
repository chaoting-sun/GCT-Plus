import os
import numpy as np
import pandas as pd
import random
import itertools
from Model.build_model import get_generator
from Utils.properties import mols_to_props, get_property_fn
from Utils.smiles import get_mol, mol_to_smi, plot_smiles_group
from Utils.mapper import mapper
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict
from Utils.metric import get_metric_fn
from moses.metrics import metrics


from torch import load

trg_prop_settings = {
    'logP': [ 1.0,   2.0,  3.0],
    'tPSA': [30.0,  60.0, 90.0],
    'QED' : [ 0.6, 0.725, 0.85],
    'SAS' : [ 2.0,  2.75,  3.5],
}


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


def get_num(train, trg_prop_list, property_list):
    half_distance = []
    for p in property_list:
        half_distance.append((trg_prop_settings[p][1] - trg_prop_settings[p][0])/2)

    for i, trg_prop in enumerate(trg_prop_list):
        _train = train.copy()
        for j, p in enumerate(property_list):
            _train = _train[(trg_prop[j]-half_distance[j] <= _train[p]) & 
                            (_train[p] <= trg_prop[j]+half_distance[j])]
        print(len(_train))


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def plot_property_conditions(arr, property_list, save_path):
    colors = ['r', 'b', 'g'] * 9

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')    
    ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2], marker='o', s=40, c=colors)

    ax.set_xlabel(property_list[0], fontsize=14, labelpad=7)
    ax.set_ylabel(property_list[1], fontsize=14, labelpad=7)
    ax.set_zlabel(property_list[2], fontsize=14, labelpad=7)
    
    ax.tick_params(axis='x', which='major', labelsize=12, width=4)
    ax.tick_params(axis='y', which='major', labelsize=12, width=4)
    ax.tick_params(axis='z', which='major', labelsize=12, width=4)

    plt.savefig(save_path)


from scipy.stats import kde

def plot_3d_figure(x, y, z, save_path):
    # Create a 3D scatter plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='b', marker='o', alpha=0.6)

    # Compute the KDEs
    kde_xy = kde.gaussian_kde(np.vstack([x, y]))
    kde_yz = kde.gaussian_kde(np.vstack([y, z]))
    kde_xz = kde.gaussian_kde(np.vstack([x, z]))

    # Create a grid for the KDE projections
    grid_size = 100
    xy_space = np.linspace(-3, 3, grid_size)
    yz_space = np.linspace(-3, 3, grid_size)
    xz_space = np.linspace(-3, 3, grid_size)
    xy_grid, yz_grid = np.meshgrid(xy_space, yz_space)
    xz_grid, _ = np.meshgrid(xz_space, xy_space)

    # Compute the KDE values at the grid points
    xy_kde_vals = kde_xy(np.vstack([xy_grid.ravel(), yz_grid.ravel()])).reshape(grid_size, grid_size)
    yz_kde_vals = kde_yz(np.vstack([yz_grid.ravel(), xz_grid.ravel()])).reshape(grid_size, grid_size)
    xz_kde_vals = kde_xz(np.vstack([xy_grid.ravel(), xz_grid.ravel()])).reshape(grid_size, grid_size)

    # Plot the KDE projections on the respective planes
    ax.contourf(xy_grid, yz_grid, xy_kde_vals, zdir='z', offset=np.min(z), alpha=0.6, cmap='viridis')
    ax.contourf(yz_grid, xz_grid, yz_kde_vals, zdir='x', offset=np.min(x), alpha=0.6, cmap='viridis')
    ax.contourf(xy_grid, xz_grid, xz_kde_vals, zdir='y', offset=np.min(y), alpha=0.6, cmap='viridis')

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.savefig(save_path)


def plot_good_smiles(model_name, epoch, save_folder, trg_prop_list,
                     property_list, df_train, n=25):
    trg_prop = [3.0, 60.0, 0.725]
    
    prop_no = -1
    for i, p in enumerate(trg_prop_list):
        if p == trg_prop:
            prop_no = i
    if prop_no == -1:
        exit('find nothing')

    gen = pd.read_csv(os.path.join(save_folder, f'{model_name}-{epoch}_prop_{prop_no}.csv'),
                      index_col=[0])
    for i, p in enumerate(property_list):
        gen[f'{p}-normalized_AE'] = gen[p].apply(lambda x: abs(x - float(trg_prop[i]))) / (df_train[p].max() - df_train[p].min())
    gen = gen.sort_values(by=[f'{p}-normalized_AE' for p in property_list],
                          ignore_index=True)
    smiles_list = gen['SMILES'].iloc[:n].tolist()
    gen[property_list].iloc[:n].to_numpy()

    descriptions = []
    for i in range(n):
        p = gen.loc[i, property_list].tolist()
        p = f'logP: {p[0]:.2f}, tPSA: {p[1]:.1f}, QED: {p[2]:.2f}'
        descriptions.append(p)
    print(smiles_list)

    save_path = os.path.join(save_folder, f'good_smiles_{prop_no}.png')
    plot_smiles_group(smiles_list, save_path, n_per_mol=5, descriptions=descriptions,
                      img_size=(310, 220), n_jobs=1)
    exit()


def p_sampling(
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
    task_path = os.path.join(args.infer_path,
                             args.benchmark,
                             'prop_sampling')
    os.makedirs(task_path, exist_ok=True)

    # property conditions

    prop_set = (trg_prop_settings[p] for p in args.property_list)
    prop_comb = list(itertools.product(*prop_set))
    trg_prop_list = [list(c) for c in prop_comb]

    # 3d plot of property conditions

    conditions = np.array(trg_prop_list)
    plot_property_conditions(conditions, args.property_list,
                             os.path.join(task_path, 'condition.png'))

    # settings
    
    n_samples = 10000
    prop_tolerance = { 'logP': 0.4, 'tPSA': 8, 'QED' : 0.03 }
    # prop_tolerance = { 'logP': 0.2, 'tPSA': 4, 'QED' : 0.02 }

    # trg_prop_settings = {
    #     'logP': [ 1.0,   2.0,  3.0],
    #     'tPSA': [30.0,  60.0, 90.0],
    #     'QED' : [ 0.3,   0.5,  0.7],
    #     'SAS' : [ 2.0,   3.0,  4.0],
    # }

    # get_num(df_train, trg_prop_list, args.property_list)

    interested_properties = ['logP', 'tPSA', 'QED',
                              'SAS',   'NP',  'MW',
                              'HAC',  'HBA', 'HBD']
    property_fn = get_property_fn(interested_properties)

    df_trg_prop = pd.DataFrame(trg_prop_list, columns=args.property_list)
    df_trg_prop.to_csv(os.path.join(task_path, f'property_{"-".join(args.property_list)}.csv'))

    # create file path and folder

    save_folder = os.path.join(task_path, f'{args.model_name}_{args.epoch}')
    os.makedirs(save_folder, exist_ok=True)

    # plot_good_smiles(args.model_name, args.epoch, save_folder, trg_prop_list,
    #                  args.property_list, df_train)

    LOG = logger(name='scaffold sampling', log_path=os.path.join(save_folder, 'record.log'))

    LOG.info('create a generator...')
    
    if args.use_molgct:
        args.model_path = '/fileserver-gamma/chaoting/ML/molGCT/molgct.pt'
    else:
        args.model_path = os.path.join(args.train_path,
                                       args.benchmark,
                                       args.model_name,
                                       f'model_{args.epoch}.pt')
    generator = get_generator(args, SRC, TRG, toklen_data,
                              scaler, device)
    # generator = None

    LOG.info('create a save folder')

    if args.use_molgct:
        prefix = 'molgct'
    else:
        prefix = f'{args.model_name}-{args.epoch}'
    save_folder = os.path.join(task_path, prefix)
    os.makedirs(save_folder, exist_ok=True)

    train_prop_path = os.path.join(task_path, 'train_prop.csv')
    if not os.path.exists(train_prop_path):
        sampled_train = df_train['smiles'].sample(n=10000, ignore_index=True)
        mols = mapper(get_mol, sampled_train, n_jobs=args.n_jobs)
        train_prop = mols_to_props(mols, property_fn)
        train_prop = pd.concat([df_train['smiles'], train_prop], axis=1)
        train_prop.to_csv(train_prop_path)

    property_path = os.path.join(save_folder, 'property.csv')
    
    # print('Files to save:', gen_path, property_path, metric_path)

    LOG.info('sample molecules')

    error_path = os.path.join(save_folder, f'{prefix}_error.csv')
    errors = OrderedDict()
    for p in args.property_list:
        errors[f'{p}-MSE'] = []
        errors[f'{p}-MAE'] = []
        errors[f'{p}-SD'] = []
    errors['valid_in_tolerance'] = []
    errors['unique_in_tolerance'] = []

    for i, trg_prop in enumerate(trg_prop_list):
        LOG.info('properties:%s', trg_prop)
        
        gen_path = os.path.join(save_folder, f'{prefix}_gen_{i}.csv')
        property_path = os.path.join(save_folder, f'{prefix}_prop_{i}.csv')
        
        LOG.info('generate: sample smiles')
        
        print(gen_path)

        if not os.path.exists(gen_path):
            gen = sample_smiles(generator, trg_prop, n_samples)
            gen = pd.DataFrame(gen, columns=['SMILES'])
            gen.to_csv(gen_path)

        LOG.info('evaluate: compute properties')

        if not os.path.exists(property_path):
            gen = pd.read_csv(gen_path, index_col=[0])
            gen = gen.dropna(subset=['SMILES'])

            mols = mapper(get_mol, gen['SMILES'], args.n_jobs)
            mols = [m for m in mols if m is not None]
            smiles = pd.DataFrame(mol_to_smi(mols, args.n_jobs), columns=['SMILES'])
            props = mols_to_props(mols, property_fn, n_jobs=args.n_jobs)
            props = pd.concat([smiles, props], axis=1)
            props.to_csv(property_path)
    
        LOG.info('compute errors')

        props = pd.read_csv(property_path, index_col=[0])

        for j, p in enumerate(args.property_list):
            delp = props[p] - trg_prop[j]
            mse = delp.mean()
            mae = delp.abs().mean()
            sd = delp.std()
            
            errors[f'{p}-MSE'].append(mse)
            errors[f'{p}-MAE'].append(mae)
            errors[f'{p}-SD'].append(sd)
        
        good_gen = props.copy()
        for k, p in enumerate(args.property_list):
            good_gen = good_gen[(good_gen[p] - trg_prop[k]).abs() <= prop_tolerance[p]]
        errors['valid_in_tolerance'].append(len(good_gen))
        errors['unique_in_tolerance'].append(len(good_gen.drop_duplicates('SMILES')))

        df_errors = pd.DataFrame(errors)
        df_errors.to_csv(error_path)
        print(df_errors)

    LOG.info('gather all properties')

    ### average the three models

    # model_dict = { 'cvaetf1': 8,
    #                'cvaetf2': 10,
    #                'cvaetf3': 8,
    #              }

    metric = {
        'valid'  : [],
        'unique' : [],
        'novel'  : [],
        'intDiv' : [],
        'valid_in_tolerance': [],
        'unique_in_tolerance': []
    }

    if args.use_molgct:
        metric_path = os.path.join(task_path, 'metric-molgct.csv')
    else:
        model_dict = { 'cvaetf1': 15,
                       'cvaetf2': 15,
                       'cvaetf3': 15,
                    }
        metric_path = os.path.join(task_path, f'metric-{"_".join(map(str, model_dict.values()))}.csv')

    train_set = set(df_train['smiles'])

    if not os.path.exists(metric_path):
        for i, trg_prop in enumerate(trg_prop_list):
            print(f'({i}) metric:', trg_prop)
            smiles = []
            
            if args.use_molgct:
                _current_smi = pd.read_csv(os.path.join(task_path, 'molgct',
                                                        f'molgct_gen_{i}.csv'),
                                                        index_col=[0]
                                        ).dropna(subset=['SMILES'])                
            else:
                for j, (model_name, epoch) in enumerate(model_dict.items()):
                    _current_smi = pd.read_csv(os.path.join(task_path,
                                            f'{model_name}_{epoch}',
                                            f'{model_name}-{epoch}_gen_{i}.csv'),
                                            index_col=[0]
                                            ).dropna(subset=['SMILES'])

            smiles.extend(_current_smi['SMILES'].tolist())
            
            mols = mapper(get_mol, smiles, args.n_jobs)
            mols = [m for m in mols if m is not None]
            valid_smi = mol_to_smi(mols, args.n_jobs) # canonicalized smiles
            unique_smi = set(valid_smi)

            print('compute valid...')
            metric['valid'].append(len(valid_smi) / len(smiles))
            print('compute unique...')
            metric['unique'].append(len(unique_smi) / len(valid_smi))
            print('compute novel...')
            metric['novel'].append(len(unique_smi - train_set) / len(unique_smi))
            print('compute intDiv...')
            metric['intDiv'].append(metrics.internal_diversity(list(unique_smi), args.n_jobs))

            _metric = pd.DataFrame(metric)
            _metric.to_csv(metric_path)
    
    # plot property distributions

    for i, trg_prop in enumerate(trg_prop_list):
        for j, (model_name, epoch) in enumerate(model_dict.items()):
            _current_prop = pd.read_csv(os.path.join(task_path,
                                        f'{model_name}_{epoch}',
                                        f'{model_name}-{epoch}_prop_{i}.csv'),
                                        index_col=[0]
                                        )
            if j == 0:
                current_prop = _current_prop.copy()
            else:
                current_prop = pd.concat([current_prop, _current_prop], axis=0)
        
        current_prop = current_prop.reset_index(drop=True)        
        _trg_prop = pd.DataFrame(np.tile(np.array(trg_prop),
                                         (len(current_prop),1)),
                                 columns=[f'trg_{p}' for p in args.property_list])
        current_prop = pd.concat([_trg_prop, current_prop], axis=1)

        if i == 0:
            cumm_prop = current_prop.copy()
        else:
            cumm_prop = pd.concat([cumm_prop, current_prop], axis=0)

    train_prop = pd.read_csv(train_prop_path, index_col=[0])

    LOG.info('plot distributions of properties')

    xlimit = {
        'logP': [-2, 6],
        'tPSA': [0, 120],
        'QED' : [0.2, 1],
        'MW'  : [0, 600],
        'SAS' : [1, 10],
        'NP'  : [-5, 5],
    }
    
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16.5, 4.5))

    # for i, p in enumerate(args.property_list):
    for i, p in enumerate(args.property_list):
        # rowi = i // 3
        # coli = i % 3
        # ax = axes[rowi, coli]

        ax = axes[i]

        prop_values = trg_prop_settings[p]

        print(prop_values)

        sns.kdeplot(data=train_prop.loc[:, p], ax=ax,
                    shade=False, linewidth=2.5,
                    color='red', legend=False)

        for v in prop_values:
            sns.kdeplot(data=cumm_prop[cumm_prop[f'trg_{p}'] == v].loc[:, p], ax=ax,
                        shade=True, linewidth=2.5, legend=False)
            ax.set_xlabel(xlabel=p, fontsize=17)
            if i == 0:
                ax.set_ylabel(ylabel='Density', fontsize=17)
            else:
                ax.set_ylabel(None)
            ax.tick_params(axis="both", which="major", labelsize=13)
        ax.legend(['train']+prop_values, fontsize=16)
        # ax.legend(fontsize=16)
        for v in prop_values:
            ax.axvline(x=v, linestyle='--', color='gray')
        ax.set_xlim(left=xlimit[p][0], right=xlimit[p][1])

    fig.savefig(os.path.join(task_path,
                             f'prop_dist-{"_".join(map(str, model_dict.values()))}.png'),
                bbox_inches="tight")

    # plot uncontrolled properties

    interested_properties = ['SAS',   'NP',  'MW',
                             'HAC',  'HBA', 'HBD']
    property_fn = get_property_fn(interested_properties)

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16.5, 9.5))
    
    for i, p in enumerate(interested_properties):
        rowi = i // 3
        coli = i % 3
        ax = axes[rowi, coli]

        sns.kdeplot(data=train_prop.loc[:, p], ax=ax,
                    shade=True, linewidth=2,
                    color='red', legend=False, label='train')
        
        for j, trg_prop in enumerate(trg_prop_list):
            print('j =', j, trg_prop)

            _cumm_prop = cumm_prop.copy()

            for k, pname in enumerate(args.property_list):
                _cumm_prop = _cumm_prop[_cumm_prop[f'trg_{pname}'] == trg_prop[k]]
            _cumm_prop = _cumm_prop.reset_index(drop=True)
            sns.kdeplot(data=_cumm_prop.loc[:, p], ax=ax, shade=False,
                                 linewidth=0.8, legend=False, color='black', alpha=0.6, label='gen')
             

        ax.set_xlabel(xlabel=p, fontsize=17)
        if coli == 0:
            ax.set_ylabel(ylabel='Density', fontsize=17)
        else:
            ax.set_ylabel(None)
        ax.tick_params(axis="both", which="major", labelsize=13)

        handles, labels = plt.gca().get_legend_handles_labels()

        print(handles)
        print(labels)
        exit()
        new_handles = [handles[0], handles[1]]
        new_labels = [labels[0], labels[2]]
        plt.legend(new_handles, new_labels, loc='upper left')
        
        # ax.legend(fontsize=16)
        # ax.set_xlim(left=xlimit[p][0], right=xlimit[p][1])

    fig.savefig(os.path.join(task_path, f'other_prop_dist-{"_".join(map(str, model_dict.values()))}.png'),
                bbox_inches="tight")
    
    
# def plot_property_distribution():
    
    
# def plot_number_distribution():