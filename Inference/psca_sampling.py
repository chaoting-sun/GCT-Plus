import os
import torch
import itertools
import numpy as np
import pandas as pd
from moses.metrics import metrics
from functools import partial
from collections import OrderedDict
from typing import Callable, List, Dict
from pathos.multiprocessing import ProcessingPool as Pool
import matplotlib.pyplot as plt
import seaborn as sns
from Model.build_model import get_generator
from Utils.smiles import (
    get_mol,
    murcko_scaffold,
    murcko_scaffold_similarity,
    plot_smiles_group,
    plot_smiles,
    mol_to_smi,
    plot_highlighted_smiles_group
)
from Utils.properties import mols_to_props, get_property_fn
from Utils.metric import get_error_fn
from Utils.mapper import mapper
from Utils.seed import set_seed
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles


def generate_scaffold(smiles):
    if smiles is not None and isinstance(smiles, str) and len(smiles) > 0:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return MurckoScaffoldSmiles(mol=mol)
    else:
        return None
    

def plot_highlighted_smiles(scaffold_sample, save_folder, property_list,
                            df_train, n_jobs, n=25):
    # test_scaffolds
    sid = 76
    trg_prop = ['3.0', '60.0', '0.85']

    substructure = scaffold_sample.loc[sid, 'scaffold']
    gen = pd.read_csv(os.path.join(save_folder, f's{sid}_p{"-".join(trg_prop)}_prop.csv'),
                      index_col=[0])
    # get smiles with correct substructure
    gen['substructure'] = gen['smiles'].apply(generate_scaffold)
    mols = mapper(get_mol, gen['smiles'], n_jobs)
    gen['canonical'] = mol_to_smi(mols, n_jobs)
    gen = gen.drop_duplicates(subset=['canonical'])
    gen = gen[gen.substructure == substructure]
    # get smiles with low error
    for i, p in enumerate(property_list):
        gen[f'{p}-normalized_AE'] = gen[p].apply(lambda x: abs(x - float(trg_prop[i]))) / (df_train[p].max() - df_train[p].min())
    
    gen = gen.sort_values(by=[f'{p}-normalized_AE' for p in property_list],
                          ignore_index=True)
    # gen = gen.round({ 'logP': 2, 'tPSA': 1, 'QED': 2 })
    smiles_list = gen['smiles'].iloc[:n].tolist()
    gen[property_list].iloc[:n].to_numpy()

    descriptions = []
    for i in range(n):
        p = gen.loc[i, property_list].tolist()
        p = f'logP: {p[0]:.2f}, tPSA: {p[1]:.1f}, QED: {p[2]:.2f}'
        descriptions.append(p)
    print(smiles_list)

    plot_smiles(substructure, os.path.join(save_folder, f's{sid}.png'))
    plot_highlighted_smiles_group(smiles_list, substructure,
                                  img_size=(400, 260),
                                  save_path=os.path.join(save_folder, 
                                            f's{sid}_p{"-".join(trg_prop)}_gen.png'),
                                  n_per_mol=5,
                                  descriptions=descriptions)
    exit()


def get_trg_prop(benchmark, property_list):
    if benchmark == 'guacamol':
        trg_prop = {
            'logP': [2.0, 4.0, 6.0],
            'tPSA': [40.0, 80.0, 120.0],
            'QED' : [0.3, 0.5, 0.7],
            'SAS' : [2.0, 3.0, 4.0],
        }
    elif benchmark == 'moses':
        trg_prop = {
            'logP': [ 1.0,   2.0,  3.0],
            'tPSA': [30.0,  60.0, 90.0],
            'QED' : [ 0.6, 0.725, 0.85],
            'SAS' : [ 2.0,  2.75,  3.5],
        }
    else:
        exit(f'No benchmark named: {benchmark}')

    prop_set = (trg_prop[p] for p in property_list)
    prop_comb = list(itertools.product(*prop_set))
    return [list(c) for c in prop_comb]


def get_molgpt_valid_smi(src, smiles_list, include_mol,
                         sim_bound=0.8, n_jobs=1):
    similarity_fn = partial(murcko_scaffold_similarity, smi_or_mol2=src)
    with Pool(n_jobs) as pool:
        similarity = pool.map(similarity_fn, smiles_list)
    valid_smi = [smiles_list[i] for i, sim in enumerate(similarity)
                 if sim != None and sim >= sim_bound]
    if include_mol:
        with Pool(n_jobs) as pool:
            valid_mol = pool.map(get_mol, valid_smi)
        return valid_smi, valid_mol
    return valid_smi


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


import scipy as sp
from numpy import unravel_index


def plot2d(data, properyty_list, save_path):
    fig = plt.figure(figsize=(5,5))
    sns.kdeplot(data=data, x=properyty_list[0], y=properyty_list[1], hue="kind")
    fig.savefig(save_path)


def plot3d(prop_name, data, figpath, xval, yval, zval,
           figsize=10, s=1, alpha=0.75, cmap='GnBu', # cmap='RdBu_r'
           levels=20, ax_limits=None, ngrids=20j):

    """
    plot a scattering 3D figure with distributions on
    three planes
    ----------
    prop_name: list
        a list of the three variables
    data: pandas.DataFrame
        a DataFrame recording the points
    figpath: string
        the figure path
    figsize: float, 10
        figure size
    s: float, 1
        the size of each point
    alpha: float, 0.75
        the transparency
    cmap: string, 'RdBu_r'
        the color
    levels: int, 20
        the number of the contour lines
    ax_limits: dict, None
        the limits of the axes in three directions
    ngrids: int, 20
        number of the grids each sides
    """

    assert len(prop_name) == 3

    if ax_limits == None:
        ax_limits = {
            prop_name[0]: [np.floor(data[prop_name[0]].min()),
                           np.ceil(data[prop_name[0]].max())],
            prop_name[1]: [np.floor(data[prop_name[1]].min()),
                           np.ceil(data[prop_name[1]].max())],
            prop_name[2]: [np.floor(data[prop_name[2]].min()),
                           np.ceil(data[prop_name[2]].max())]
        }

    xmin = ax_limits[prop_name[0]][0]
    xmax = ax_limits[prop_name[0]][1]
    ymin = ax_limits[prop_name[1]][0]
    ymax = ax_limits[prop_name[1]][1]
    zmin = ax_limits[prop_name[2]][0]
    zmax = ax_limits[prop_name[2]][1]

    x, y, z = np.mgrid[xmin:xmax:ngrids,
                       ymin:ymax:ngrids,
                       zmin:zmax:ngrids]


    # Convert DataFrame to Numpy array
    data = data.to_numpy().T

    # Compute kernel density
    kernel = sp.stats.gaussian_kde(data)
    positions = np.vstack((x.ravel(), y.ravel(), z.ravel()))
    density = np.reshape(kernel(positions).T, x.shape)

    d1, d2, d3 = unravel_index(density.argmax(), density.shape)
    # highest_freq_values = [x[d1,d2,d3], y[d1,d2,d3], z[d1,d2,d3]]

    # plot data
    ax = plt.subplot(projection='3d')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    fig.set_size_inches(15, 15)

    n = 10
    # yz plane
    ax.plot(xs=[xmin]*n, ys=[yval]*n, zs=np.linspace(zmin, zmax, n), c='black')
    ax.plot(xs=[xmin]*n, ys=np.linspace(ymin, ymax, n), zs=[zval]*n, c='black')
    # xy plane
    ax.plot(xs=[xval]*n, ys=np.linspace(ymin, ymax, n), zs=[zmin]*n, c='black')
    ax.plot(xs=np.linspace(xmin, xmax, n), ys=[yval]*n, zs=[zmin]*n, c='black')
    # zx plane
    ax.plot(xs=[xval]*n, ys=[ymax]*n, zs=np.linspace(zmin, zmax, n), c='black')
    ax.plot(xs=np.linspace(xmin, xmax, n), ys=[ymax]*n, zs=[zval]*n, c='black')

    # Set figure size
    # fig = plt.gcf()
    # fig.set_size_inches(figsize, figsize, figsize)

    ax.scatter(data[0, :], data[1, :], data[2, :], s=s, marker='o', c='k')

    ax.set_xlabel(prop_name[0], fontsize=32)
    ax.set_ylabel(prop_name[1], fontsize=32)
    ax.set_zlabel(prop_name[2], fontsize=32)

    ax.xaxis.labelpad = 26
    ax.yaxis.labelpad = 26
    ax.zaxis.labelpad = 26

    ax.tick_params(axis='x', labelsize=24)
    ax.tick_params(axis='y', labelsize=24)
    ax.tick_params(axis='z', labelsize=24)

    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))
    ax.set_zlim((zmin, zmax))
    
    print('plot projection of density onto x-axis')
    plotdat = np.sum(density, axis=0)  # summing up density along z-axis
    plotdat = plotdat / np.max(plotdat)
    ploty, plotz = np.mgrid[ymin:ymax:ngrids, zmin:zmax:ngrids]
    
    colorx = ax.contourf(plotdat, ploty, plotz, levels=levels,
                        alpha=alpha, cmap=cmap, offset=xmin, zdir='x')

    print('plot projection of density onto y-axis')
    plotdat = np.sum(density, axis=1)  # summing up density along y-axis
    plotdat = plotdat / np.max(plotdat)
    plotx, plotz = np.mgrid[xmin:xmax:ngrids, zmin:zmax:ngrids]
    colory = ax.contourf(plotx, plotdat, plotz, levels=levels,
                        alpha=alpha, cmap=cmap, offset=ymax, zdir='y')

    print('plot projection of density onto z-axis')
    plotdat = np.sum(density, axis=2)
    plotdat = plotdat / np.max(plotdat)
    plotx, ploty = np.mgrid[xmin:xmax:ngrids, ymin:ymax:ngrids]
    colorz = ax.contourf(plotx, ploty, plotdat, levels=levels,
                        alpha=alpha, cmap=cmap, offset=zmin, zdir='z')

    cbar = fig.colorbar(colorx, ax=ax, shrink=0.5, pad=0.1)
    cbar.ax.tick_params(labelsize=20)

    plt.tight_layout()
    fig.savefig(figpath)
    # exit()


# def plot3d(X, Y, Z, save_path):

#     # Create figure, add subplot with 3d projection
#     fig = plt.figure(figsize=(5,5))
#     ax = fig.add_subplot(111, projection='3d')
#     ax.set_xlabel("X")
#     ax.set_ylabel("Y")
#     ax.set_zlabel("Z")

#     # Plot the data cloud
#     ax.scatter(X, Y, Z, s=.8, alpha=.5, color='k')

#     hist, binx, biny = np.histogram2d(X, Y)
#     x = np.linspace(X.min(), X.max(), hist.shape[0])
#     y = np.linspace(Y.min(), Y.max(), hist.shape[1])
#     x, y = np.meshgrid(x, y)
#     ax.contour(x, y, hist, zdir='z', offset=-3.)

#     hist, binx, biny = np.histogram2d(X, Z)
#     x = np.linspace(X.min(), X.max(), hist.shape[0])
#     z = np.linspace(Z.min(), Z.max(), hist.shape[1])
#     x, z = np.meshgrid(x, z)
#     ax.contour(x, hist, z, zdir='y', offset=3)

#     hist, binx, biny = np.histogram2d(Y, Z)
#     y = np.linspace(Y.min(), Y.max(), hist.shape[0])
#     z = np.linspace(Z.min(), Z.max(), hist.shape[1])
#     z, y = np.meshgrid(z, y)
#     ax.contour(hist, y, z, zdir='x', offset=-3)

#     # ax.set_xlim([-3, 3])
#     # ax.set_ylim([-3, 3])
#     # ax.set_zlim([-3, 3])
#     ax.set_xlim(X.min(), X.max())
#     ax.set_ylim(Y.min(), Y.max())
#     ax.set_zlim(Z.min(), Z.max())

#     # Show the plot
#     fig.savefig(save_path, bbox_inches="tight")
    
def generate_smiles(sampler, trg_prop, trg_scaffold,
                    batch_size, n):
    samples = []
    dconds = np.repeat([trg_prop], batch_size, axis=0)
    while n > 0:
        print(f'n samples left: {n}')
        smiles, *_ = sampler.sample_smiles(
            dconds[:min(n, batch_size)],
            scaffold=trg_scaffold)
        samples.extend(smiles)
        n -= len(smiles)
    samples = pd.DataFrame(samples, columns=['smiles'])
    return samples


def generate_scaffold(smiles):
    if smiles is not None and isinstance(smiles, str) and len(smiles) > 0:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return MurckoScaffoldSmiles(mol=mol)
    else:
        return None


@torch.no_grad()
def psca_sampling(
        args,
        toklen_data,
        df_train,
        df_test,
        df_test_scaffolds,
        scaler,
        SRC,
        TRG,
        device,
        logger
    ):

    task_path = os.path.join(args.infer_path, args.benchmark, 'psca_sampling')
    os.makedirs(task_path, exist_ok=True)
    LOG = logger(name='psca_sampling', log_path=os.path.join(task_path, 'record.log'))

    # n_batch = int(np.ceil(n_samples / batch_size))
    train_set = set(df_train['smiles'])
    
    # prop_tolerance = { 'logP': 0.2, 'tPSA': 4, 'QED' : 0.02 } # 2%
    prop_tolerance = { 'logP': 0.4, 'tPSA': 8, 'QED' : 0.03, 'SAS': 0.25 } # 4%
    
    # interested_properties = ['logP', 'tPSA', 'QED',
    #                          'SAS',   'NP', 'MW' ,
    #                         'HAC',  'HBA', 'HBD']

    # error_fn = get_error_fn(['MSE', 'MAE'])
    property_fn = get_property_fn(args.property_list)


    if args.sample_from == 'molgpt':
        trg_prop_settings = {
            'logP': [ 1.0, 3.0],
            'tPSA': [  40,  80],
            'SAS' : [ 2.0, 3.5],
        }
    else:
        trg_prop_settings = {
            'logP': [ 1.0,   2.0,  3.0],
            'tPSA': [30.0,  60.0, 90.0],
            'QED' : [ 0.6, 0.725, 0.85],
        }


    prop_set = (trg_prop_settings[p] for p in args.property_list)
    prop_comb = list(itertools.product(*prop_set))
    trg_prop_list = [list(c) for c in prop_comb]
    # trg_prop_list = get_trg_prop(args.benchmark, args.property_list)

    LOG.info('get scaffold')

    if args.sample_from == 'train':
        scaffold_sample = get_sample(df_train, df_train, task_path,
                                     'train', n=args.n_scaffolds)
    elif args.sample_from == 'test_scaffolds':
        scaffold_sample = get_sample(df_test_scaffolds, df_train, task_path,
                                     'test_scaffolds', n=args.n_scaffolds)
    elif args.sample_from == 'molgpt':
        molgpt_scaffold = [ 'O=C(Cc1ccccc1)NCc1ccccc1',
                            'c1cnc2[nH]ccc2c1',
                            'c1ccc(-c2ccnnc2)cc1',
                            'c1ccc(-n2cnc3ccccc32)cc1',
                            'O=C(c1cc[nH]c1)N1CCN(c2ccccc2)CC1'
                          ]
        scaffold_sample = pd.DataFrame({ 'scaffold': molgpt_scaffold })
        scaffold_sample.to_csv(os.path.join(task_path, 'molgpt_scaffold.csv'))

    save_folder = os.path.join(task_path, f'{args.model_name}-{args.epoch}',
                               args.sample_from)
    os.makedirs(save_folder, exist_ok=True)

    # plot_highlighted_smiles(scaffold_sample, save_folder, args.property_list,
    #                         df_train, args.n_jobs)
    
    LOG.info('get sampler')

    args.model_path = os.path.join(args.train_path, args.benchmark,
                                   args.model_name, f'model_{args.epoch}.pt')
    sampler = get_generator(args, SRC, TRG, toklen_data, scaler, device)
    # sampler = None
    
    LOG.info('start generation')

    for sid in range(len(scaffold_sample)):
        LOG.info(f'sid: {sid}')
        
        metric = OrderedDict()
        n_good = OrderedDict() # number of SMILES that meet specified conditions

        metric_path = os.path.join(save_folder, f's{sid}_metric.csv')
        good_path = os.path.join(save_folder, f's{sid}_good.csv')

        if os.path.exists(metric_path):
            continue
        
        # if os.path.exists(metric_path):
        #     _metric = pd.read_csv(metric_path, index_col=[0])
        #     metric = _metric.to_dict(orient='list')
        # else:
        
        for p in args.property_list:
            metric[p] = []
        for met in ('scaffold', 'valid', 'unique', 'novel', 'intDiv', 'sim', 'SSF', 'sim80'):
            metric[met] = []
        for p in args.property_list:
            metric[f'{p}-MSE'] = []
            metric[f'{p}-MAE'] = []
            metric[f'{p}-SD'] = []

        n_good['n_good_scaffold'] = [] # same scaffold
        n_good['n_good_property'] = [] # falling in 4% range
        n_good['n_good_both'] = [] # meeting both conditions
        
        trg_scaffold = scaffold_sample.loc[sid, 'scaffold']

        LOG.info(f'sample smiles from scaffold: {trg_scaffold}')
        
        for pid, trg_prop in enumerate(trg_prop_list):
            # if pid < len(metric['valid']):
            #     continue

            LOG.info('sample smiles from property set: %s', trg_prop)
            
            suffix = '-'.join(map(str, trg_prop))
            gen_path = os.path.join(save_folder, f's{sid}_p{suffix}_gen.csv')
            property_path = os.path.join(save_folder, f's{sid}_p{suffix}_prop.csv')

            for i, p in enumerate(args.property_list):
                metric[p].append(trg_prop[i])
            metric['scaffold'].append(trg_scaffold)
            
            if not os.path.exists(gen_path):
                samples = generate_smiles(sampler, trg_prop, trg_scaffold,
                                          args.batch_size, args.n_samples)
                samples.to_csv(gen_path)
            
            samples = pd.read_csv(gen_path, index_col=[0])
            samples = samples.dropna(subset=['smiles'])
            mols = mapper(get_mol, samples['smiles'], args.n_jobs)
            mols = [m for m in mols if m is not None and isinstance(m, float) is False]
            scaffolds = mapper(murcko_scaffold, mols, args.n_jobs)            
            valid_smi = mol_to_smi(mols, args.n_jobs)

            # the definition of validity is different in molgpt from us
            # molgpt: the fraction of SMILES that "satisfy chemical valencies
            # and contain scaffolds that have a Tanimoto similarity of at least
            # 0.8 to the desired scaffold.

            if args.sample_from == 'molgpt':
                similarity_fn = partial(murcko_scaffold_similarity, smi_or_mol2=trg_scaffold)
                similarity = mapper(similarity_fn, scaffolds, args.n_jobs)
                _valid_smi, _mols, _scaffolds = [], [], []
                for i, sim in enumerate(similarity):                
                    if sim is not None and sim >= 0.8:
                        _valid_smi.append(valid_smi[i])
                        _mols.append(mols[i])
                        _scaffolds.append(scaffolds[i])

                valid_smi = _valid_smi
                mols = _mols
                scaffolds = _scaffolds

            unique_smi = set(valid_smi)

            LOG.info('evaluate: compute properties')

            if not os.path.exists(property_path):
                props = mols_to_props(mols, property_fn, n_jobs=args.n_jobs)
                props.insert(0, 'smiles', valid_smi)
                props['scaffold'] = scaffolds
                props.to_csv(property_path)

            props = pd.read_csv(property_path, index_col=[0])

            # good scaffold
            props1 = props[props.scaffold == trg_scaffold]
            n_good['n_good_scaffold'].append(len(props1))

            # good property
            prop2 = props.copy()
            for k, p in enumerate(args.property_list):
                prop2 = prop2[(prop2[p] - trg_prop[k]).abs() <= prop_tolerance[p]]
            n_good['n_good_property'].append(len(prop2))

            # good scaffold and property
            prop2 = prop2[prop2.scaffold == trg_scaffold]
            n_good['n_good_both'].append(len(prop2))
            
            _n_good = pd.DataFrame(n_good)
            _n_good.to_csv(good_path)

            LOG.info('evaluate: compute metrics')

            similarity_fn = partial(murcko_scaffold_similarity, smi_or_mol2=trg_scaffold)
            scaffold_similarity = mapper(similarity_fn, valid_smi, args.n_jobs)
            scaffold_similarity = [s for s in scaffold_similarity if s is not None]

            metric['valid'].append(len(valid_smi) / len(samples))

            if len(valid_smi) > 0:
                unique = len(unique_smi) / len(valid_smi)
                novel = len(unique_smi - train_set) / len(unique_smi)
                intDiv = metrics.internal_diversity(valid_smi, args.n_jobs)
                
                if len(scaffold_similarity) > 0:
                    sim = sum(scaffold_similarity) / len(scaffold_similarity)
                    ssf = len([1 for s in scaffold_similarity
                        if s == 1]) / len(scaffold_similarity)
                    sim80 = len([1 for s in scaffold_similarity
                            if s >= 0.80]) / len(scaffold_similarity)
                else:
                    sim = ssf = sim80 = np.nan
            else:
                unique = novel = intDiv = sim = ssf = sim80 = np.nan
                
                
            

            metric['unique'].append(unique)
            metric['novel'].append(novel)
            metric['intDiv'].append(intDiv)
            metric['sim'].append(sim)
            metric['SSF'].append(ssf)
            metric['sim80'].append(sim80)

            LOG.info('evaluate: compute errors')

            for k, p in enumerate(args.property_list):
                if len(valid_smi) > 0:
                    delp = props[p] - trg_prop[k]
                    mse = delp.mean()
                    mae = delp.abs().mean()
                    sd = delp.std()
                else:
                    mse = mae = sd = np.nan

                metric[f'{p}-MSE'].append(mse)
                metric[f'{p}-MAE'].append(mae)
                metric[f'{p}-SD'].append(sd)
            
            print(metric)
            
            _metrics = pd.DataFrame(metric)
            _metrics.to_csv(metric_path)

    for sid in range(args.n_scaffolds):
        df = pd.read_csv(os.path.join(save_folder, f's{sid}_good.csv'), index_col=[0])        
        if sid == 0:
            df_add = df.copy()    
        else:
            df_add = df_add.add(df, fill_value=0)
    df_add = df_add.apply(lambda x: x / args.n_scaffolds)
    df_add.to_csv(os.path.join(save_folder, 'avg_good.csv'))

    # # save metrics

    avg_prop_metric_path = os.path.join(save_folder, 'avg_prop_metric.csv')

    # if not os.path.exists(avg_prop_metric_path):
    all_met = []
    for sid in range(args.n_scaffolds):
        all_met.append(pd.read_csv(os.path.join(save_folder,
                        f's{sid}_metric.csv'), index_col=[0]))
    all_met = pd.concat(all_met, axis=0)
    all_met = all_met.select_dtypes(include=['float', 'int'])

    avg_met = all_met.groupby(args.property_list).mean()
    avg_met = avg_met.reset_index()
    avg_met.to_csv(avg_prop_metric_path, index=False)
    
    # plot property distribution

    print('plot property distribution')

    for pid, trg_prop in enumerate(trg_prop_list):
        for sid in range(args.n_scaffolds):
            suffix = '-'.join(map(str, trg_prop))
            _current_prop = pd.read_csv(os.path.join(save_folder, 
                                        f's{sid}_p{suffix}_prop.csv'))
            if sid == 0:
                current_prop = _current_prop.copy()
            else:
                current_prop = pd.concat([current_prop, _current_prop], axis=0)
        
        current_prop = current_prop.reset_index(drop=True)
        _trg_prop = pd.DataFrame(np.tile(np.array(trg_prop),
                                         (len(current_prop),1)),
                                 columns=[f'trg_{p}' for p in args.property_list])
        current_prop = pd.concat([_trg_prop, current_prop], axis=1)
        current_prop['kind'] = [pid]*len(current_prop)

        if pid == 0:
            cumm_prop = current_prop.copy()
        else:
            cumm_prop = pd.concat([cumm_prop, current_prop], axis=0)

    cumm_prop = cumm_prop.reset_index(drop=True)

    print(cumm_prop)

    train_prop_path = os.path.join(task_path, 'train_prop.csv')
    if not os.path.exists(train_prop_path):
        sampled_train = df_train['smiles'].sample(n=10000, ignore_index=True)
        mols = mapper(get_mol, sampled_train, n_jobs=args.n_jobs)
        train_prop = mols_to_props(mols, property_fn)
        train_prop = pd.concat([df_train['smiles'], train_prop], axis=1)
        train_prop.to_csv(train_prop_path)
    train_prop = pd.read_csv(train_prop_path, index_col=[0])

    plot2d(cumm_prop, args.property_list, os.path.join(save_folder, '2d.png'))

    exit()

    for pid, trg_prop in enumerate(trg_prop_list):
        save_path = os.path.join(save_folder, '-'.join(map(str, trg_prop))+'.png')
        
        val = cumm_prop[(cumm_prop[f'trg_{args.property_list[0]}'] == trg_prop[0])
                      & (cumm_prop[f'trg_{args.property_list[1]}'] == trg_prop[1])
                      & (cumm_prop[f'trg_{args.property_list[2]}'] == trg_prop[2])]

        val = val[args.property_list].astype(float)
        plot3d(args.property_list, val, save_path,
               trg_prop[0], trg_prop[1], trg_prop[2])
    exit()




    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16.5, 4.5))

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
        # ax.set_xlim(left=xlimit[p][0], right=xlimit[p][1])

    fig.savefig(os.path.join(save_folder, f'prop_dist.png'), bbox_inches="tight")



