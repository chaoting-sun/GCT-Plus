import os
import torch
import random
import numpy as np
import pandas as pd
import seaborn as sns
from rdkit import Chem
from functools import partial
import matplotlib.pyplot as plt
from moses.metrics import metrics
from collections import OrderedDict

from Model.build_model import get_sampler
from Utils import mapper, get_mol, set_seed, get_canonical,     \
    is_substructure, tanimoto_similarity, murcko_scaffold,      \
    mols_to_props, get_property_fn, murcko_scaffold_similarity, \
    plot_smiles, plot_smiles_group, plot_highlighted_smiles_group


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


def sample_smiles(sampler, n, scaffold, batch_size, LOG):
    gen = []
    while n > 0:
        LOG.info(f'# Samples left: {n}')
        current_gen, *_ = sampler.sample_smiles(n=min(n, batch_size),
                                                scaffold=scaffold)
        gen.extend(current_gen)
        n -= len(current_gen)
    return gen


# def generate_scaffold(smiles):
#     if smiles is not None and isinstance(smiles, str) and len(smiles) > 0:
#         mol = Chem.MolFromSmiles(smiles)
#         if mol is None:
#             return None
#         return MurckoScaffoldSmiles(mol=mol)
#     else:
#         return None
    

# def plot_highlighted_smiles(scaffold_sample, save_folder, n_jobs=1, gen_id=85, n=36):
#     # row : col = 4 : 5 = 300 : 140
    
#     substructure = scaffold_sample.loc[gen_id, 'scaffold']
#     gen = pd.read_csv(os.path.join(save_folder, f's{gen_id}_gen.csv'),
#                       index_col=[0])
#     gen['substructure'] = gen['smiles'].apply(generate_scaffold)
#     mols = mapper(get_mol, gen['smiles'], n_jobs)
#     gen['canonical'] = mapper(get_canonical, mols, n_jobs)
#     gen = gen.drop_duplicates(subset=['canonical'])
#     gen = gen[gen.substructure == substructure]
#     smiles_list = gen.sample(n=n)['canonical'].tolist()
#     print(smiles_list)

#     plot_smiles(substructure, os.path.join(save_folder, f's{gen_id}.png'))
#     print(os.path.join(save_folder, f's{gen_id}.png'))
#     print(os.path.join(save_folder, f's{gen_id}_gen.png'))
#     plot_highlighted_smiles_group(smiles_list, substructure, 
#                                   img_size=(250, 140),
#                                   save_path=os.path.join(save_folder, f's{gen_id}_gen.png'),
#                                   n_per_mol=6)
#     exit()


# def substructure_sampling(args, sampler):
#     save_folder = '/fileserver-gamma/chaoting/ML/cvae-transformer/Inference-Dataset/moses/sca_sampling/non-scaffold'

#     # substructure = 'CC1C2CCC(C2)C1CN(CCO)C(=O)c1ccc(Cl)cc1'
#     # substr_list = [
#     #     'Oc1c(Br)cc(Br)c2cccnc12',
#     #     'CC(=O)Nc1ccc(OC(=O)c2ccccc2OC(C)=O)cc1',
#     #     'OCCN(CCO)c1nc(-c2ccccc2)c(-c2ccccc2)o1'
#     # ]
    
#     substr_list = ['CCCCCC', 'CCCCCCCCC', 'CCCC=CCCC=C', 'CC(CC(=C)C)C=C(C)C', 'CCCC(CCC)CC=C']
#     plot_smiles_group(smiles=substr_list, save_path=os.path.join(save_folder, 'carbonchain.png'), n_per_mol=5)
    
#     # substr_list = ['COc1ccc(-c2cc(C(=O)Nc3cccc(O)c3)no2)cc1',
#     #                'CSc1ccc(-c2csc3nnc(C#N)n23)cc1',
#     #                'Cc1occc1C(=O)NCC(=O)N1CCCC(c2ccccc2)CC1',
#     #                'O=C(NC1CCOC1=O)c1ccccc1Br',
#     #                'CC(O)c1nc2ccccc2n1CC(=O)N(C)Cc1cccs1']
#     # plot_smiles_group(smiles=substr_list, save_path=os.path.join(save_folder, 'molecule.png'), n_per_mol=5)
    
#     for i, substr in enumerate(substr_list):
#         save_path = os.path.join(save_folder, substr)
#         os.makedirs(save_path, exist_ok=True)
        
#         smiles, *_ = sampler.sample_smiles(n=1000, scaffold=substr)


#         mols = mapper(get_mol, smiles, args.n_jobs)
#         mols = [m for m in mols if m is not None and isinstance(m, float) is False]        
#         valid_smi = mapper(get_canonical, mols, args.n_jobs)
        
#         is_subst = partial(is_substructure, subst=substr)
#         res = mapper(is_subst, valid_smi, n_jobs=args.n_jobs)
#         valid_smi = [smi for i, smi in enumerate(valid_smi) if res[i] is True]
#         valid_smi = list(set(valid_smi))

#         print(f'# unique SMILES: {len(valid_smi)}')
#         print('# matched with substructures:', sum(res))
        
#         if len(valid_smi):
#             plot_highlighted_smiles_group(valid_smi[:6],
#                                         substructure_smiles=substr,
#                                         save_path=os.path.join(save_path, 'gen_group.png'),
#                                         img_size=(270, 200),
#                                         n_per_mol=6
#                                         )

#         # plot_smiles_group(smiles[:15], save_path=f'./{i}-gen.png', n_per_mol=4)
#         plot_smiles(substr, save_path=os.path.join(save_path, 'substr.png'))

        
#         sim = [tanimoto_similarity(substr, str(smi)) for smi in valid_smi]
#         sim = [s for s in sim if s != None]
#         print(sim)
        
#         # print('average similarity:', sum(sim) / len(sim))
    

@torch.no_grad()
def sca_sampling(
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

    LOG = logger(name='sca_sampling', log_path=os.path.join(args.save_folder, 'record.log'))

    if args.use_molgpt:
        save_folder = os.path.join(args.save_folder, f'{args.scaffold_source}-molgpt')
    else:
        save_folder = os.path.join(args.save_folder, f'{args.scaffold_source}')
    metric_path = os.path.join(save_folder, 'metric.csv')
    scaffold_sim_path = os.path.join(save_folder, 'scaffold_sim.csv')
    scaffold_sim_png_path = os.path.join(save_folder, 'scaffold_sim.png')

    os.makedirs(save_folder, exist_ok=True)

    # get sampler

    args.model_path = os.path.join(args.model_folder, args.model_name)
    sampler = get_sampler(args, SRC, TRG, toklen_data, scaler, device)

    # get scaffold

    if args.scaffold_source == 'train':
        scaffold_path = os.path.join(args.scaffold_folder, 'train_sample.csv')
        scaffold_sample = get_sample(args.n_scaffolds, scaffold_path,
                                     train, dataset=train)
        
    elif args.scaffold_source == 'test_scaffolds' and not args.use_molgpt:
        scaffold_path = os.path.join(args.scaffold_folder, 'test_scaffolds_sample.csv')
        scaffold_sample = get_sample(args.n_scaffolds, scaffold_path,
                                     train, dataset=test_scaffolds)

    elif args.scaffold_source == 'test_scaffolds' and args.use_molgpt:
        scaffold_path = os.path.join(args.scaffold_folder, 'test_scaffolds_sample-molgpt.csv')
        scaffold_sample = get_sample(args.n_scaffolds, scaffold_path,
                                     train, dataset=test_scaffolds)

    # if False:
    #     plot_highlighted_smiles(scaffold_sample, save_folder, args.n_jobs)

    # generate SMILES

    for sid in range(len(scaffold_sample)):
        scaffold = scaffold_sample.loc[sid, 'scaffold']        
        gen_path = os.path.join(save_folder, f'gen{sid}.csv')
        
        if os.path.exists(gen_path):
            continue
        
        LOG.info(f'id = {sid}\tscaffold = {scaffold}')

        gen = sample_smiles(sampler, args.n_samples, scaffold,
                            args.batch_size, LOG)
        gen = pd.DataFrame(gen, columns=['SMILES'])
        gen.to_csv(gen_path)

    # define metrics

    used_metrics = ['scaffold', 'valid', 'unique', 'novel', 'intDiv', 'SSF',
                    'sim80', 'valid_in_tolerance', 'unique_in_tolerance']
    
    metric = OrderedDict()
    for met in used_metrics:
        metric[met] = []
    
    # compute metrics

    scaffold_sim = OrderedDict()

    for sid in range(len(scaffold_sample)):
        scaffold = scaffold_sample.loc[sid, 'scaffold']
        gen_path = os.path.join(save_folder, f'gen{sid}.csv')

        LOG.info(f'id = {sid}\tscaffold = {scaffold}')

        gen = pd.read_csv(gen_path, index_col=[0])
        gen = gen.dropna(subset=['smiles'])
        gen['mol'] = mapper(get_mol, gen['smiles'], args.n_jobs)
        
        valid = gen.dropna(subset='mol').copy()
        valid['smiles'] = mapper(get_canonical, valid['mol'], args.n_jobs)
        valid['scaffold'] = mapper(murcko_scaffold, valid['smiles'], args.n_jobs)
        similarity_fn = partial(murcko_scaffold_similarity, smi_or_mol2=scaffold)
        valid['scaffold_sim'] = mapper(similarity_fn, valid['smiles'], args.n_jobs)
        unique = valid.drop_duplicates(subset='SMILES').copy()

        metric['scaffold'].append(scaffold)
        metric['valid'].append(len(valid) / args.n_samples)
        metric['unique'].append(len(unique) / len(valid))
        metric['novel'].append(len(set(unique['smiles']) - set(train['smiles'])) / len(unique))
        metric['intDiv'].append(metrics.internal_diversity(unique['smiles'], args.n_jobs))
        metric['SSF'].append(len(valid[valid.scaffold_sim == 1]) / len(valid))
        metric['sim80'].append(len(valid[valid.scaffold_sim >= 0.8]) / len(valid))
        metric['valid_in_tolerance'].append(len(valid[valid.scaffold == scaffold]) / len(gen))
        metric['unique_in_tolerance'].append(len(unique[unique.scaffold == scaffold]) / len(gen))

        current_metric = pd.DataFrame(metric)
        current_metric.to_csv(metric_path)
        print(metric)
        
        scaffold_sim[sid] = valid['scaffold_sim']
        
    scaffold_sim = pd.DataFrame.from_dict(scaffold_sim, orient="index")
    scaffold_sim = scaffold_sim.transpose()
    
    print(scaffold_sim)
    scaffold_sim.to_csv(scaffold_sim_path)

    # plot Murcko scaffold similarity

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4.6))

    for sid in range(len(scaffold_sample)):
        murcko_scaffold_sim = scaffold_sim[sid].dropna('scaffold_sim')
        sns.kdeplot(data=murcko_scaffold_sim, ax=ax, shade=True,
                    linewidth=2, legend=False)
        ax.set_xlabel(xlabel='Murcko scaffold similarity', fontsize=16)
        ax.set_ylabel(ylabel='Density', fontsize=16)
        ax.tick_params(axis="both", which="major", labelsize=14)
    ax.set_xlim(0, 1.)

    fig.savefig(scaffold_sim_png_path, bbox_inches="tight")
