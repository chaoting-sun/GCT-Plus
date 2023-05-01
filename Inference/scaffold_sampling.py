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


def compute_errors(
        trg_prop: np.array,
        gen_prop: pd.DataFrame,
        property_name: List,
        error_fn: List[Callable]
    ):
    error_dict = OrderedDict()
    
    for i, p in enumerate(property_name):
        trgv_list = [trg_prop[i] for _ in range(len(gen_prop))]
        genv_list = gen_prop[p]
        error_dict.update((f'{fn.__name__}_{p}',
                           fn(trgv_list, genv_list))
                           for fn in error_fn)
    return error_dict


def compute_records(
        scaffold_smi: str,
        gen_smi: List,
        trg_prop: List,
        property_fn: Dict[str, Callable],
        error_fn: Dict[str, Callable],
        similarity_fn: Callable,
        train_smiles: List,
        n_jobs=1
    ):
    """compute and return records
    
    Args:
        src_smi: scaffold in this case
        gen_smi: a list of generated smiles
        trg_prop: a list of target properties
        property_fn: a list of property functions
        error_fn: a dict of error functions
        similarity_fn: similarity function
        n_jobs: available cpus
    
    Returns:
        records
    """
    records = OrderedDict()

    # valid_mol, valid_smi = smi_to_mol(gen_smi, include_smi=True, n_jobs=n_jobs)
    
    
    # valid_mol = mapper(get_mol, gen_smi, n_jobs)
    # valid_mol = [m for m in valid_mol if m is not None]
    # valid_smi = mol_to_smi(valid_mol, n_jobs)
    
    valid_smi, valid_mol = get_molgpt_valid_smi(
        scaffold_smi, gen_smi, include_mol=True,
        n_jobs=n_jobs)

    unique_smi = list(set(valid_smi))
    # valid_smi is canonicalized
    
    # properties
    records['scaffold'] = scaffold_smi
    for i, p in enumerate(property_fn):    
        records[f'trg_{p}'] = trg_prop[i]

    # validity
    records['valid'] = len(valid_smi) / len(gen_smi)
    
    if len(valid_smi) > 0:
        # uniqueness, novelty, diversity, SSF
        records['unique'] = len(unique_smi) / len(valid_smi)
        records['novel'] = metrics.novelty(valid_smi, train_smiles, n_jobs)
        records['diversity'] = metrics.internal_diversity(valid_smi, n_jobs)
        with Pool(n_jobs) as pool:
            murcko_sca = pool.map(murcko_scaffold, valid_smi)
        records['SSF'] = (sum([1 for sca in murcko_sca
                          if sca == scaffold_smi]) / len(murcko_sca))
        
        # similarity
        similarity_to_src = partial(similarity_fn, smi_or_mol2=get_mol(scaffold_smi))
        with Pool(n_jobs) as pool:
            similarity = pool.map(similarity_to_src, valid_mol)
        similarity = [s for s in similarity if s is not None]
        records['avg_similarity'] = sum(similarity) / len(similarity)

        # error        
        gen_prop = mols_to_props(valid_mol, property_fn, n_jobs=n_jobs)
        for i, p in enumerate(property_fn):
            genv_list = gen_prop[p]
            trgv_list = [trg_prop[i] for _ in range(len(gen_prop))]
            records[f'SD_{p}'] = genv_list.std()
            for e in error_fn:
                records[f'{e}_{p}'] = error_fn[e](trgv_list, genv_list)
        
    else:
        records['unique'] = records['novel'] = records['diversity'] = records['SSF'] = 0
        records['avg_similarity'] = 0
        for p in property_fn:
            records[f'SD_{p}'] = 0
            for e in error_fn:
                records[f'{e}_{p}'] = 0
        murcko_sca = []
    
    return records


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


class RecordComputation:
    def __init__(self, property_list, n_jobs=8):
        self.n_jobs = n_jobs
        self.property_fn = get_property_fn(property_list)
        self.error_fn = get_error_fn(['MSE', 'MAE', 'AMSD', 'AARD'])
    
    
    def get_valid(self, smiles_list, structure=None,
                  method='molgpt'):
        if method == 'molgpt':
            assert structure is not None
            similarity_fn = partial(murcko_scaffold_similarity,
                                    smi_or_mol2=structure)
            with Pool(self.n_jobs) as pool:
                similarity = pool.map(similarity_fn, smiles_list)
            valid_smi = []
            for i, sim in enumerate(similarity):
                if sim != None and sim >= 0.80:
                    valid_smi.append(smiles_list[i])
        else:
            with Pool(self.n_jobs) as pool:
                valid_smi = pool.map(get_mol, smiles_list)
        
        with Pool(self.n_jobs) as pool:
            valid_mol = pool.map(get_mol, valid_smi)
        return valid_smi, valid_mol
    
    
    def compute_basis_metric(self, gen_smi, structure):
        records = OrderedDict()
        
        valid_smi, valid_mol = self.get_valid(gen_smi, structure=structure)
        unique_smi = list(set(valid_smi))
        
        records['structure'] = structure
        records['validity'] = len(valid_smi) / len(gen_smi)
        
    

    # records = OrderedDict()

    # valid_mol, valid_smi = smi_to_mol(gen_smi, include_smi=True, n_jobs=n_jobs)
    # # valid_smi, valid_mol = get_molgpt_valid_smi(
    # #     src_smi, gen_smi, include_mol=True,
    # #     n_jobs=n_jobs)    
    # # print('valid', valid_smi, valid_mol)

    # unique_smi = list(set(valid_smi))
    # # valid_smi is canonicalized
    
    # # properties
    # src_mol = get_mol(src_smi)
    # records['src'] = src_smi
    # for i, p in enumerate(property_fn):
    #     records[f'src_{p}'] = property_fn[p](src_mol)
    # for i, p in enumerate(property_fn):    
    #     records[f'trg_{p}'] = trg_prop[i]

    # # validity
    # records['valid'] = len(valid_smi) / len(gen_smi)
    
    # if len(valid_smi) > 0:
    #     # uniqueness, novelty, diversity, SSF
    #     records['unique'] = len(unique_smi) / len(valid_smi)
    #     records['novel'] = metrics.novelty(valid_smi, train_smiles, n_jobs)
    #     records['diversity'] = metrics.internal_diversity(valid_smi, n_jobs)
    #     with Pool(n_jobs) as pool:
    #         murcko_sca = pool.map(murcko_scaffold, valid_smi)
    #     records['SSF'] = (sum([1 for sca in murcko_sca
    #                       if sca == src_smi]) / len(murcko_sca))
        
    #     # similarity
    #     similarity_to_src = partial(similarity_fn, smi_or_mol2=src_mol)
    #     with Pool(n_jobs) as pool:
    #         similarity = pool.map(similarity_to_src, valid_mol)
    #     records['avg_similarity'] = sum(similarity) / len(similarity)

    #     # error        
    #     gen_prop = mols_to_props(valid_mol, property_fn, n_jobs=n_jobs)
    #     for i, p in enumerate(property_fn):
    #         genv_list = gen_prop[p]
    #         trgv_list = [trg_prop[i] for _ in range(len(gen_prop))]
    #         records[f'SD_{p}'] = genv_list.std()
    #         for e in error_fn:
    #             records[f'{e}_{p}'] = error_fn[e](trgv_list, genv_list)
        
    # else:
    #     records['unique'] = records['novel'] = records['diversity'] = records['SSF'] = 0
    #     records['avg_similarity'] = 0
    #     for p in property_fn:
    #         records[f'SD_{p}'] = 0
    #         for e in error_fn:
    #             records[f'{e}_{p}'] = 0
    #     murcko_sca = []



"""
smiles generation
- fn1: input
"""



class StructureConditionedSampling:
    def __init__(self, args, generator, property_list,
                 error_fn, property_fn):
        self.generator = generator
        self.error_fn = error_fn
        self.property_fn = property_fn
        self.property_list = property_list
        
    def generate_conditioned_smiles_sample(self, structure, n, prop=None,
                                           transform=True, batch_size=512):
        gens, toklens, toklen_gens = [], [], []

        prop = np.repeat([prop], n, axis=0)
        n_batch = int(np.ceil(n / batch_size))
        
        for i in range(n_batch):
            sid = batch_size * i
            eid = batch_size * (i+1) if eid <= n else n
            
            gen, toklen, toklen_gen = self.generator.sample_smiles(
                prop[sid:eid, :], structure, transform=transform
            )
            gens.extend(gen)
            toklens.extend(toklen)
            toklen_gens.extend(toklen_gen)

        return gens, toklens, toklen_gens

     
    def sample_smiles(self, df_data, n, LOG):
        for i in range(len(df_data)):
            structure = df_data.loc[i, 'scaffold']
            properties = df_data.loc[i, self.property_list]
            
            LOG.info('generate smiles conditioned on structures (and properties)')
            
            outputs = self.generate_conditioned_smiles_sample(structure, n,
                                                              properties)
            gens, toklens, toklen_gens = outputs
            
            LOG.info('compute records...')

            records, unique_smi = compute_records(
                src, gen, prop, property_fn,
                error_fn, murcko_scaffold_similarity,
                train_smiles, args.n_jobs)

            LOG.info('save records...')
            LOG.info(records)
            
            records = OrderedDict([(k, [records[k]])
                                   for k in records])
            records = pd.DataFrame(records)
            if pid == 0:
                records.to_csv(os.path.join(save_folder, f'record{src_id}.csv'),
                               index=False)
            else:
                records.to_csv(os.path.join(save_folder, f'record{src_id}.csv'),
                               index=False, mode='a', header=False)
            
            # LOG.info('plot smiles...')
            
            # plot_smiles(src_sca, os.path.join(save_folder, f'scaffold{sid}.png'))
            # print(len(unique_smi))
            # sampled_smi = np.random.choice(unique_smi, 24, replace=False)
            # plot_smiles_group(sampled_smi,
            #                   save_path=os.path.join(save_folder, f'prediction{sid}.png'),
            #                   n_per_mol=6,
            #                   n_jobs=args.n_jobs)        
    

@torch.no_grad()
def scaffold_sampling(
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

    n_samples = 1000
    batch_size = 1000
    n_batch = int(np.ceil(n_samples / batch_size))

    # interested_properties = ['logP', 'tPSA', 'QED',
    #                         'SAS',   'NP', 'MW' ,
    #                         'HAC',  'HBA', 'HBD']

    error_fn = get_error_fn(['MSE', 'MAE', 'AMSD', 'AARD'])
    property_fn = get_property_fn(args.property_list)
    trg_prop_list = get_trg_prop(args.benchmark, args.property_list)

    args.model_path = os.path.join(args.train_path, args.benchmark,
                                   args.model_name, f'model_{args.epoch}.pt')
    generator = get_generator(args, SRC, TRG, toklen_data, scaler, device)

    data_folder = os.path.join(args.infer_path, args.benchmark, 'scaffold_sampling')

    if False:
        save_folder = os.path.join(args.infer_path, args.benchmark, 'scaffold_sampling',
                                f'{args.model_name}_{args.epoch}_molgpt')
        os.makedirs(save_folder, exist_ok=True)

        LOG = logger(name='scaffold sampling', log_path=os.path.join(save_folder, 'record.log'))

        LOG.info('prepare input scaffold and target properties...')
        
        data_path = os.path.join(data_folder, f'molgpt_sample.csv')

        molgpt_samples = [ 'O=C(Cc1ccccc1)NCc1ccccc1',
                           'c1cnc2[nH]ccc2c1',
                           'c1ccc(-c2ccnnc2)cc1',
                           'c1ccc(-n2cnc3ccccc32)cc1',
                           'O=C(c1cc[nH]c1)N1CCN(c2ccccc2)CC1'
                        ]
        samples = pd.DataFrame(molgpt_samples, columns=['scaffold'])
        samples.to_csv(data_path)
        samples = pd.read_csv(data_path, index_col=[0])

    else:
        save_folder = os.path.join(args.infer_path, args.benchmark, 'scaffold_sampling',
                                f'{args.model_name}_{args.epoch}')
        os.makedirs(save_folder, exist_ok=True)

        LOG = logger(name='scaffold sampling', log_path=os.path.join(save_folder, 'record.log'))

        LOG.info('prepare input scaffold and target properties...')
        
        scaffold_cnt = df_train['scaffold'].value_counts()
        top_10 = pd.DataFrame({ 'scaffold': scaffold_cnt.index[:10] })
        top_10.to_csv(os.path.join(data_folder, 'train_top10.csv'))
        
        # scaffold_cnt = df_test_scaffolds['scaffold'].value_counts()
        # top_10 = pd.DataFrame({ 'scaffold': scaffold_cnt.index[:10] })
        # top_10.to_csv(os.path.join(data_folder, 'test_scaffolds_top10.csv'))
        
        # df_test_scaffolds = df_test_scaffolds.dropna(subset=['scaffold'])
        # df_test_scaffolds = df_test_scaffolds.drop_duplicates(ignore_index=True)
        
        # data_path = os.path.join(data_folder, f'sample.csv')

        # set_seed(0)
    
        # if not os.path.exists(data_path):
        #     samples = df_test_scaffolds.sample(n=n_samples, ignore_index=True)
        #     samples.to_csv(data_path)

    samples = pd.read_csv(os.path.join(data_folder, 'sample.csv'), index_col=[0])
    
    for sid in range(len(samples)):
        for pid, prop in enumerate(trg_prop_list):
            print('sample molecules - properties:', prop)
            
            suffix = '-'.join(map(str, prop))
            gen_path = os.path.join(save_folder, f's{sid}_p{suffix}_gen.csv')
            property_path = os.path.join(save_folder, f's{sid}_p{suffix}_prop.csv')

            scaffold = samples.loc[sid, 'scaffold']
            dconds = np.repeat([prop], n_samples, axis=0)

            gen = []

            # if not os.path.exists(gen_path):
            for b in range(n_batch):
                batch_head_id = batch_size * b
                batch_tail_id = batch_size * (b+1)
                if batch_tail_id > n_samples:
                    batch_tail_id = n_samples
                
                current_gen, *_ = generator.sample_smiles(
                    dconds[batch_head_id:batch_tail_id, :],
                    scaffold, transform=True)
                gen.extend(current_gen)

                print('process:', batch_tail_id)
            
            gen = pd.DataFrame(gen, columns=['SMILES'])
            gen.to_csv(gen_path)

            gen = pd.read_csv(gen_path, index_col=[0])
            print('gen:\n', gen)

            print('evaluate: compute properties')

            # if not os.path.exists(property_path):
            gen = gen.dropna(subset=['SMILES'])
            mols = mapper(get_mol, gen['SMILES'], args.n_jobs)
            mols = [m for m in mols if m is not None]
            smiles = pd.DataFrame(mol_to_smi(mols, args.n_jobs), columns=['SMILES'])

            props = mols_to_props(mols, property_fn, n_jobs=args.n_jobs)
            props = pd.concat([smiles, props], axis=1)
            props.to_csv(property_path)

            props = pd.read_csv(property_path, index_col=[0])
            print('properties:\n', props)

            records = compute_records(
                scaffold, gen['SMILES'], prop, property_fn,
                error_fn, murcko_scaffold_similarity,
                df_train['smiles'], args.n_jobs)

            LOG.info('save records...')
            LOG.info(records)

            records = OrderedDict([(k, [records[k]])
                                   for k in records])
            records = pd.DataFrame(records)
            
            if pid == 0:
                records.to_csv(os.path.join(save_folder, f'record{sid}.csv'),
                               index=False)
            else:
                records.to_csv(os.path.join(save_folder, f'record{sid}.csv'),
                               index=False, mode='a', header=False)

        # LOG.info('plot smiles...')
        
        # plot_smiles(src_sca, os.path.join(save_folder, f'scaffold{sid}.png'))
        # print(len(unique_smi))
        # sampled_smi = np.random.choice(unique_smi, 24, replace=False)
        # plot_smiles_group(sampled_smi,
        #                   save_path=os.path.join(save_folder, f'prediction{sid}.png'),
        #                   n_per_mol=6,
        #                   n_jobs=args.n_jobs)