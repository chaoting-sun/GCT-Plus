import os
import itertools
import numpy as np
import pandas as pd
from moses.metrics import metrics
from functools import partial
from collections import OrderedDict
from typing import Callable, List, Dict
from pathos.multiprocessing import ProcessingPool as Pool

from Model.build_model import get_generator
from Utils.smiles import get_mol, murcko_scaffold, murcko_scaffold_similarity, plot_smiles_group, plot_smiles, smi_to_mol
from Utils.properties import mols_to_props, get_property_fn
from Utils.metric import get_error_fn
from Utils import DataloaderPreparation


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
        src_smi: str,
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

    valid_mol, valid_smi = smi_to_mol(gen_smi, include_smi=True, n_jobs=n_jobs)
    # valid_smi, valid_mol = get_molgpt_valid_smi(
    #     src_smi, gen_smi, include_mol=True,
    #     n_jobs=n_jobs)    
    # print('valid', valid_smi, valid_mol)

    unique_smi = list(set(valid_smi))
    # valid_smi is canonicalized
    
    # properties
    src_mol = get_mol(src_smi)
    records['src'] = src_smi
    for i, p in enumerate(property_fn):
        records[f'src_{p}'] = property_fn[p](src_mol)
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
                          if sca == src_smi]) / len(murcko_sca))
        
        # similarity
        similarity_to_src = partial(similarity_fn, smi_or_mol2=src_mol)
        with Pool(n_jobs) as pool:
            similarity = pool.map(similarity_to_src, valid_mol)
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
    
    return records, unique_smi


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


def lerp(z1, z2, alpha):
    """linear interpolation"""
    z = (1 - alpha) * z1 + alpha * z2
    return z


def slerp(z1, z2, alpha):
    """spherical linear interpolation"""
    z1_normalized = z1 / np.linalg.norm(z1)
    z2_normalized = z2 / np.linalg.norm(z2)

    omega = np.arccos(np.dot(z1_normalized, z2_normalized))
    z = (np.sin((1 - alpha) * omega) * z1 + np.sin(alpha * omega) * z2) / np.sin(omega)
    return z


def smoothness_check(
        args,
        toklen_data,
        train_smiles,
        test_smiles,
        scaler,
        SRC,
        TRG,
        device,
        logger
    ):

    n_tests = 5
    n_samples = 128
    n_each_batch = 128

    # create file path and folder
    save_folder = os.path.join(args.infer_path, args.benchmark,
                               'smoothness_check', args.model_name)
    os.makedirs(save_folder, exist_ok=True)

    LOG = logger(name='scaffold sampling', log_path=os.path.join(save_folder, 'record.log'))

    LOG.info('create a generator...')

    args.model_path = os.path.join(args.train_path, args.benchmark,
                                   args.model_name, f'model_{args.epoch}.pt')
    generator = get_generator(args, SRC, TRG, toklen_data, scaler, device)

    LOG.info('prepare error/property functions...')

    error_fn = get_error_fn(['MSE', 'MAE', 'AMSD', 'AARD'])
    property_fn = get_property_fn(args.property_list)

    LOG.info('prepare input scaffold and target properties...')

    test_smiles = np.random.choice(test_smiles, n_tests, replace=False)
    
    property_fn = get_property_fn(args.property_list)
    