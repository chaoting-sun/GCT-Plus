import os
import numpy as np
import pandas as pd
from time import time
from rdkit import Chem
from rdkit.Chem import MolFromSmiles

import torch
from pathos.multiprocessing import ProcessingPool as Pool

from Utils.property import get_mol, property_prediction
from Inference.metrics import get_all_metrics, print_all_metrics
from Inference.utils import augment_props


# def augment_props(n, props, bound, varid=None, std=None):
#     props = np.array(props).reshape((1, 3))
#     props = np.repeat(props, n, axis=0)
#     if std is None:
#         return props

#     for id in varid:
#         props[:, id] = np.random.normal(0, std, (n,))
#         for i in range(n):
#             props[i, id] = min(props[i, id], bound[id][1])
#             props[i, id] = max(props[i, id], bound[id][0])
#     return props



def property_from_smiles(smiles, props_predictor, n_jobs=1):
    """
    Compute validity and properties of a
    list of smiles and return a pd.DataFrame
    """
    with Pool(n_jobs) as pool:
        props = pool.map(props_predictor, smiles)
    valids, props = map(lambda x: np.asarray(x), zip(*props))
    valids = pd.DataFrame(data={'valid': valids})
    props = pd.DataFrame(props, columns=["logp_p", "tpsa_p", "qed_p"])
    props = pd.concat([valids, props], axis=1)

    smiles = pd.DataFrame(data={'smiles': smiles})
    smiles_props = pd.concat([smiles, props], axis=1)

    return smiles_props


def props_predictor_wrapper(predictor, conditions):
    def props_predictor(smiles):
        mol = MolFromSmiles(smiles)
        if mol is not None:
            valid = 1
            props = [predictor[c](mol) for c in conditions]
        else:
            valid = 0
            props = [np.nan]*len(conditions)
        return valid, props
    return props_predictor


def store_properties(conditions, gen_smiles, smiles_path,
                     logp_t, tpsa_t, qed_t, logger):
    with open(smiles_path, "w", buffering=10) as ptr:
        ptr.write(f"number\tsmiles\tvalid\t"
                  f"logp_t\ttpsa_t\tqed_t\t"
                  f"logp_p\ttpsa_p\tqed_p\n")

        for i in range(len(gen_smiles)):
            mol = Chem.MolFromSmiles(gen_smiles[i])
            if mol is not None:
                valid = 1
                logp_p, tpsa_p, qed_p = (property_prediction[c](mol)
                                         for c in conditions)
            else:
                valid = 0
                logp_p = tpsa_p = qed_p = np.nan

            line = f"{i+1}\t{gen_smiles[i]}\t{valid}\t"     \
                   f"{logp_t:.2f}\t{tpsa_t:.2f}\t{qed_t:.2f}\t" \
                   f"{logp_p:.2f}\t{tpsa_p:.2f}\t{qed_p:.2f}"

            ptr.write(line+"\n")
            logger.info(
                f'- {gen_smiles[i]:<50} -> {logp_p:.2f}\t{tpsa_p:.2f}\t{qed_p:.2f}')


def save_smiles_props(sampler, target_props, props_bounds,
                      calc_props, n, n_jobs, save_path):
    for i, props in enumerate(target_props):
        file_name = f'{props[0]:.2f}_{props[1]:.2f}_{props[2]:.2f}'
        file_path = os.path.join(save_path, f'{file_name}.csv')
        print(f'{i:<3} target props:', props)
        if os.path.exists(file_path):
            continue
        
        z, _ = sampler.sample_z_from_data(n)
        props_t = augment_props(n, props, props_bounds)
        smiles = sampler.sample_smiles(torch.Tensor(props_t),
                                       z, transform=True)[0]
        smiles_props = property_from_smiles(smiles, calc_props, n_jobs)
        props_t = pd.DataFrame(props_t, columns=["logp_t","tpsa_t","qed_t"])

        smiles_props = pd.concat([smiles_props, props_t], axis=1)
        smiles_props.to_csv(file_path)
        print("Generate smiles:", smiles[:4])


def save_metrics(target_props, save_path, train_smiles, n_jobs):
    all_gen = None
    
    with open(os.path.join(save_path, 'statistics.csv'), 'w') as ptr:
        for i, props in enumerate(target_props):
            file_name = f'{props[0]:.2f}_{props[1]:.2f}_{props[2]:.2f}'
            file_path = os.path.join(save_path, f'{file_name}.csv')
            
            gen = pd.read_csv(file_path, index_col=[0])
            all_gen = pd.concat([all_gen, gen], axis=0)
            
            all_metrics = get_all_metrics(gen, train_smiles, n_jobs)
            header, body = print_all_metrics(all_metrics)
            if i == 0:
                ptr.write(header+'\n')
            ptr.write(body+'\n')

    all_gen = all_gen.reset_index()
    all_metrics = get_all_metrics(all_gen, train_smiles, n_jobs)
    header, body = print_all_metrics(all_metrics)
    
    with open(os.path.join(save_path, "overall_statistics.csv"), 'w') as ptr:
        ptr.write(header+'\n')
        ptr.write(body+'\n')


def uniform_generation(args, sampler, train_smiles, logger):
    save_folder = os.path.join(args.storage_path, "uniform_generation")
    os.makedirs(save_folder, exist_ok=True)

    log_path = os.path.join(save_folder, "record.log")
    LOG = logger('uniform_generation', log_path=log_path)

    calc_props = props_predictor_wrapper(property_prediction,
                                         args.conditions)

    props_bounds = {
        "logp": [args.logp_lb, args.logp_ub],
        "tpsa": [args.tpsa_lb, args.tpsa_ub],
        "qed":  [args.qed_lb, args.qed_ub]
    }

    target_props = np.array(np.meshgrid(
        np.linspace(args.logp_lb, args.logp_ub, num=args.n_each_prop),
        np.linspace(args.tpsa_lb, args.tpsa_ub, num=args.n_each_prop),
        np.linspace(args.qed_lb, args.qed_ub, num=args.n_each_prop))) \
        .T.reshape(-1, 3)

    LOG.info(f"Get smiles and properties...")    
    save_smiles_props(sampler, target_props, props_bounds, calc_props,
                      args.n_each_sampling, args.n_jobs, save_folder)

    LOG.info(f"Compute errors...")    
    save_metrics(target_props, save_folder, train_smiles, args.n_jobs)