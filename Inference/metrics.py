import numpy as np
from multiprocessing import Pool
from moses.metrics import metrics
from moses.utils import mapper, get_mol
from moses.metrics import SNNMetric


def get_snn_from_mol(molList1, molList2):
    # Similarity to nearest neighbour
    # molList1_fp = SNNMetric().precalc()
    if len(molList1) == 0 or len(molList2) == 0:
        return 0
    return SNNMetric()(gen=molList1, ref=molList2)

def get_valid(smiles, n_jobs=1):
    if len(smiles) == 0:
        return 0
    return metrics.fraction_valid(smiles, n_jobs)

def get_unique(valid_smiles, n_jobs=1):
    if len(valid_smiles) == 0:
        return 0
    return metrics.fraction_unique(valid_smiles, n_jobs=n_jobs)

def get_novelty(valid_smiles, train_smiles, n_jobs):
    if len(valid_smiles) == 0:
        return 0
    return metrics.novelty(valid_smiles, train_smiles, n_jobs)

def get_interval_diversity(valid_smiles, n_jobs=1, p=1):
    if len(valid_smiles) == 0:
        return 0
    return metrics.internal_diversity(gen=valid_smiles, n_jobs=n_jobs, p=p)

def get_errors(smiles_props, target_props):
    # inputs are pd.Series / pd.DataFrame
    props_diff = target_props - smiles_props
    errors = {
        "mae": props_diff.apply(np.abs).mean(),
        "mse": props_diff.mean(),
        "max": props_diff.max(),
        "min": props_diff.min()
    }
    errors["aard"] = (props_diff / target_props).abs().mean()
    errors["amsd"] = (props_diff / target_props).mean()
    return errors

def get_basic_metrics(gen, train_smiles, n_jobs=1):
    validsmi = metrics.remove_invalid(gen)
    valid = get_valid(gen, n_jobs)
    unique = get_unique(validsmi, n_jobs)
    novel = get_novelty(validsmi, train_smiles, n_jobs)
    intDiv = get_interval_diversity(validsmi, n_jobs)    

    all_metrics = {
        "valid": valid,
        "unique": unique,
        "novel": novel,
        "intDiv": intDiv,
    }
    return all_metrics