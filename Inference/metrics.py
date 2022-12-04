import numpy as np

from multiprocessing import Pool
# from Utils.mapper import mapper
# from Utils.property import get_mol
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
    

def get_all_metrics(gen, train_smiles, n_jobs):
    validsmi = metrics.remove_invalid(gen["smiles"])
    valid = get_valid(gen["smiles"], n_jobs)
    unique = get_unique(validsmi, n_jobs)
    novel = get_novelty(validsmi, train_smiles, n_jobs)
    intDiv = get_interval_diversity(validsmi, n_jobs)

    logpErr = get_errors(gen['logp_p'], gen['logp_t'])
    tpsaErr = get_errors(gen['tpsa_p'], gen['tpsa_t'])
    qedErr = get_errors(gen['qed_p'], gen['qed_t'])
    
    all_metrics = {
        "valid": valid,
        "unique": unique,
        "novel": novel,
        "intDiv": intDiv,
        "logpErr": logpErr,
        "tpsaErr": tpsaErr,
        "qedErr": qedErr
    }
    return all_metrics


def print_all_metrics(metrics):
    header = 'valid,unique,novel,intDiv,'  \
             'logpMAE,tpsaMAE,qedMAE,'     \
             'logpMSE,tpsaMSE,qedMSE,'     \
             'logpMAX,tpsaMAX,qedMAX,'     \
             'logpMIN,tpsaMIN,qedMIN,'     \
             'logpAARD,tpsaAARD,qedAARD,'  \
             'logpAMSD,tpsaAMSD,qedAMSD'

    body = f"{metrics['valid']:.4f},"   \
           f"{metrics['unique']:.4f},"  \
           f"{metrics['novel']:.4f},"   \
           f"{metrics['intDiv']:.4f},"  \
        \
           f"{metrics['logpErr']['mae']:.4f},"  \
           f"{metrics['tpsaErr']['mae']:.4f},"  \
           f"{metrics['qedErr']['mae']:.4f},"   \
        \
           f"{metrics['logpErr']['mse']:.4f},"  \
           f"{metrics['tpsaErr']['mse']:.4f},"  \
           f"{metrics['qedErr']['mse']:.4f},"   \
        \
           f"{metrics['logpErr']['max']:.4f},"  \
           f"{metrics['tpsaErr']['max']:.4f},"  \
           f"{metrics['qedErr']['max']:.4f},"   \
        \
           f"{metrics['logpErr']['min']:.4f},"  \
           f"{metrics['tpsaErr']['min']:.4f},"  \
           f"{metrics['qedErr']['min']:.4f},"   \
        \
           f"{metrics['logpErr']['aard']:.4f}," \
           f"{metrics['tpsaErr']['aard']:.4f}," \
           f"{metrics['qedErr']['aard']:.4f},"  \
        \
           f"{metrics['logpErr']['amsd']:.4f}," \
           f"{metrics['tpsaErr']['amsd']:.4f}," \
           f"{metrics['qedErr']['amsd']:.4f}"   \

    return header, body


# def _valid(valid_preds, preds): return len(valid_preds) / \
#     len(preds)*100 if len(preds) else 0


# def _unique(smiles): return len(np.unique(smiles)) / \
#     len(smiles)*100 if len(smiles) > 0 else 0


# def _novelty(gen_smiles, train, n_jobs):
#     if len(gen_smiles) == 0:
#         return 0
#     close_pool = False
#     if n_jobs != 1:
#         pool = Pool(n_jobs)
#         close_pool = True
#     else:
#         pool = 1
#     mols = mapper(pool)(get_mol, gen_smiles)
#     gen_novelty = metrics.novelty(mols, train, n_jobs)
#     if close_pool:
#         pool.close()
#         pool.join()
#     return gen_novelty*100


# def _intdiv(valid_smiles): return metrics.internal_diversity(
#     valid_smiles)*100 if len(valid_smiles) > 0 else 0


# def _mae(valid_dif): return valid_dif.apply(np.abs).mean()
# def _mse(valid_dif): return valid_dif.mean()
# def _max(valid_dif): return valid_dif.max()
# def _min(valid_dif): return valid_dif.min()


# def all_metrics(preds, train, n_jobs):
#     valid_preds = preds.loc[preds['valid'] == 1].copy()
#     valid_preds['logp_diff'] = valid_preds['logp_p'] - valid_preds['logp_t']
#     valid_preds['tpsa_diff'] = valid_preds['tpsa_p'] - valid_preds['tpsa_t']
#     valid_preds['qed_diff'] = valid_preds['qed_p'] - valid_preds['qed_t']

#     header = 'valid(%)\tunique(%)\tnovelty(%)\tdiversity(%)\t'  \
#              'logp_mae\ttpsa_mae\tqed_mae\t'                    \
#              'logp_mse\ttpsa_mse\tqed_mse\t'                    \
#              'logp_max\ttpsa_max\tqed_max\t'                    \
#              'logp_min\ttpsa_min\tqed_min\t'                    \
#              'logp_aard(%)\ttpsa_aard(%)\tqed_aard(%)\t'        \
#              'logp_amsd(%)\ttpsa_amsd(%)\tqed_amsd(%)'          \

#     line = f"{_valid(valid_preds, preds):.3f}\t"                     \
#            f"{_unique(valid_preds['smiles']):.3f}\t"                 \
#            f"{_novelty(valid_preds['smiles'], train, n_jobs):.3f}\t" \
#            f"{_intdiv(valid_preds['smiles']):.3f}\t" \
#            f"{_mae(valid_preds['logp_diff']):.3f}\t" \
#            f"{_mae(valid_preds['tpsa_diff']):.3f}\t" \
#            f"{_mae(valid_preds['qed_diff']):.3f}\t"  \
#            f"{_mse(valid_preds['logp_diff']):.3f}\t" \
#            f"{_mse(valid_preds['tpsa_diff']):.3f}\t" \
#            f"{_mse(valid_preds['qed_diff']):.3f}\t"  \
#            f"{_max(valid_preds['logp_diff']):.3f}\t" \
#            f"{_max(valid_preds['tpsa_diff']):.3f}\t" \
#            f"{_max(valid_preds['qed_diff']):.3f}\t"  \
#            f"{_min(valid_preds['logp_diff']):.3f}\t" \
#            f"{_min(valid_preds['tpsa_diff']):.3f}\t" \
#            f"{_min(valid_preds['qed_diff']):.3f}\t"  \
#            f"{(valid_preds['logp_diff'] / valid_preds['logp_t']).abs().mean()*100:.3f}\t" \
#            f"{(valid_preds['tpsa_diff'] / valid_preds['tpsa_t']).abs().mean()*100:.3f}\t" \
#            f"{(valid_preds['qed_diff'] / valid_preds['qed_t']).abs().mean()*100:.3f}\t"   \
#            f"{(valid_preds['logp_diff'] / valid_preds['logp_t']).mean()*100:.3f}\t"       \
#            f"{(valid_preds['tpsa_diff'] / valid_preds['tpsa_t']).mean()*100:.3f}\t"       \
#            f"{(valid_preds['qed_diff'] / valid_preds['qed_t']).mean()*100:.3f}"           \

#     return header, line