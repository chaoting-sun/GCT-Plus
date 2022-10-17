import numpy as np

from multiprocessing import Pool
from Utils.mapper import mapper
from Utils.property import get_mol
from moses.metrics import metrics


def _valid(valid_preds, preds): return len(valid_preds) / \
    len(preds)*100 if len(preds) else 0


def _unique(smiles): return len(np.unique(smiles)) / \
    len(smiles)*100 if len(smiles) > 0 else 0


def _novelty(gen_smiles, train, n_jobs):
    if len(gen_smiles) == 0:
        return 0
    close_pool = False
    if n_jobs != 1:
        pool = Pool(n_jobs)
        close_pool = True
    else:
        pool = 1
    mols = mapper(pool)(get_mol, gen_smiles)
    gen_novelty = metrics.novelty(mols, train, n_jobs)
    if close_pool:
        pool.close()
        pool.join()
    return gen_novelty*100


def _intdiv(valid_smiles): return metrics.internal_diversity(
    valid_smiles)*100 if len(valid_smiles) > 0 else 0


def _mae(valid_dif): return valid_dif.apply(np.abs).mean()
def _mse(valid_dif): return valid_dif.mean()
def _max(valid_dif): return valid_dif.max()
def _min(valid_dif): return valid_dif.min()


def all_metrics(preds, train, n_jobs):
    valid_preds = preds.loc[preds['valid'] == 1].copy()
    valid_preds['logp_diff'] = valid_preds['logp_p'] - valid_preds['logp_t']
    valid_preds['tpsa_diff'] = valid_preds['tpsa_p'] - valid_preds['tpsa_t']
    valid_preds['qed_diff'] = valid_preds['qed_p'] - valid_preds['qed_t']

    header = 'valid(%)\tunique(%)\tnovelty(%)\tdiversity(%)\t'  \
             'logp_mae\ttpsa_mae\tqed_mae\t'                    \
             'logp_mse\ttpsa_mse\tqed_mse\t'                    \
             'logp_max\ttpsa_max\tqed_max\t'                    \
             'logp_min\ttpsa_min\tqed_min\t'                    \
             'logp_aard(%)\ttpsa_aard(%)\tqed_aard(%)\t'        \
             'logp_amsd(%)\ttpsa_amsd(%)\tqed_amsd(%)'          \

    line = f"{_valid(valid_preds, preds):.3f}\t"                     \
           f"{_unique(valid_preds['smiles']):.3f}\t"                 \
           f"{_novelty(valid_preds['smiles'], train, n_jobs):.3f}\t" \
           f"{_intdiv(valid_preds['smiles']):.3f}\t" \
           f"{_mae(valid_preds['logp_diff']):.3f}\t" \
           f"{_mae(valid_preds['tpsa_diff']):.3f}\t" \
           f"{_mae(valid_preds['qed_diff']):.3f}\t"  \
           f"{_mse(valid_preds['logp_diff']):.3f}\t" \
           f"{_mse(valid_preds['tpsa_diff']):.3f}\t" \
           f"{_mse(valid_preds['qed_diff']):.3f}\t"  \
           f"{_max(valid_preds['logp_diff']):.3f}\t" \
           f"{_max(valid_preds['tpsa_diff']):.3f}\t" \
           f"{_max(valid_preds['qed_diff']):.3f}\t"  \
           f"{_min(valid_preds['logp_diff']):.3f}\t" \
           f"{_min(valid_preds['tpsa_diff']):.3f}\t" \
           f"{_min(valid_preds['qed_diff']):.3f}\t"  \
           f"{(valid_preds['logp_diff'] / valid_preds['logp_t']).abs().mean()*100:.3f}\t" \
           f"{(valid_preds['tpsa_diff'] / valid_preds['tpsa_t']).abs().mean()*100:.3f}\t" \
           f"{(valid_preds['qed_diff'] / valid_preds['qed_t']).abs().mean()*100:.3f}\t"   \
           f"{(valid_preds['logp_diff'] / valid_preds['logp_t']).mean()*100:.3f}\t"       \
           f"{(valid_preds['tpsa_diff'] / valid_preds['tpsa_t']).mean()*100:.3f}\t"       \
           f"{(valid_preds['qed_diff'] / valid_preds['qed_t']).mean()*100:.3f}"           \

    return header, line