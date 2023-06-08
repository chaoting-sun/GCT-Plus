from functools import partial
from moses.metrics import metrics


def MSE(trgv, genv):
    delv = [genv[i] - trgv[i] for i in range(len(trgv))]
    return sum(delv) / len(delv)


def MAE(trgv, genv):
    abs_delv = [abs(genv[i] - trgv[i]) for i in range(len(trgv))]
    return sum(abs_delv) / len(abs_delv)
    
    
def AMSD(trgv, genv):
    delv = [genv[i] - trgv[i] for i in range(len(trgv))]
    rel_delv = [delv[i] / trgv[i] for i in range(len(trgv))]
    return sum(rel_delv) / len(rel_delv)


def AARD(trgv, genv):
    abs_delv = [abs(genv[i] - trgv[i]) for i in range(len(trgv))]
    rel_abs_delv = [abs_delv[i] / trgv[i] for i in range(len(trgv))]
    return sum(rel_abs_delv) / len(rel_abs_delv)


def get_error_fn(err_type):
    all_fn = {
        'MSE' : MSE,
        'MAE' : MAE,
        'AMSD': AMSD,
        'AARD': AARD,   
    }
    return { err: all_fn[err] for err in err_type }


def get_metric_fn(metric_type, train_smiles=None, n_jobs=1):
    if 'novelty' in metric_type:
        assert train_smiles is not None
    all_fn = {
        'valid'    : partial(metrics.fraction_valid, n_jobs=n_jobs),
        'unique'   : partial(metrics.fraction_unique, n_jobs=n_jobs),
        'novel'    : partial(metrics.novelty, train=train_smiles, n_jobs=n_jobs),
        'intDiv'   : partial(metrics.internal_diversity, n_jobs=n_jobs)
        # 'diversity': partial(metrics.internal_diversity, n_jobs=n_jobs)
    }
    return { met: all_fn[met] for met in metric_type }