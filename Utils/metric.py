import numpy as np
    
    
def mae(trg_prop, gen_prop):
    abs_dprop = np.zeros((len(trg_prop),))
    for i in range(len(trg_prop)):
        abs_dprop[i] = abs(gen_prop[i] - trg_prop[i])
    return abs_dprop.sum() / len(abs_dprop)


def mse(trg_prop, gen_prop):
    dprop = np.zeros((len(trg_prop),))
    for i in range(len(trg_prop)):
        dprop[i] = gen_prop[i] - trg_prop[i]
    return dprop.sum() / len(dprop)


def aard(trg_prop, gen_prop):
    abs_rel_dprop = np.zeros((len(trg_prop),))
    for i in range(len(trg_prop)):
        abs_rel_dprop[i] = abs(gen_prop[i] - trg_prop[i]) / trg_prop[i]
    return abs_rel_dprop.sum() / len(abs_rel_dprop)
    
    
def amsd(trg_prop, gen_prop):
    rel_dprop = np.zeros((len(trg_prop),))
    for i in range(len(trg_prop)):
        rel_dprop[i] = (gen_prop[i] - trg_prop[i]) / trg_prop[i]
    return rel_dprop.sum() / len(rel_dprop)


def get_error(trg_prop, gen_prop, err_type):
    error_dict = {}
    for err in err_type:
        err_fcn = eval(err)
        error_dict[err] = err_fcn(trg_prop, gen_prop)
    return error_dict
    
    

        