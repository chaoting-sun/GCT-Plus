import numpy as np
import torch


def augment_props(n, props, bound=None, varid=None, std=None):
    props = np.array(props).reshape((1, 3))
    props = np.repeat(props, n, axis=0)
    if std is None:
        return props

    # for i, id in enumerate(varid):
    #     props[:, id] = np.random.normal(0, std[i], (n,))
    #     for i in range(n):
    #         props[i, id] = min(props[i, id], bound[id][1])
    #         props[i, id] = max(props[i, id], bound[id][0])
    varid = int(varid)
    props[:, varid] += np.random.normal(0, std, (n,))
    for i in range(n):
        props[i, varid] = min(props[i, varid], bound[1])
        props[i, varid] = max(props[i, varid], bound[0])
    return props


def augment_z(n, z, std=None):
    # z: torch.Tensor
    assert z.dim() == 3
    z = z.repeat(n, 1, 1)
    if std:
        z += torch.empty_like(z).normal_(mean=0, std=std)
        return z
    return z