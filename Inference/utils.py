import numpy as np
import torch
from Model.build_model import get_model
from Inference.model_prediction import Predictor
from Inference.decode_algo import get_generator


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


def prepare_generator(args, SRC, TRG, toklen_data, scaler, device):
    model = get_model(args, len(SRC.vocab), len(TRG.vocab))
    model = model.to(device)
    model.eval()

    predictor = Predictor(args.use_cond2dec,
                          getattr(model, args.decode_type),
                          getattr(model, args.encode_type))

    kwargs = {
        'latent_dim'  : args.latent_dim,
        'max_strlen'  : args.max_strlen,
        'use_cond2dec': args.use_cond2dec,
        'decode_algo' : args.decode_algo,
        'toklen_data' : toklen_data,
        'scaler'      : scaler,
        'device'      : device,
        'TRG'         : TRG,
    }

    return get_generator(predictor, args.decode_algo, kwargs)