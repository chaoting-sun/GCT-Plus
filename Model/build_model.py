import torch
from collections import OrderedDict
from Inference.decode_algo_test import sampling_tools
from Model import (
    CTF,
    Vaetf,
    Cvaetf,
    ATTENCVAETF,
    SEPCVAETF,
    SEPCVAETF2,
    ATTENCTF,
    ScaCvaetfV2
)

"""
model mapper
"""

model_dict = {
    'vaetf'      : Vaetf,
    'cvaetf'     : Cvaetf,
    'scacvaetfv1': Cvaetf,
    'scacvaetfv2': ScaCvaetfV2,
    'scacvaetfv3': Cvaetf,
    'ctf'        : CTF,
    'attenctf'   : ATTENCTF,
    'attencvaetf': ATTENCVAETF,
    'sepcvaetf'  : SEPCVAETF,
    'sepcvaetf2' : SEPCVAETF2,
}


transfer_model_dict = {
    'scacvaetfv2': 'scacvaetfv2',
    'attenctf'   : 'ctf',
    'attencvaetf': 'cvaetf',
    'mlpcvaetf'  : 'cvaetf',
    'sepcvaetf'  : 'cvaetf',
    'sepcvaetf2' : 'cvaetf',
}


transfer_train_params =  {
    'attenctf'   : ['rotator'],
    'attencvaetf': ['atten_mu', 'atten_logvar', 'atten_z'],
    'sepcvaetf'  : ['mu_rotator', 'logvar_rotator'],
}


def transfer_params(trained_model, new_model):
    new_dict = new_model.state_dict()
    for name, param in trained_model.named_parameters():
        if name in new_dict:
            new_dict[name].copy_(param)
        else:
            print('Layer Not Found:', name)
            exit()
    return new_model


def freeze_params(model, freeze_names=None, train_names=None):
    """
    freeze parameters of a model.
    freeze_names, train_names are lists
    """
    for name, param in model.named_parameters():
        name_split = name.split('.')
        if freeze_names and name_split[0] in freeze_names:
            param.requires_grad = False
        if train_names:
            if name_split[0] in train_names:
                param.requires_grad = True
            else:
                param.requires_grad = False
    return model


def extract_params(args, src_vocab_len, trg_vocab_len):
    return {
        'src_vocab'   : src_vocab_len,
        'trg_vocab'   : trg_vocab_len,
        'N'           : args.N,
        'd_model'     : args.d_model,
        'dff'         : args.d_ff,
        'h'           : args.H,
        'latent_dim'  : args.latent_dim,
        'dropout'     : args.dropout,
        'use_cond2dec': args.use_cond2dec,
        'use_cond2lat': args.use_cond2lat,
        'nconds'      : len(args.property_list)
    }


def load_state(model, model_path, rank):
    map_location = { 'cuda:%d' % 0: 'cuda:%d' % rank }
    model_state = torch.load(model_path, map_location)['model_state_dict']
    
    if list(model_state.keys())[0].split('.')[0] == 'module':
        model_state = OrderedDict([(k[7:], v) for k, v in model_state.items()])
        # remove 'modules' if it exists
        # ref: https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/14
    
    model.load_state_dict(model_state)
    return model 


def get_model(args, src_vocab_len, trg_vocab_len, rank):
    hyper_params = extract_params(args, src_vocab_len, trg_vocab_len)
    model = model_dict[args.model_type](**hyper_params)
    
    # model_path is not in args
    if hasattr(args, 'model_path'):
        model = load_state(model, args.model_path, rank)
        return model
    
    # original_model_path is in args
    if args.original_model_path is not None:
        o_model_type = transfer_model_dict[args.model_type]
        o_model = model_dict[o_model_type](**hyper_params)
        o_model = load_state(o_model, args.original_model_path, rank)

        model = transfer_params(o_model, model)
        model = freeze_params(model, train_names=transfer_train_params[args.model_type])
        return model
    return model


def get_generator(args, SRC, TRG, toklen_data, scaler, device):
    model = get_model(args, len(SRC.vocab), len(TRG.vocab), device)
    model = model.to(device)
    model.eval()

    print(f'#parameters: {sum(p.numel() for p in model.parameters())}')

    kwargs = {
        'top_k'       : args.top_k,
        'latent_dim'  : args.latent_dim,
        'max_strlen'  : args.max_strlen,
        'use_cond2dec': args.use_cond2dec,
        'decode_algo' : args.decode_algo,
        'toklen_data' : toklen_data,
        'cond_dim'    : len(args.property_list),
        'scaler'      : scaler,
        'device'      : device,
        'SRC'         : SRC,
        'TRG'         : TRG,
    }
    
    return sampling_tools[args.model_type](model, kwargs)


# def get_model(args, SRC_vocab_len, TRG_vocab_len):
#     if args.model_type == "transformer":
#         if args.use_molgct:
#             model_path = os.path.join(args.molgct_path, 'molgct.pt')
#             assert args.use_epoch == 0
#         elif args.use_epoch >= 1:
#             model_path = os.path.join(args.model_path, f'model_{args.use_epoch}.pt')
#         elif args.use_epoch == 0:
#             model_path = None

#         print(f"Build transformer with model path: {model_path}")
#         model = build_transformer(
#             SRC_vocab_len, TRG_vocab_len, args.N, args.d_model,
#             args.d_ff, args.H, args.latent_dim, args.dropout,
#             args.nconds, args.use_cond2dec, args.use_cond2lat, model_path
#         )
#         return model


# def build_transformer(hyperParameters, model_path, use_molgct):
#     model = Transformer(**hyperParameters)
#     if model_path is not None:
#         model_state = torch.load(model_path)
#         if use_molgct:
#             model.load_state_dict(model_state)
#         else:
#             model.load_state_dict(model_state['model_state_dict'])
#     return model


# def build_transformer(src_vocab_len, trg_vocab_len, N, d_model, d_ff,
#                       H, latent_dim, dropout, nconds, use_cond2dec,
#                       use_cond2lat, model_path):
#     tf = Transformer(src_vocab_len, trg_vocab_len, N, d_model, d_ff, H,
#                      latent_dim, dropout, nconds, use_cond2dec, use_cond2lat)
#     if model_path is not None:
#         tf.load_state_dict(torch.load(model_path))
#     return tfs


# def transfer_parameters(transformer, mlptransformer):
#     mlptf_dict = mlptransformer.state_dict()
#     for name, param in transformer.named_parameters():
#         if name in mlptf_dict:
#             mlptf_dict[name].copy_(param)
#         else:
#             print('Layer Not Found:', name)
#             exit()


# def freeze_parameters(mlptransformer, pass_keywords):
#     for name, param in mlptransformer.named_parameters():
#         name_split = name.split('.')
#         if name_split[0] in pass_keywords:
#             continue
#         param.requires_grad = False


# def build_ctf(hyperParameters, model_path=None):
#     hyperParameters['variational'] = False
#     model = CTF(**hyperParameters)
#     if model_path:
#         print("Use model path:", model_path)
#         model_state = torch.load(model_path)
#         model.load_state_dict(model_state['model_state_dict'])
#     return model


# def build_cvaetfcut(hyperParameters, model_path=None):
#     model = CVAETFCUT(**hyperParameters)
#     if model_path:
#         model_state = torch.load(model_path)
#         model.load_state_dict(model_state['model_state_dict'])
#     return model


# def build_cvaetf(rank, hyperParameters, model_path=None):
#     model = CVAETF(**hyperParameters)

#     if model_path:
#         map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
#         model_state = torch.load(model_path, map_location)

#         if 'molgct.pt' in model_path:
#             model.load_state_dict(model_state)
#         else:
#             model.load_state_dict(model_state['model_state_dict'])

#     return model


# def build_attenctf(hyperparams, ctf_path, aug_ctf_path=None):
#     hyperparams['variational'] = False
#     aug_ctf = ATTENCTF(**hyperparams)
#     if aug_ctf_path:
#         aug_ctf.load_state_dict(torch.load(aug_ctf_path)['model_state_dict'])
#     else:
#         ctf = build_ctf(hyperparams, ctf_path)
#         ctf.load_state_dict(torch.load(ctf_path)['model_state_dict'])
#         transfer_params(ctf, aug_ctf)
#     freeze_params(aug_ctf, train_names=['rotator'])
#     return aug_ctf


# def build_mlpcvaetf(hyperParams, cvaetf_path, aug_cvaetf_path=None):
#     aug_cvaetf = MLPCVAETF(**hyperParams)
#     if aug_cvaetf_path:
#         aug_cvaetf.load_state_dict(torch.load(
#             aug_cvaetf_path)['model_state_dict'])
#     else:
#         cvaetf = build_cvaetf(hyperParams, cvaetf_path)
#         # cvaetf.load_state_dict(torch.load(cvaetf_path)) # for molgct.pt
#         # for self-trained cvaetf
#         cvaetf.load_state_dict(torch.load(cvaetf_path)['model_state_dict'])
#         transfer_params(cvaetf, aug_cvaetf)
#     freeze_params(aug_cvaetf, train_names=['mlp'])
#     return aug_cvaetf


# def build_sepcvaetf(hyperParams, cvaetf_path, aug_cvaetf_path=None):
#     aug_cvaetf = SEPCVAETF(**hyperParams)
#     if aug_cvaetf_path:
#         aug_cvaetf.load_state_dict(torch.load(
#             aug_cvaetf_path)['model_state_dict'])
#     else:
#         cvaetf = build_cvaetf(hyperParams, cvaetf_path)
#         # cvaetf.load_state_dict(torch.load(cvaetf_path)) # for molgct.pt
#         # for self-trained cvaetf
#         cvaetf.load_state_dict(torch.load(cvaetf_path)['model_state_dict'])
#         transfer_params(cvaetf, aug_cvaetf)
#     freeze_params(aug_cvaetf, train_names=['mu_rotator', 'logvar_rotator'])
#     return aug_cvaetf


# def build_sepcvaetf2(hyperParams, cvaetf_path, aug_cvaetf_path=None):
#     aug_cvaetf = SEPCVAETF2(**hyperParams)
#     if aug_cvaetf_path:
#         aug_cvaetf.load_state_dict(torch.load(
#             aug_cvaetf_path)['model_state_dict'])
#     else:
#         cvaetf = build_cvaetf(hyperParams, cvaetf_path)
#         # cvaetf.load_state_dict(torch.load(cvaetf_path)) # for molgct.pt
#         # for self-trained cvaetf
#         cvaetf.load_state_dict(torch.load(cvaetf_path)['model_state_dict'])
#         transfer_params(cvaetf, aug_cvaetf)
#     freeze_params(aug_cvaetf, train_names=['mu_rotator', 'logvar_rotator'])
#     return aug_cvaetf


# def build_mlpcvaetfencoder(hyperParams, cvaetf_path, aug_cvaetf_path=None):
#     aug_cvaetf = MLPCVAETFEncoder(**hyperParams)
#     if aug_cvaetf_path:
#         aug_cvaetf.load_state_dict(torch.load(cvaetf_path)['model_state_dict'])
#     else:
#         cvaetf = build_cvaetf(hyperParams, cvaetf_path)
#         cvaetf.load_state_dict(torch.load(cvaetf_path))
#         # cvaetf.load_state_dict(torch.load(cvaetf_path)['model_state_dict'])
#         transfer_params(cvaetf, aug_cvaetf)
#     freeze_params(aug_cvaetf, train_names=['mlp'])
#     return aug_cvaetf


# def build_attencvaetf(rank,
#                       hyperParams,
#                       original_model_path,
#                       model_path=None,
#                       ):
#     aug_cvaetf = ATTENCVAETF(**hyperParams)

#     if model_path:
#         map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
#         aug_cvaetf.load_state_dict(
#             torch.load(model_path, map_location=map_location)['model_state_dict'])
#     else:
#         assert original_model_path is not None
#         cvaetf = build_cvaetf(rank, hyperParams, original_model_path)
#         # cvaetf.load_state_dict(torch.load(cvaetf_path))
#         transfer_params(cvaetf, aug_cvaetf)
#     freeze_params(aug_cvaetf, train_names=['atten_mu', 'atten_logvar'])
#     return aug_cvaetf