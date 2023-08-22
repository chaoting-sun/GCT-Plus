import os
import torch
from collections import OrderedDict
from Inference.sampling_tool import sampling_tool_dict
from Model import Ctf, Vaetf, Cvaetf


model_dict = {
    'vaetf'      : Vaetf,
    'pvaetf'     : Cvaetf,
    'scavaetf'   : Cvaetf,
    'pscavaetf'  : Cvaetf,
    'ptf'        : Ctf,
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
    """freeze parameters of a model"""
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
        'nconds'      : len(args.property_list),
        'get_attn'    : args.get_attn
    }


def load_state(model, model_path, rank):
    # for distributed learning: https://www.zhihu.com/question/67209417/answer/866488638
    # map_location = { 'cuda:%d' % 0: 'cuda:%d' % rank }
    map_location = torch.device('cpu')
    try:
        model_state = torch.load(model_path, map_location)['model_state_dict']
    except KeyError:
        model_state = torch.load(model_path, map_location)
    except:
        exit(f'Cannot load model path: {model_path}')
    
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
    return model


def get_sampler(args, SRC, TRG, toklen_data, scaler, device):
    src_vocab_len = len(SRC.vocab)
    trg_vocab_len = len(TRG.vocab)

    args.model_path = os.path.join(args.model_folder, args.model_name)
    model = get_model(args, src_vocab_len, trg_vocab_len, device)
    model = model.to(device)
    model.eval()

    print(f'#parameters: {sum(p.numel() for p in model.parameters())}')

    kwargs = {
        'top_k'       : args.top_k,
        'latent_dim'  : args.latent_dim,
        'max_strlen'  : args.max_strlen,
        'use_cond2dec': args.use_cond2dec,
        'decode_algo' : args.decode_algo,
        'n_jobs'      : args.n_jobs,
        'toklen_data' : toklen_data,
        'cond_dim'    : len(args.property_list),
        'scaler'      : scaler,
        'device'      : device,
        'SRC'         : SRC,
        'TRG'         : TRG,
    }
    
    return sampling_tool_dict[args.model_type](model, kwargs)
