import os
import torch
import argparse
from Configuration.config import options
from Utils import get_dataset, get_fields
from Model.build_model import build_mlptransformer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    options(parser)
    args = parser.parse_args()
    
    args.load_field = True
    args.transferring_model_path = 'molGCT/molgct.pt'

    fields = get_fields(args.conditions, args.field_path)
    field_dict = {p: f for p, f in fields}

    # train_data, valid_data = get_dataset(data_path=os.path.join(args.data_path, 'aug', 'data'), 
    #                                      conditions=args.conditions, 
    #                                      fields=fields,
    #                                      train='train.csv', 
    #                                      validation='validation.csv',
    #                                      test=None)
    print('- get src/trg')

    SRC = field_dict['src']
    TRG = field_dict['trg']

    mlptf_path = 'Experiment/stage2_train/model_1.pt'
    model1 = build_mlptransformer(len(SRC.vocab), len(TRG.vocab), args.N, args.d_model, args.d_ff,
                                    args.H, args.latent_dim, args.dropout, args.nconds, args.use_cond2dec,
                                    args.use_cond2lat, args.variational, args.transferring_model_path, mlptf_path)
    exit()
    for n, p in model1.named_parameters():
        n1 = n.split('.')
        print(n, p)
        if n1[0] == 'mlp':
            print(n, p)
            exit()
            

    exit(0)

    # for name, params in model1.named_parameters():
    #     if params.requires_grad:
    #         print(name, '-> yes')
    #     else:
    #         print(name, '-> no')
    
    state1 = model1.state_dict()
    
    print('- get model')
    mlptf_path = 'Experiment/stage2_train/model_2.pt'
    model2 = build_mlptransformer(len(SRC.vocab), len(TRG.vocab), args.N, args.d_model, args.d_ff, 
                                    args.H, args.latent_dim, args.dropout, args.nconds, args.use_cond2dec,
                                    args.use_cond2lat, args.variational, args.transferring_model_path, mlptf_path)
    state2 = model2.state_dict()
    
    for name, params in state1.items():
        change = False in (state1[name] == state2[name])
        if change:
            print(name, params.size(), '-> yes')
        else:
            print(name, params.size(), '-> no')
    
    