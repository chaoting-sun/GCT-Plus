import os

import torch

from Train.trainer import Trainer
from Utils import set_seed, allocate_gpu, get_dataset, get_iterator
from Model import transformer as mt
from Model import mlptransformer as mlpt
from Model.build_model import build_transformer, build_mlptransformer


def train(args, debug=False):
    set_seed(51)

    print('>>> GET DEVICE - GPU')
    device = allocate_gpu()

    print('>>> GET DATASET - TRAIN/VALIDATION DATASET & SRC/TRG')
    (train_data, valid_data), (SRC, TRG) = get_dataset(data_path=os.path.join(args.processed_path, 'data'), 
                                                       conditions=args.conditions, 
                                                       field_path=args.field_path,
                                                       load_field=args.load_field,
                                                       train='train.csv', 
                                                       validation='validation.csv',
                                                       test=None)
    args.src_pad_idx = SRC.vocab.stoi['<pad>']
    args.trg_pad_idx = SRC.vocab.stoi['<pad>']

    print('>>> GET ITERATOR - TRAINING DATASET')
    train_iter = get_iterator(train_data, 'train', args.batch_size, device)    

    print('>>> GET ITERATOR - VALIDATION DATASET')
    valid_iter = get_iterator(train_data, 'validation', args.batch_size, device)

    if args.train_stage == 1:
        print('>>> GET MODEL - TRAINING STAGE:', args.train_stage)
        tf_path = f'Experiment/checkpoint/model_{args.starting_epoch-1}.pt'
        model = mt.build_transformer(len(SRC.vocab), len(TRG.vocab), args.N, args.d_model, args.d_ff, 
                                     args.H, args.latent_dim, args.dropout, args.nconds, args.use_cond2dec,
                                     args.use_cond2lat, tf_path)
    
    elif args.train_stage == 2:
        if args.starting_epoch == 1:
            mlptf_path = None
        else:
            tf_path = f'Experiment/checkpoint/model_{args.starting_epoch-1}.pt'
        
        print('>>> GET MODEL - TRAINING STAGE:', args.train_stage)
        model = build_mlptransformer(len(SRC.vocab), len(TRG.vocab), args.N, args.d_model, args.d_ff, 
                                     args.H, args.latent_dim, args.dropout, args.nconds, args.use_cond2dec,
                                     args.use_cond2lat, args.variational, args.transferring_model_path, mlptf_path)

    
    print(">>> TOTAL PARAMETERS:", sum(p.numel() for p in model.parameters()))
    print(">>> TRAINABLE PARAMTERS:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    if debug is True:
        torch.set_printoptions(threshold=10_000)
    
    trainer = Trainer(args)
    trainer.train(model, train_iter, valid_iter, SRC, TRG, device)
