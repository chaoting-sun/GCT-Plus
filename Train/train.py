import os
import gc
import numpy as np
from time import time

import torch
from torchtext import data

from Train.trainer import Trainer
from Model.modules import NoamOpt as moptim
from Utils import set_seed, allocate_gpu, get_fields, save_fields
from Model.build_model import build_model

"""
branch: mlp-only-training
"""

def train(args, debug=False):
    set_seed(51)
    torch.set_printoptions(profile="full")
    device = allocate_gpu()
    
    fields, SRC, TRG = get_fields(args.conditions, args.field_path)

    """ Preparing data """
    print('Preparing training/validation dataset')
    prepare_dataset_time = -time()
    train_data, valid_data = data.TabularDataset.splits(
        path=os.path.join(args.data_path, 'aug', f'data_sim{args.similarity:.2f}'),
        train='train.csv', validation='validation.csv', test=None,
        format='csv', fields=fields, skip_header=True)
    prepare_dataset_time += time()

    args.train_nbatches = int(np.ceil(len(train_data) / args.batch_size))
    args.valid_nbatches = int(np.ceil(len(valid_data) / args.batch_size))

    print(f'Pairs in Train/Validation Data: {len(train_data)}/{len(valid_data)}')
    print('Elipsed time (s):', prepare_dataset_time)

    print('Preparing training/validation dataloader')
    train_iter, valid_iter = data.BucketIterator.splits(
        (train_data, valid_data), batch_sizes=(args.batch_size, args.batch_size),
        sort_key=lambda x: (len(x.src), len(x.trg)))

    if args.load_field is False:
        SRC.build_vocab(train_data)
        TRG.build_vocab(valid_data)
        save_fields(SRC, TRG, args.field_path)

    # print("--- SRC VOCAB:", SRC.vocab.stoi)
    # print("--- TRG VOCAB:", TRG.vocab.stoi)

    args.sos_idx = TRG.vocab.stoi['<sos>']
    args.eos_idx = TRG.vocab.stoi['<eos>']
    args.pad_idx = SRC.vocab.stoi['<pad>']

    assert SRC.vocab.stoi['<pad>'] == TRG.vocab.stoi['<pad>']

    del train_data
    del valid_data
    gc.collect()

    """ Preparing Model """
    if args.start_epoch > 1:
        model_path = os.path.join(args.save_directory, f'model_{args.start_epoch-1}.pt')
    else:
        model_path = None
    model = build_model(args, len(SRC.vocab), len(TRG.vocab), model_path).to(device)
    
    # for n, p in model.named_parameters():
    #     if p.requires_grad:
    #         print(n, p.size())

    print('Parameters:', f'{sum(p.numel() for p in model.parameters()):<40}\t')
    print('Trainable Parameters:', f'{sum(p.numel() for p in model.parameters() if p.requires_grad):<40}')
    # exit()

    trainer = Trainer(args)
    trainer.train(model, train_iter, valid_iter, SRC, TRG, device)