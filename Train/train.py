import os
import gc

from numpy import ceil
from time import time

import torch
from torchtext import data

from Train.mlpcvae_trainer import MLPCVAE_Trainer
from Train.att_trainer import ATT_Trainer
from Model.modules import NoamOpt as moptim
from Utils import set_seed, allocate_gpu, get_fields, save_fields
from Model.build_model import build_model


def train(args, debug=False):
    set_seed(100)

    print('Getting GPU')
    device = allocate_gpu()    

    print('Getting feilds / SRC / TRG')

    fields, SRC, TRG = get_fields(args.conditions, args.molgct_path)

    print('Preparing training / validation dataset')

    train_data, valid_data = data.TabularDataset.splits(
        path=os.path.join(args.data_path, 'aug', f'data_sim{args.similarity:.2f}'),
        train='train.csv', validation='validation.csv', test=None, format='csv',
        fields=fields, skip_header=True
    )

    print(f'#pairs in training / validation dataset: {len(train_data)}/{len(valid_data)}')

    if args.load_field is False:
        SRC.build_vocab(train_data)
        TRG.build_vocab(valid_data)
        save_fields(SRC, TRG, args.molgct_path)

    args.train_nbatches, args.valid_nbatches = int(ceil(len(train_data) / args.batch_size)), \
                                               int(ceil(len(valid_data) / args.batch_size))

    print('Preparing training / validation dataloader')

    train_iter, valid_iter = data.BucketIterator.splits(
        (train_data, valid_data), batch_sizes=(args.batch_size, args.batch_size),
        sort_key=lambda x: (len(x.src), len(x.trg))
    )

    del train_data, valid_data
    gc.collect()

    args.sos_idx = TRG.vocab.stoi['<sos>']
    args.eos_idx = TRG.vocab.stoi['<eos>']
    args.pad_idx = SRC.vocab.stoi['<pad>']

    assert SRC.vocab.stoi['<pad>'] == TRG.vocab.stoi['<pad>']

    print(f'Preparing model with starting epoch: {args.start_epoch}')

    if args.model_type == 'att_encoder':
        model = build_model(args, len(SRC.vocab), len(TRG.vocab), att_type='ATT_v5').to(device)
    elif args.model_type == 'mlp_encoder':
        model = build_model(args, len(SRC.vocab), len(TRG.vocab)).to(device)

    # for n, p in model.named_parameters():
    #     if p.requires_grad:
    #         print(n, p.size())

    total_parameters = sum(p.numel() for p in model.parameters())
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert trainable_parameters > 0

    print('Parameters / Trainable Parameters:', f'{total_parameters:<11}/{trainable_parameters:<11}\t')
    
    if args.model_type == 'att_encoder':
        trainer = ATT_Trainer(args.conditions, args.save_directory, args.pad_idx, args.max_strlen)
        trainer.train(args, model, train_iter, valid_iter, SRC, TRG, device)
    elif args.model_type == 'mlp_encoder':
        trainer = MLPCVAE_Trainer(args)
        trainer.train(model, train_iter, valid_iter, SRC, TRG, device)