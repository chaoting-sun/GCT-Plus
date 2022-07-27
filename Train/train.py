import os
import gc

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
    torch.set_printoptions(threshold=10_000)
    device = allocate_gpu()

    fields, SRC, TRG = get_fields(args.conditions, args.field_path)

    """ Preparing Model """
    train_data, valid_data = data.TabularDataset.splits(
        path=args.data_path, train='train.csv', validation='validation.csv',
        test=None, format='csv', fields=fields, skip_header=True)
    print(f'#Pairs in Training Data: {len(train_data):<40}')
    print(f'#Pairs in Validation Data: {len(valid_data):<40}')

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
    args.src_pad_idx = SRC.vocab.stoi['<pad>']
    args.trg_pad_idx = TRG.vocab.stoi['<pad>']

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


    print('Parameters:', f'{sum(p.numel() for p in model.parameters()):<40}\t')
    print('Trainable Parameters:', f'{sum(p.numel() for p in model.parameters() if p.requires_grad):<40}')
    
    """ Preparing Optimizer """
    trainable_parameters = filter(lambda p: p.requires_grad, model.parameters())
    if args.starting_epoch == 1:
        optimizer = torch.optim.Adam(trainable_parameters, lr=0,
                                     betas=(args.adam_beta1, args.adam_beta2), eps=args.adam_eps)
        optim = moptim(args.d_model, factor=args.factor,
                       warmup_steps=args.warmup_steps, optimizer=optimizer)
    else:
        optim_dict = torch.load(os.path.join(args.save_directory, f'model_{args.starting_epoch-1}.pt'),
                                map_location='cuda:0')['optimizer_state_dict']
        optim = moptim(optim_dict['model_size'], optim_dict['factor'], optim_dict['warmup'], 
                       torch.optim.Adam(trainable_parameters, lr=0))
        optim.load_state_dict(optim_dict)
    
    trainer = Trainer(args)
    trainer.train(model, train_iter, valid_iter, SRC, TRG, device)