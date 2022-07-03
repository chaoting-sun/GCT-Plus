import os
import gc

import torch
from torchtext import data

from Train.trainer import Trainer
from Utils import set_seed, allocate_gpu, get_dataset, get_iterator, get_fields, save_fields
from Model import transformer as mt
from Model import mlptransformer as mlpt
from Model.build_model import build_transformer, build_mlptransformer
from Utils.dataset import to_dataloader

def train(args, debug=False):
    set_seed(51)
    torch.set_printoptions(threshold=10_000)

    print('>>> GET DEVICE - GPU')
    device = allocate_gpu()

    print('>>> GET DATASET - TRAIN/VALIDATION DATASET & SRC/TRG')

    fields = get_fields(args.conditions, args.field_path)
    print('>>> FIELDS:', fields)
    field_dict = {p: f for p, f in fields}
    
    train_data, valid_data = data.TabularDataset.splits(
        path=args.data_path, train='train.csv', validation='validation.csv',
        test=None, format='csv', fields=fields, skip_header=True)

    SRC = field_dict['src']
    TRG = field_dict['trg']

    # the first data should be training data
    train_iter, valid_iter = data.BucketIterator.splits(
        (train_data, valid_data), batch_sizes=(args.batch_size, args.batch_size),
        sort_key=lambda x: (len(x.src), len(x.trg)))

    # train_dataloader = to_dataloader(train_iter, args.conditions)
    # for i, b in enumerate(train_dataloader):
    #     print(b.src)    

    if args.load_field is False:
        SRC.build_vocab(train_data)
        TRG.build_vocab(valid_data)
        save_fields(SRC, TRG, args.field_path)

    print(">>> SRC VOCAB:", SRC.vocab.stoi)
    print(">>> TRG VOCAB:", TRG.vocab.stoi)

    args.sos_idx = TRG.vocab.stoi['<sos>']
    args.eos_idx = TRG.vocab.stoi['<eos>']
    args.src_pad_idx = SRC.vocab.stoi['<pad>']
    args.trg_pad_idx = TRG.vocab.stoi['<pad>']

    assert SRC.vocab.stoi['<pad>'] == TRG.vocab.stoi['<pad>']

    # print('>>> GET ITERATOR - TRAINING DATASET')
    # train_iter = get_iterator(train_data, 'train', args.batch_size, device)

    # print('>>> GET ITERATOR - VALIDATION DATASET')
    # valid_iter = get_iterator(valid_data, 'validation', args.batch_size, device)
        
    del train_data
    del valid_data
    gc.collect()

    if args.train_stage == 1:
        print('>>> GET MODEL - TRAINING STAGE:', args.train_stage)
        tf_path = os.path.join(args.save_directory, 'checkpoint', f'model_{args.starting_epoch-1}.pt')
        model = mt.build_transformer(len(SRC.vocab), len(TRG.vocab), args.N, args.d_model, args.d_ff, 
                                     args.H, args.latent_dim, args.dropout, args.nconds, args.use_cond2dec,
                                     args.use_cond2lat, tf_path)
    
    elif args.train_stage == 2:
        if args.starting_epoch == 1:
            mlptf_path = None
        else:
            mlptf_path = os.path.join(args.save_directory, f'model_{args.starting_epoch-1}.pt')
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
