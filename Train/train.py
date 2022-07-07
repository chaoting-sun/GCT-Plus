import os
import gc

import torch
from torchtext import data

from Train.trainer import Trainer
from Utils import set_seed, allocate_gpu, get_fields, save_fields
from Model.build_model import build_transformer, build_mlptransformer, build_mlpencoder


def train(args, debug=False):
    set_seed(51)
    torch.set_printoptions(threshold=10_000)

    print('>>> ALLOCATING DEVCIE - GPU')
    device = allocate_gpu()

    print('>>> PREPARING FIELDS - COND/SRC/TRG')
    fields = get_fields(args.conditions, args.field_path)
    field_dict = {p: f for p, f in fields}
    SRC = field_dict['src']
    TRG = field_dict['trg']

    print('>>> PREPARING DATA - TRAINING/VALIDATION SET')
    train_data, valid_data = data.TabularDataset.splits(
        path=args.data_path, train='train.csv', validation='validation.csv',
        test=None, format='csv', fields=fields, skip_header=True)

    train_iter, valid_iter = data.BucketIterator.splits(
        (train_data, valid_data), batch_sizes=(args.batch_size, args.batch_size),
        sort_key=lambda x: (len(x.src), len(x.trg)))

    if args.load_field is False:
        SRC.build_vocab(train_data)
        TRG.build_vocab(valid_data)
        save_fields(SRC, TRG, args.field_path)

    print("--- SRC VOCAB:", SRC.vocab.stoi)
    print("--- TRG VOCAB:", TRG.vocab.stoi)

    args.sos_idx = TRG.vocab.stoi['<sos>']
    args.eos_idx = TRG.vocab.stoi['<eos>']
    args.src_pad_idx = SRC.vocab.stoi['<pad>']
    args.trg_pad_idx = TRG.vocab.stoi['<pad>']

    assert SRC.vocab.stoi['<pad>'] == TRG.vocab.stoi['<pad>']

    del train_data
    del valid_data
    gc.collect()

    print('>>> PREPARING MODEL - TRAINING STAGE', args.train_stage)
    if args.train_stage == 1:
        tf_path = os.path.join(args.save_directory, 'checkpoint', f'model_{args.starting_epoch-1}.pt')
        model = build_transformer(len(SRC.vocab), len(TRG.vocab), args.N, args.d_model, args.d_ff, 
                                  args.H, args.latent_dim, args.dropout, args.nconds, args.use_cond2dec,
                                  args.use_cond2lat, tf_path)
    
    elif args.train_stage == 2:
        if args.starting_epoch == 1:
            mlptf_path = None
        else:
            mlptf_path = os.path.join(args.save_directory, f'model_{args.starting_epoch-1}.pt')
        model = build_mlpencoder(len(SRC.vocab), len(TRG.vocab), args.N, args.d_model, args.d_ff, 
                                 args.H, args.latent_dim, args.dropout, args.nconds, args.use_cond2dec,
                                 args.use_cond2lat, args.variational, args.transferring_model_path, mlptf_path)
    model = model.to(device)        

    print("--- total parameters:", sum(p.numel() for p in model.parameters()))
    print("--- trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    trainer = Trainer(args)
    trainer.train(model, train_iter, valid_iter, SRC, TRG, device)