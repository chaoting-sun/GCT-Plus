import os
import gc
import joblib
import pandas as pd
import numpy as np

import torch
from torchtext import data

from Train.mlp_trainer import Trainer
from Utils import set_seed, allocate_gpu, get_fields, save_fields
from Model.build_model import build_model

from Utils.dataset import mlpDataset
from torch.utils.data import DataLoader

"""
branch: mlp-only-training
"""

def mlp_train(args, debug=False):
    set_seed(51)
    torch.set_printoptions(profile="full")
    device = allocate_gpu()

    scaler = joblib.load(args.scaler_path)

    """ training data """

    train_raw_folder = os.path.join(args.data_path, 'raw', 'train')
    train_aug_folder = os.path.join(args.data_path, 'aug', 'train')

    if not os.path.exists(os.path.join(train_raw_folder, 'prop_serial_tf.csv')):
        prop_data = pd.read_csv(os.path.join(train_raw_folder, 'prop_serial.csv'))
        prop_data = prop_data.set_index('no').T.to_dict('list')
        for key, value in prop_data.items():
            prop_data[key] = scaler.transform(np.array([value])).tolist()[0]
            print(key)
        prop_data = pd.DataFrame.from_dict(prop_data, orient='index', columns=args.conditions)
        prop_data['no'] = prop_data.index
        prop_data.to_csv(os.path.join(train_raw_folder, 'prop_serial_tf.csv'), index=False)

    dataset = mlpDataset(conditions=args.conditions,
                         tensor_folder=os.path.join(train_raw_folder, 'tensor'),
                         pair_path=os.path.join(train_aug_folder, f'pair_serial_{args.similarity:.2f}.csv'),
                         prop_path=os.path.join(train_raw_folder, 'prop_serial_tf.csv'),
                         device=device)

    train_dl = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    """ validation data """

    valid_raw_folder = os.path.join(args.data_path, 'raw', 'validation')
    valid_aug_folder = os.path.join(args.data_path, 'aug', 'validation')

    if not os.path.exists(os.path.join(valid_raw_folder, 'prop_serial_tf.csv')):
        prop_data = pd.read_csv(os.path.join(valid_raw_folder, 'prop_serial.csv'))
        prop_data = prop_data.set_index('no').T.to_dict('list')
        for key, value in prop_data.items():
            prop_data[key] = scaler.transform(np.array([value])).tolist()[0]
            print(key)
        prop_data = pd.DataFrame.from_dict(prop_data, orient='index', columns=args.conditions)
        prop_data['no'] = prop_data.index
        prop_data.to_csv(os.path.join(valid_raw_folder, 'prop_serial_tf.csv'), index=False)

    dataset = mlpDataset(conditions=args.conditions,
                         tensor_folder=os.path.join(valid_raw_folder, 'tensor'),
                         pair_path=os.path.join(valid_aug_folder, f'pair_serial_{args.similarity:.2f}.csv'),
                         prop_path=os.path.join(valid_raw_folder, 'prop_serial_tf.csv'),
                         device=device)

    valid_dl = DataLoader(dataset, batch_size=args.batch_size)


    # for i, batch in enumerate(dataloader):
    #     print(batch['src'].size(), batch['trg'].size(), batch['mconds'].size())
    #     print(batch['src'])
    #     print(batch['trg'])
    #     print(batch['mconds'])
    #     break

    fields, SRC, TRG = get_fields(args.conditions, args.field_path)

    args.sos_idx = TRG.vocab.stoi['<sos>']
    args.eos_idx = TRG.vocab.stoi['<eos>']
    args.src_pad_idx = SRC.vocab.stoi['<pad>']
    args.trg_pad_idx = TRG.vocab.stoi['<pad>']

    assert SRC.vocab.stoi['<pad>'] == TRG.vocab.stoi['<pad>']

    """ Preparing Model """
    if args.start_epoch > 1:
        model_path = os.path.join(args.save_directory, f'model_{args.start_epoch-1}.pt')
    else:
        model_path = None
    model = build_model(args, len(SRC.vocab), len(TRG.vocab), model_path).to(device)

    print('Parameters:', f'{sum(p.numel() for p in model.parameters()):<40}\t')
    print('Trainable Parameters:', f'{sum(p.numel() for p in model.parameters() if p.requires_grad):<40}')
    
    trainer = Trainer(args)
    trainer.train(model, train_dl, valid_dl, SRC, TRG)