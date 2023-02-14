import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchtext import data
from pathos.multiprocessing import ProcessingPool as Pool

from Utils.seed import set_seed
from Utils.field import get_tf_fields, save_fields
from Model.build_model import get_model
from Inference.utils import prepare_generator
from Utils.dataset import get_loader

# prop_constraints = {
#     'logP': [ 0.03,   4.97],
#     'tPSA': [17.92, 112.83],
#     'QED' : [ 0.58,   0.95]
# }


# target_props = np.array(np.meshgrid(
#     np.linspace(prop_constraints['logP'][0], prop_constraints['logP'][1], num=20),
#     np.linspace(prop_constraints['tPSA'][0], prop_constraints['tPSA'][1], num=20),
#     np.linspace(prop_constraints['QED'][0], prop_constraints['QED'][1], num=20))) \
#     .T.reshape(-1, 3)


# prop_intervals = [
#     (prop_constraints['logP'][1]-prop_constraints['logP'][0])/4,
#     (prop_constraints['tPSA'][1]-prop_constraints['tPSA'][0])/4,
#     (prop_constraints['QED'][1]-prop_constraints['QED'][0])/4,
# ]

# train = pd.read_csv('/fileserver-gamma/chaoting/ML/dataset/moses/raw/train/prop_serial.csv')
# p1_range, p2_range, p3_range = prop_intervals[0]/2, prop_intervals[1]/2, prop_intervals[2]/2    
# n_trains = np.zeros((len(target_props),), dtype=np.int32)

# for i, (c_logP, c_tPSA, c_QED) in enumerate(target_props):
#     f = train.loc[(c_logP-p1_range <= train.logP) & (train.logP <= c_logP+p1_range) &
#                     (c_tPSA-p2_range <= train.tPSA) & (train.tPSA <= c_tPSA+p2_range) &
#                     (c_QED-p3_range <= train.QED) & (train.QED <= c_QED+p3_range)]
#     n_trains[i] = len(f)


# def n_trains_by_props(props):
#     logP, tPSA, QED = props
#     def in_bound(lb, ub, val):
#         return lb <= val <= ub
#     for i, (logP_t, tPSA_t, QED_t) in enumerate(target_props):
#         if in_bound(logP_t-p1_range, logP_t+p1_range, logP) and \
#            in_bound(tPSA_t-p2_range, tPSA_t+p2_range, tPSA) and \
#            in_bound(QED_t-p3_range, QED_t+p3_range, QED):
#             return n_trains[i]


class Batch:
    def __init__(self, src, trg, trg_en=None,
                 econds=None, dconds=None, mconds=None):
        self.src = src
        self.trg_y = trg[:, 1:]
        self.trg = trg[:, :-1]

        if trg_en:
            self.trg_en = trg_en

        self.trg_en = None if trg_en is None else trg_en
        self.econds = None if econds is None else econds
        self.dconds = None if dconds is None else dconds
        self.mconds = None if mconds is None else mconds


def rebatch(batch, conds):
    batch_data = {}
    batch_data['src'] = batch.src.transpose(0, 1)
    batch_data['trg'] = batch.trg.transpose(0, 1)

    if len(conds) > 0:
        econds, dconds = [], []
        for c in conds:
            econds.append(getattr(batch, f"src_{c}").view(-1, 1))
            dconds.append(getattr(batch, f"trg_{c}").view(-1, 1))
        batch_data['econds'] = torch.cat(econds, dim=1)
        batch_data['dconds'] = torch.cat(dconds, dim=1)
        delconds = batch_data['dconds'] - batch_data['econds']
        batch_data['mconds'] = torch.cat([batch_data['econds'], delconds], dim=1)

    return Batch(**batch_data)


def rebatch_pad(batch, conds, pad_id, max_strlen):
    def padding(obj, max_strlen, cond_len):
        obj_pad = torch.ones(obj.size(0), abs(max_strlen - obj.size(1)
                           - cond_len), dtype=torch.long) * pad_id
        return torch.cat([obj, obj_pad], dim=1)
    
    batch_data = {}
    batch_data['src'] = padding(batch.src.transpose(0, 1), max_strlen, len(conds))
    batch_data['trg'] = padding(batch.trg.transpose(0, 1), max_strlen, len(conds))

    if len(conds) > 0:
        econds, dconds = [], []
        for c in conds:
            econds.append(getattr(batch, f"src_{c}").view(-1, 1))
            dconds.append(getattr(batch, f"trg_{c}").view(-1, 1))
        batch_data['econds'] = torch.cat(econds, dim=1)
        batch_data['dconds'] = torch.cat(dconds, dim=1)
        delconds = batch_data['dconds'] - batch_data['econds']
        batch_data['mconds'] = torch.cat([batch_data['econds'], delconds], dim=1)

    return Batch(**batch_data)


def get_dataloader(data_iter, conds):
    return (rebatch(batch, conds) for batch in data_iter)


def get_pad_dataloader(data_iter, conds, pad_id, max_strlen):
    return (rebatch_pad(batch, conds, pad_id, max_strlen) for batch in data_iter)


def tensor_to_smiles(FIELD):
    def convert(tensor):
        smiles = ""
        for i in tensor:
            s = FIELD.vocab.itos[i]
            if s == "<pad>":
                break
            if s == "<eos>":
                break
            smiles += s
        return smiles
    return convert

# def smiles2tensor(tensor, FIELD):
#     smiles = ""
#     for i in tensor:
#         s = FIELD.vocab.itos[i]
#         if s == "<pad>":
#             return
#         smiles += s
#     return smiles


def smiles_converter(x, FIELD, n_jobs=1, reverse=True):
    converter = tensor_to_smiles(FIELD)
    if reverse:
        with Pool(n_jobs) as pool:
            smiles_list = pool.map(converter, x)
        return smiles_list
    return None


def prepare_dataiter(data_path, fields, batch_size=1, n_samples=1000): # 256, 10000
    def sample_data(data_type, n):
        df = pd.read_csv(os.path.join(data_path, f'{data_type}.csv'))
        df_same = df.loc[df.src == df.trg]
        df_not_same = df.loc[df.src != df.trg]
        
        df_same = df_same.sample(n, random_state=1, ignore_index=True)
        if len(df_not_same) != 0:
            df_not_same = df_not_same.sample(n, random_state=1, ignore_index=True)
            df = pd.concat([df_same, df_not_same], axis=0)
        else:
            df = df_same
        df.to_csv(os.path.join(data_path, f'{data_type}_sample.csv'), index=False)

    sample_data('train', n_samples)
    sample_data('validation', n_samples)

    train, valid = data.TabularDataset.splits(
        path=data_path, train='train_sample.csv', validation='validation_sample.csv',
        test=None, format='csv', fields=fields, skip_header=True
    )

    train_iter, valid_iter = data.BucketIterator.splits(
        (train, valid), (batch_size, batch_size), shuffle=True,
        sort_key=lambda x: (len(x.src), len(x.trg))
    )
    return train_iter, valid_iter


def src_generation(generator, SRC, TRG, batch, n_samples=1, n_jobs=1):
    smi_in = smiles_converter(batch.src, SRC, n_jobs)
    smi_trg = smiles_converter(batch.trg_y, TRG, n_jobs)
    smi_out = [[] for _ in range(len(smi_in))]

    for _ in range(n_samples):
        z, mu, logvar = generator.encode_smiles(batch.src,
                                                batch.econds,
                                                transform=False)
        mu = mu[:, :35, :]
        smi_gen, *_ = generator.sample_smiles(batch.dconds, mu, transform=False)
        for i, smi in enumerate(smi_gen):
            smi_out[i].append(smi)
    return smi_in, smi_trg, smi_out

def compute_accuracy(dataloader, generator, SRC, TRG, scaler, n_samples=1, n_jobs=1):
    n_data = n_same = n_not_same = 0
    n_total_acc = n_same_acc = n_not_same_acc = 0

    # for i, batch in enumerate(tqdm(dataloader)):
    for i, batch in enumerate(dataloader):
        smi_in, smi_trg, smi_out = src_generation(generator, SRC, TRG, batch, n_samples, n_jobs)
        n_data += len(smi_in)
        print(i, smi_in, smi_trg, smi_out)
        # dconds_truth = scaler.inverse_transform(batch.dconds)
        
        for j, (st, so_list) in enumerate(zip(smi_trg, smi_out)):
            for k, so in enumerate(so_list):
                if smi_in[j] == st: # src = trg
                    n_same += 1
                    if st == so: # trg = gen                        
                        n_total_acc += 1
                        n_same_acc += 1
                        break        
                else:
                    n_not_same += 1
                    if st == so:
                        n_total_acc += 1
                        n_not_same_acc += 1
                        break
        print(n_same_acc, n_same, n_not_same_acc, n_not_same)
        if (i+1) % 5 == 0:
            if n_same:
                print(f'% accuracy (same): {n_same_acc/n_same*100:.3f}%\t')
            if n_not_same:
                print(f'% accuracy (not same): {n_not_same_acc/n_not_same*100:.3f}%')
    return n_total_acc, n_same_acc, n_not_same_acc, n_data, n_same, n_not_same


def reconstruction(args, toklen_data, scaler, device, logger, n_samples=1):
    set_seed(0)
    
    save_folder = os.path.join(args.train_path, args.model_name, 'reconstruction')
    os.makedirs(save_folder, exist_ok=True)

    LOG = logger(name='augment data by conditions',
                 log_path=os.path.join(save_folder, "records.log"))
    
    fields, SRC, TRG = get_tf_fields(args.conditions, args.molgct_path)
    args.pad_id = SRC.vocab.stoi['<pad>']
    
    data_path = os.path.join(args.data_path, 'aug',
                             f'data_sim{args.similarity:.2f}_tol{args.tolerance:.2f}')
    train_iter, valid_iter = prepare_dataiter(data_path, fields)

    acc_dict = {
        'n_train': [],
        'n_train_same': [],
        'n_train_not_same': [],
        'n_train_acc': [],
        'n_train_same_acc': [],
        'n_train_not_same_acc': [],
        'n_valid': [],
        'n_valid_same': [],
        'n_valid_not_same': [],
        'n_valid_acc': [],
        'n_valid_same_acc': [],
        'n_valid_not_same_acc': [],
    }

    with torch.no_grad():
        for epoch in args.epoch_list:
            LOG.info(f"model epoch: {epoch}")
            args.use_model_path = os.path.join(args.train_path,
                                               args.model_name,
                                               f'model_{epoch}.pt')

            generator = prepare_generator(args, SRC, TRG, toklen_data, scaler, device)

            dataloader = get_loader(train_iter,
                                    args.conditions,
                                    args.pad_id,
                                    args.max_strlen,
                                    args.pad_to_same_len)
            n_acc, n_same_acc, n_not_same_acc, n_data, n_same, n_not_same = compute_accuracy(
                dataloader, generator, SRC, TRG, scaler, n_samples, args.n_jobs)
            dataloader = get_loader(valid_iter,
                                    args.conditions,
                                    args.pad_id,
                                    args.max_strlen,
                                    args.pad_to_same_len)

            acc_dict['n_train'].append(n_data)
            acc_dict['n_train_same'].append(n_same)
            acc_dict['n_train_not_same'].append(n_not_same)
            acc_dict['n_train_acc'].append(n_acc)
            acc_dict['n_train_same_acc'].append(n_same_acc)
            acc_dict['n_train_not_same_acc'].append(n_not_same_acc)

            n_acc, n_same_acc, n_not_same_acc, n_data, n_same, n_not_same = compute_accuracy(
                dataloader, generator, SRC, TRG, scaler, n_samples, args.n_jobs)

            acc_dict['n_valid'].append(n_data)
            acc_dict['n_valid_same'].append(n_same)
            acc_dict['n_valid_not_same'].append(n_not_same)
            acc_dict['n_valid_acc'].append(n_acc)
            acc_dict['n_valid_same_acc'].append(n_same_acc)
            acc_dict['n_valid_not_same_acc'].append(n_not_same_acc)

            print('Accuracy:', acc_dict)

    model_acc = pd.DataFrame(data=acc_dict, index=args.epoch_list)
    model_acc.to_csv(os.path.join(save_folder, "rec_accuracy.csv"))
