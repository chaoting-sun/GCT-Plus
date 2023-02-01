import os
import pandas as pd
import torch
import numpy as np
import dill as pickle
from Inference.utils import prepare_generator
from torchtext.data import Example, Dataset
from pathos.multiprocessing import ProcessingPool as Pool
from Utils.property import tanimoto_similarity as similarity_fcn
from sklearn.utils import shuffle


def distance_fcn(z1, z2):
    return torch.sqrt(torch.sum((z2 - z1)**2)).item()
 

def get_toklen(SRC, smiles):
    return len(SRC.tokenize(smiles))


class DataFrameDataset(Dataset):
    def __init__(self, df: pd.DataFrame, fields: list):
        super(DataFrameDataset, self).__init__([
                Example.fromlist(list(r), fields)
                for i, r in df.iterrows()
            ],
            fields
        )


def get_fields(SRC, COND, conditions):
    def get_one_set_fields(name):
        fields = [(name, SRC)]
        for i, c in enumerate(conditions):
            fields.append((f'{name}_{c}', COND[i]))
        return fields
    field_list = []
    field_list.extend(get_one_set_fields('src'))
    field_list.extend(get_one_set_fields('trg'))
    return field_list


def prepare_data(data_type):
    smiles = pd.read_csv(f'/fileserver-gamma/chaoting/ML/dataset/moses/raw/{data_type}/smiles_serial.csv')
    props = pd.read_csv(f'/fileserver-gamma/chaoting/ML/dataset/moses/raw/{data_type}/prop_serial.csv')
    # smiles_props = pd.concat([smiles, props], axis=1)
    smiles_props = smiles.merge(props, how='inner')
    return smiles_props

 
def sample_high_similarity_pairs(each_iterval_cnt, SRC):
    smiles = pd.read_csv("/fileserver-gamma/chaoting/ML/dataset/moses/aug/data_sim0.50_tol0.30/train.csv")
    smiles = shuffle(smiles).reset_index(drop=True)

    current_cnt = 0
    start_sim, end_sim, interval = 0.5, 1.0, 0.1
    sim_interval = np.arange(start_sim, end_sim+interval-(10E-3), interval)

    total_cnt = each_iterval_cnt * (len(sim_interval)-1)
    pair_ids = [set() for _ in range(len(sim_interval)-1)]
    current_id = 0

    while current_cnt < total_cnt:
        no1 = smiles['src_no'].iloc[current_id]
        no2 = smiles['trg_no'].iloc[current_id]
        smi1 = smiles['src'].iloc[current_id]
        smi2 = smiles['trg'].iloc[current_id]

        if get_toklen(SRC, smi1) != get_toklen(SRC, smi2):
            current_id += 1
            continue

        if no1 == no2:
            current_id += 1
            continue
        pair_no = (no1, no2) if no1 < no2 else (no2, no1)

        sim = similarity_fcn(smi1, smi2)

        for i in range(len(sim_interval)-1):
            if sim_interval[i] < sim < sim_interval[i+1] and len(pair_ids[i]) < each_iterval_cnt:
                print(f'progress: {current_cnt} / {total_cnt}', sep='\r')
                pair_ids[i].update([pair_no])
                current_cnt += 1
                break
        current_id += 1
    return pair_ids


def sample_low_similarity_pairs(each_iterval_cnt, SRC):
    smiles = pd.read_csv("/fileserver-gamma/chaoting/ML/dataset/moses/raw/train/smiles_serial.csv")
    smiles = shuffle(smiles).reset_index(drop=True)

    current_cnt = 0
    start_sim, end_sim, interval = 0, 0.5, 0.1
    sim_interval = np.arange(start_sim, end_sim+interval, interval)

    total_cnt = each_iterval_cnt * (len(sim_interval)-1)
    pair_ids = [set() for _ in range(len(sim_interval)-1)]
    candidate_ids = np.arange(len(smiles))
    current_id = 0

    while current_cnt < total_cnt:
        pair = np.random.choice(candidate_ids, 2)
        no1 = smiles['no'].iloc[pair[0]]
        no2 = smiles['no'].iloc[pair[1]]
        smi1 = smiles['smiles'].iloc[pair[0]]
        smi2 = smiles['smiles'].iloc[pair[1]]

        if get_toklen(SRC, smi1) != get_toklen(SRC, smi2):
            continue

        pair_no = (no1, no2) if no1 < no2 else (no2, no1)
        
        sim = similarity_fcn(smi1, smi2)
        for i in range(len(sim_interval)-1):
            if sim_interval[i] < sim < sim_interval[i+1] and len(pair_ids[i]) < each_iterval_cnt:
                print(f'progress: {current_cnt} / {total_cnt}', sep='\r')
                pair_ids[i].update([pair_no])
                current_cnt += 1
                break
        current_id += 1
    return pair_ids


def fast_test_encoder(args, toklen_data, scaler, SRC, TRG, COND, device):
    # TODO: the same SMILES with different padding

    each_iterval_cnt = 50
    low_sim_pairs = sample_low_similarity_pairs(each_iterval_cnt, SRC)
    high_sim_pairs = sample_high_similarity_pairs(each_iterval_cnt, SRC)
    
    all_pairs = low_sim_pairs + high_sim_pairs

    smiles = pd.read_csv("/fileserver-gamma/chaoting/ML/dataset/moses/raw/train/smiles_serial.csv")
    props = pd.read_csv(f'/fileserver-gamma/chaoting/ML/dataset/moses/raw/train/prop_serial.csv')
    smiles_props = smiles.merge(props, how='inner')
    smiles_props = smiles_props.set_index('no')

    data_inputs = None
    for i in range(len(all_pairs)):
        src_no, trg_no = list(zip(*all_pairs[i]))
        src = smiles_props.loc[src_no, ['smiles']+args.conditions]
        trg = smiles_props.loc[trg_no, ['smiles']+args.conditions]
        src = src.rename(columns={ 'smiles': 'src',
                                   'logP'  : 'src_logP',
                                   'tPSA'  : 'src_tPSA',
                                   'QED'   : 'src_QED' })
        trg = trg.rename(columns={ 'smiles': 'trg',
                                   'logP'  : 'trg_logP',
                                   'tPSA'  : 'trg_tPSA',
                                   'QED'   : 'trg_QED' })
        src = src.reset_index(drop=True)
        trg = trg.reset_index(drop=True)
        pair_inputs = pd.concat([src, trg], axis=1)
        
        data_inputs = pd.concat([data_inputs, pair_inputs], axis=0)
    data_inputs = data_inputs.reset_index(drop=True)

    fields = get_fields(SRC, COND, args.conditions)
    dataset = DataFrameDataset(df=data_inputs, fields=fields)

    args.use_model_path = os.path.join(args.train_path, args.model_name,
                                       f'model_{args.epoch_list[0]}.pt')
    generator = prepare_generator(args, SRC, TRG, toklen_data, scaler, device)

    def rewrap_input(name, data):
        smiles = torch.LongTensor([[SRC.vocab.stoi[t] for t in
                                    getattr(data, name)]])
        props = np.zeros((1,3))
        for j, c in enumerate(args.conditions):
            props[0, j] = getattr(data, f'{name}_{c}')
        return smiles, props

    similarity_list = np.zeros((len(dataset),))
    distance_list = np.zeros((len(dataset),))

    max_length = 80

    for i, data in enumerate(dataset):
        src, src_props = rewrap_input('src', data)
        trg, trg_props = rewrap_input('trg', data)
        
        # src_pad = torch.zeros((1,abs(max_length-src.size(1))), dtype=torch.long)
        # trg_pad = torch.zeros((1,abs(max_length-trg.size(1))), dtype=torch.long)
        
        # src = torch.concat([src, src_pad], axis=1)
        # trg = torch.concat([trg, trg_pad], axis=1)

        _, src_mu, _ = generator.encode_smiles(src, src_props)
        _, trg_mu, _ = generator.encode_smiles(trg, trg_props)

        src_pad = torch.zeros((1,abs(max_length-src_mu.size(1)), src_mu.size(2)), dtype=torch.long)
        trg_pad = torch.zeros((1,abs(max_length-trg_mu.size(1)), src_mu.size(2)), dtype=torch.long)
        
        src_mu = torch.concat([src_mu, src_pad.cuda()], axis=1)
        trg_mu = torch.concat([trg_mu, trg_pad.cuda()], axis=1)

        similarity_list[i] = similarity_fcn(data_inputs['src'].loc[i],
                                            data_inputs['trg'].loc[i])
        distance_list[i] = distance_fcn(src_mu, trg_mu) / max_length
 
    df = pd.DataFrame({ 'similarity': similarity_list, 'distance': distance_list })
    df['distance'] = df['distance'] / df['distance'].max()
    df.to_csv('./3.csv')