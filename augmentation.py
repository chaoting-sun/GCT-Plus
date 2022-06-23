import os
import sys
import pandas as pd
from functools import partial
from multiprocessing import Pool
import time
from datetime import timedelta


sys.path.append('/home/chaoting/tools/rdKit/similarity/')
from Utils.property import tanimoto_similarity
# sys.path.append('/fileserver-gamma/chaoting/ML/data/moses/')

import moses

# similarity_bound = 0.60
# n_jobs = 12
# n_samples = 5000

# property_path ='/fileserver-gamma/chaoting/ML/data/moses/'
# out_property_path ='/fileserver-gamma/chaoting/ML/data/moses_aug/'
# inname = 'prop_temp.csv'
# outname = 'pair_serial.csv'

def run():
    dataset = moses.get_dataset('train')
    if os.path.exists(os.path.join(out_property_path, outname)):
        os.remove(os.path.join(out_property_path, outname))
    # n_samples = len(dataset)

    dataset = dataset[:n_samples]
    total_length = len(dataset)

    start = time.time()
    interval = 10
    pairs = []  
    for start, smiles in enumerate(dataset):
        print('>>> ', start, smiles)
        right_smiles = dataset[start : total_length]

        with Pool(n_jobs) as p:
            similarities = list(p.map(partial(tanimoto_similarity, smiles), right_smiles))
            similar_smiles_no = [start + i for i in range(total_length - start) 
                                 if similarities[i] >= similarity_bound]
            for no in similar_smiles_no:
                pairs.append([start, no])
    
        if start % interval == 0 or start == total_length - 1:
            print("\n----------------- store data --------------------\n")
            df = pd.DataFrame(pairs, columns=["No1", "No2"])
            if not os.path.exists(out_property_path):
                df.to_csv(os.path.join(out_property_path, outname), index=False)
            else:
                df.to_csv(os.path.join(out_property_path, outname), 
                        index=False, mode='a', header=False)
            pairs = []

    elipsed_time = time.time() - start
    print("- preprocessing time for training set:", str(timedelta(seconds=elipsed_time)), "\n")


if __name__ == "__main__":
    # run()
    prop_path = '/fileserver-gamma/chaoting/ML/data/moses/prop_temp_serial.csv'
    pair_path = '/fileserver-gamma/chaoting/ML/data/moses_aug/pair_serial.csv'

    prop = pd.read_csv(prop_path)
    pair = pd.read_csv(pair_path)
    
    n_sample = 40
    src_no = pair['no1'].tolist()[:n_sample]
    trg_no = pair['no2'].tolist()[:n_sample]
    
    print(src_no)
    print(trg_no)
    
    prop_name = ['logP', 'tPSA', 'QED']

    dataset = moses.get_dataset('train')
    df_dataset = pd.DataFrame({
        'smiles': dataset,
        'no': [i+1 for i in range(len(dataset))]
    })    

    src_smi = df_dataset.set_index('no').loc[src_no].reset_index(inplace=False)
    src_smi = src_smi[['smiles']].rename(columns={'smiles': 'src'})
    src_prop = prop.set_index('no').loc[src_no].reset_index(inplace=False)
    src_prop = src_prop.rename(columns={ k:f'src_{k}' for k in src_prop.columns })

    trg_smi = df_dataset.set_index('no').loc[trg_no].reset_index(inplace=False)
    trg_smi = trg_smi[['smiles']].rename(columns={'smiles': 'trg'})
    trg_prop = prop.set_index('no').loc[trg_no].reset_index(inplace=False)
    trg_prop = trg_prop.rename(columns={ k:f'trg_{k}' for k in trg_prop.columns })

    res = pd.concat([src_smi, trg_smi, src_prop, trg_prop], axis=1)

    print(res)

    # trg_prop = prop.loc[prop.apply(lambda x: x['no'] in trg_no, axis=1)]

    # trg_prop = prop.query('no in @trg_no')

    # src_prop = prop[prop['no'].isin(src_no)]
    # trg_prop = prop[prop['no'].isin(trg_no)]
