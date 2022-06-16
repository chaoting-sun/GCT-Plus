import os
import sys
import pandas as pd
from functools import partial
from multiprocessing import Pool
import time
from datetime import timedelta


sys.path.append('/home/chaoting/tools/rdKit/similarity/')
from similarity import tanimoto_similarity
# sys.path.append('/fileserver-gamma/chaoting/ML/data/moses/')

import Process.data_preparation as pdp

similarity_bound = 0.60
n_jobs = 12
n_samples = 5000

property_path ='/fileserver-gamma/chaoting/ML/data/moses/'
out_property_path ='/fileserver-gamma/chaoting/ML/data/moses_aug/'
inname = 'prop_temp.csv'
outname = 'pair_serial.csv'

def run():
    dataset = pdp.get_dataset('moses', 'train')
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
    run()