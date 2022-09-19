import os
import argparse
import swifter
from pandarallel import pandarallel
import pandas as pd
from time import time
from Configuration.config import options
from Utils.property import property_prediction, get_mol, tanimoto_similarity, logP, tPSA, QED
import matplotlib.pyplot as plt


def get



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = options(parser)
    args = parser.parse_args()

    args.similarity = 0.80
    file_folder = os.path.join(args.data_path, 'aug', f'data_sim{args.similarity:.2f}')
    train_set = pd.read_csv(os.path.join(file_folder, 'train.csv'))

    train_set = train_set[:500000]
    print('start:')
    
    start_time = -time()
    pandarallel.initialize(progress_bar=True)
    train_set['sim'] = train_set.parallel_apply(lambda x: tanimoto_similarity(
                       x['src'], x['trg_en']), axis=1)
    start_time += time()

    fig = plt.figure()
    ax = train_set['sim'].plot.kde()
    fig.savefig('1.png')

    # logp_p, tpsa_p, qed_p = (property_prediction[c](mol)
    #                         for c in args.conditions)


    print(train_set['sim'])
    print('Elipsed time (s):', start_time)