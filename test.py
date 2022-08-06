import os
from time import time
import numpy as np

import torch
import dill as pickle 
from moses.metrics import metrics


"""
- purpose: test if internal diversity is an intensive property:
- conclusion: Yes
"""

def test_intdiv():
    smi1 = 'CCCS(=O)c1ccc2[nH]c(=NC(=O)OC)[nH]c2c1'
    smi2 = 'CC(C)(C)C(=O)C(Oc1ccc(Cl)cc1)n1ccnc1'
    smi3 = 'Cc1c(Cl)cccc1Nc1ncccc1C(=O)OCC(O)CO'
    smi4 = 'Cn1cnc2c1c(=O)n(CC(O)CO)c(=O)n2C'
    nums = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

    print(f'#smiles: {2}, repeat: {n}')
    for n in nums:
        smi_list = [smi1, smi2] * n
        internal_divergence = intdiv(smi_list)
        print(internal_divergence)

    print(f'#smiles: {3}, repeat: {n}')
    for n in nums:
        smi_list = [smi1, smi2, smi3] * n
        internal_divergence = intdiv(smi_list)
        print(internal_divergence)

    print(f'#smiles: {4}, repeat: {n}')
    for n in nums:
        smi_list = [smi1, smi2, smi3, smi4] * n
        internal_divergence = intdiv(smi_list)
        print(internal_divergence)


def intdiv(valid_smiles): return metrics.internal_diversity(
    valid_smiles) if len(valid_smiles) > 0 else 0


def test_speed_of_open_binaryfiles():
    number = 5000
    start1, start2 = 1001, 600001

    folder = '/fileserver-gamma/chaoting/ML/dataset/moses/raw/train/encoder_outputs'
    range_1 = np.arange(start1, start1+number+1)
    range_2 = np.arange(start2, start2+number+1)

    start_time = time()
    for n in range_1:
        f = pickle.load(open(os.path.join(folder, f'{n}.pt'), 'rb'))
    end_time = time()
    print(f'Time for opening files from {range_1[0]} to {range_1[-1]}:', end_time-start_time)

    start_time = time()
    for n in range_2:
        f = pickle.load(open(os.path.join(folder, f'{n}.pt'), 'rb'))
    end_time = time()
    print(f'Time for opening files from {range_2[0]} to {range_2[-1]}:', end_time-start_time)


    start_time = time()
    for n in range_1:
        f = pickle.load(open(os.path.join(folder, f'{n}.pt'), 'rb'))
    end_time = time()
    print(f'Time for opening files from {range_1[0]} to {range_1[-1]}:', end_time-start_time)

    start_time = time()
    for n in range_2:
        f = pickle.load(open(os.path.join(folder, f'{n}.pt'), 'rb'))
    end_time = time()
    print(f'Time for opening files from {range_2[0]} to {range_2[-1]}:', end_time-start_time)


if __name__ == '__main__':
    # test_intdiv()
    test_speed_of_open_binaryfiles()
