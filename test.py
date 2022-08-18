import os
import time
import numpy as np

import dill as pickle 

"""
- purpose: test if internal diversity is an intensive property:
- conclusion: Yes
"""

def test_intdiv():
    from moses.metrics import metrics

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

    start_time = time.time()
    for n in range_1:
        f = pickle.load(open(os.path.join(folder, f'{n}.pt'), 'rb'))
    end_time = time.time()
    print(f'Time for opening files from {range_1[0]} to {range_1[-1]}:', end_time-start_time)

    start_time = time.time()
    for n in range_2:
        f = pickle.load(open(os.path.join(folder, f'{n}.pt'), 'rb'))
    end_time = time.time()
    print(f'Time for opening files from {range_2[0]} to {range_2[-1]}:', end_time-start_time)


    start_time = time.time()
    for n in range_1:
        f = pickle.load(open(os.path.join(folder, f'{n}.pt'), 'rb'))
    end_time = time.time()
    print(f'Time for opening files from {range_1[0]} to {range_1[-1]}:', end_time-start_time)

    start_time = time.time()
    for n in range_2:
        f = pickle.load(open(os.path.join(folder, f'{n}.pt'), 'rb'))
    end_time = time.time()
    print(f'Time for opening files from {range_2[0]} to {range_2[-1]}:', end_time-start_time)


"""
- purpose: test if memory-mapped file support increase spead
"""

import dill as pickle
import numpy as np
import torch

def memmap_test():
    file_path = "/fileserver-gamma/chaoting/ML/dataset/moses/raw/train"
    # combine files into one
    os.makedirs(os.path.join(file_path, "test"), exist_ok=True)

    num_arrs = 10
    num_arrs = 1000

    all_arr = None

    if not os.path.exists(os.path.join(file_path, "test", "test_large.pt")):
        for i in range(num_arrs):
            arr = pickle.load(open(os.path.join(file_path,
                                "encoder_outputs", f"{i+1}.pt"), 'rb'))
            arr = np.expand_dims(arr, axis=0)
            print(i, arr.shape)
            if all_arr is not None:
                all_arr = np.concatenate((all_arr, arr), axis=0)
            else:
                all_arr = arr
        with open(os.path.join(file_path, "test", "test_large.pt"), "wb") as file:
            pickle.dump(all_arr, file)

    arr = np.memmap(os.path.join(file_path, "test", "test_large.pt"), 
                    dtype='float32', mode='c', shape=(num_arrs, 80, 512))
    print(arr.shape)

    order = np.arange(num_arrs)
    np.random.shuffle(order)

    two_choices = 1

    if two_choices == 0:
        start = time.time()
        for i in order:
            f = torch.from_numpy(arr[i])
            f2 = torch.FloatTensor()
        end = time.time()
        print("elapsed time (memap):", end-start)
    elif two_choices == 1:
        start = time.time()
        for i in order:
            f = pickle.load(open(os.path.join(file_path, "encoder_outputs", f"{i+1}.pt"), "rb"))
        end = time.time()
        print("elapsed time:", end-start)


"""
- Purpose: merge 100w+ 2D numpy array files into a numpy array file
"""

import sys
import numpy as np

def merge_np_file():
    preprocess = False
    data_dict = {
        "train": 1584663,
        "validation": 176074
    }
    data_type = "train"
    data_number = data_dict[data_type]

    file_path = f"/fileserver-gamma/chaoting/ML/dataset/moses/raw/{data_type}"


    # if preprocess is True:
    #     file = np.memmap(os.path.join(file_path, "encoder_outputs.npy"),
    #                     dtype=np.float32, mode="w+", shape=(data_number, 80, 512))
        
    #     for i in range(data_number):
    #         file[i, :, :] = pickle.load(open(os.path.join(file_path, "encoder_outputs", f"{i+1}.pt"), "rb"))
    #         print(i)
    #     file.flush()

    # a = np.load(os.path.join(file_path, "encoder_outputs.npy"), mmap_mode='c', allow_pickle=True)

    # x_on_disk = np.memmap(os.path.join(file_path, "encoder_outputs.npy"), 
    #                       dtype=np.float32, mode='r+', shape=(data_number, 80, 512))
    x_on_disk = np.memmap(os.path.join(file_path, "encoder_outputs.npy"), 
                          dtype=np.float32, mode='c', shape=(data_number, 80, 512))
    print(sys.getsizeof(x_on_disk))
    print(x_on_disk.shape)
    print(x_on_disk[0])
    print('test:')
    a = torch.from_numpy(x_on_disk[1])
    
    print(type(a))
    print(a)


def test_float16():
    data_dict = {
        "train": 1584663,
        "validation": 176074
    }
    data_type = 'train'
    data_number = data_dict[data_type]

    file_path = f"/fileserver-gamma/chaoting/ML/dataset/moses/raw/{data_type}"

    step1 = False

    if step1:
        arr = np.memmap(os.path.join(file_path, "test_float16.pt"),
                        dtype='float16', mode='w+', shape=(data_number, 80, 512))
        arr.flush()
    else:
        x_on_disk = np.memmap(os.path.join(file_path, "test_float16.pt"),
                            dtype=np.float16, mode='c', shape=(data_number, 80, 512))
        print(x_on_disk.shape)


def file_float32_to_float16():
    data_dict = {
        "train": 1584663,
        "validation": 176074
    }
    data_type = 'validation'
    data_number = data_dict[data_type]

    file_folder = f"/fileserver-gamma/chaoting/ML/dataset/moses/raw/{data_type}"
    
    for i in range(data_number):
        file_path = os.path.join(file_folder, "encoder_outputs", f"{i+1}.pt")
        arr = pickle.load(open(file_path, "rb"))
        arr = arr.astype(np.float16)
        pickle.dump(arr, open(file_path, "wb"))
        print(i+1)


def test_if_restart_is_possible():
    print("start test.py")
    for i in range(5):
        print("This is", i)
    print("pid:", os.getpid())
    raise Exception('Exception in loop')


def move_gamma_to_fileserver2():
    source_path = "/"
    target_path = ''


import dill as pickle

def tensor_to_numpy():
    data_dict = {
        "train": 1584663,
        "validation": 176074
    }
    data_type = "validation"
    data_number = data_dict[data_type]

    infile_folder = f"/fileserver2/chaoting/dataset/moses/raw/{data_type}/tensor"
    oufile_folder = f"/fileserver2/chaoting/dataset/moses/raw/{data_type}/encoder_outputs"
    
    for i in range(data_number):
        inpath = os.path.join(infile_folder, f"{i+1}.pt")
        outpath = os.path.join(oufile_folder, f"{i+1}.pt")

        t = torch.load(inpath)
        t = t.detach().cpu().numpy()
        pickle.dump(t, open(outpath, "wb"))
        print(i)


import csv
import linecache
from torchtext import data
from time import time
from collections import OrderedDict

class LazyTextDataset(data.Dataset):
    def __init__(self, filename):
        self._filename = filename
        self._total_data = 0
        with open(filename, "r") as f:
            self._total_data = len(f.readlines()) - 1
        
        line = linecache.getline(self._filename, 1)
        # list of header names
        self.header = list(csv.reader([line], delimiter=','))[0]

    def __getitem__(self, idx):
        # the line with index 0 is always empty
        line = linecache.getline(self._filename, idx + 2)
        csv_line = list(csv.reader([line], delimiter=','))[0]

        sample = OrderedDict()
        for i, val in enumerate(csv_line):
            sample[self.header[i]] = val

        print(sample)
        return sample

    def __len__(self):
        return self._total_data

import pandas as pd

class customDataset(data.Dataset):
    def __init__(self, file_path, transform=None):
        self.transform = transform
        self.file = pd.read_csv(file_path, engine="pyarrow")

    def __getitem__(self, idx):
        samples = self.file.iloc[idx]
        print(samples)
        return samples

    def __len__(self):
        return len(self.file)
    

def create_a_dataset():
    file_name = "validation"
    file_path = f"/fileserver-gamma/chaoting/ML/dataset/moses/aug/data_sim0.70/{file_name}.csv"
    
    import sys
    sys.setrecursionlimit(10000)
    
    construct_time = -time()
    
    construct_time += time()
    f = customDataset(file_path)
    enumerate_time = -time()
    print(f[0])
    for i, d in enumerate(f):
        print(d)
        if i == 10:
            break
    enumerate_time += time()
    
    print(f"time1: {construct_time}"
          f"time2: {enumerate_time}")


if __name__ == '__main__':
    # test_intdiv()
    # test_speed_of_open_binaryfiles()
    # memmap_test()
    # merge_np_file()
    # test_float16()
    # file_float32_to_float16()
    # test_if_restart_is_possible()
    # tensor_to_numpy()
    create_a_dataset()