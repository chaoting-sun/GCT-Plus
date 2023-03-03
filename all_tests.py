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


def test_rdkit():
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from rdkit.Chem.rdchem import AtomValenceException
    import numpy as np
    
    bad_smiles = "Cc1nnc(-c2c(C#N)c(N)n(-c3ccccc3)c2C=c1=O)N(C)C"
    bad_mol1 = Chem.MolFromSmiles(bad_smiles, sanitize=True)
    bad_mol2 = Chem.MolFromSmiles(bad_smiles, sanitize=False)    

    print("logP:", Descriptors.MolLogP(bad_mol1))
    print("tPSA:", Descriptors.TPSA(bad_mol1))
    
    try:
        print("QED:", Descriptors.qed(bad_mol1))
    except AtomValenceException:
        print("QED:", np.nan)
    print('end program')


def test_novelty():
    from multiprocessing import Pool
    from Utils.properties import get_mol
    from Utils.mapper import mapper
    from moses.metrics import metrics
    import pandas as pd

    n_jobs = 4
    train = pd.read_csv("/fileserver-gamma/chaoting/ML/dataset/moses/raw/train/smiles_serial.csv")
    # train = train['smiles'].tolist()
    
    gen_path = "/fileserver-gamma/chaoting/ML/cvae-transformer/Inference/tf_sim1.00/3.73_65.38_0.86.txt"
    gen_path1 = "/fileserver-gamma/chaoting/ML/cvae-transformer/Inference/mlptf_sim0.90_1_mse_epoch3/2.50_65.38_0.86.txt"
    gen = pd.read_csv(gen_path1, sep="\t")
    gen = gen["smiles"].tolist()
    print(f'train: {len(train)}\tgen: {len(gen)}, {len(set(gen))}')

    if n_jobs != 1:
        pool = Pool(n_jobs)
        close_pool = True
    else:
        pool = 1
    mols = mapper(pool)(get_mol, gen)
    gen_novelty = metrics.novelty(mols, train, n_jobs)
    print(gen_novelty)
    if close_pool:
        pool.close()
        pool.join()


def bar_plot():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go

    n = 83

    df = pd.DataFrame({
        'dimension': [i for i in range(n)],
        'data': np.random.normal(0, 0.5, n)
    })

    df["color"] = np.where(df["data"] < 0, 'orange', 'darkcyan')

    fig = go.Figure()
    fig.add_trace(
        go.Bar(name='data',
               x=df['dimension'],
               y=df['data'],
               marker_color=df['color'])
    )
    fig.update_layout(
        title={
            'text': 'Correlation of Mean and Sigma',
            'x': 0.5,
            'y': 0.9,
            'xanchor': 'center'
        },
        xaxis_title="dimension",
        yaxis_title="correlation coefficient",
        font={ 'size': 16 }
    )
    
    fig.update(layout_yaxis_range=[-1,1])
    fig.update_layout(barmode='stack')
    fig.write_image("./test.png")
    
    return
    # plt.figure(figsize=(14,10), dpi=80)
    df.plot(kind='bar', color=df['color'])
    plt.title('Correlation of Mean and Sigma', fontsize=20)
    plt.xlabel('dimension', fontsize=18)
    plt.ylabel('correlation coefficient', fontsize=18)
    plt.ylim(bottom=-1, top=1)
    plt.savefig("./test.png")


def multihead_attention():
    import torch
    import torch.nn as nn
    from torch.nn import MultiheadAttention

    t = torch.normal(2, 3, size=(10, 8, 4))
    net = nn.MultiheadAttention(embed_dim=4, num_heads=1)
    k = net(t, t, t)
    k1, k2 = k

    print(k1.size(), k2.size())
    print(t.size())
    print(k.size())


def test_if_different():
    import argparse
    from Model.build_model import get_model
    from Configuration.config import hard_constraints_opts
    from Utils.field import get_tf_fields
    
    parser = argparse.ArgumentParser()
    hard_constraints_opts(parser)
    
    args = parser.parse_args()

    fields, SRC, TRG = get_tf_fields(args.conditions, args.molgct_path)
    # args.use_model_path = '/fileserver-gamma/chaoting/ML/cvae-transformer/Experiment_Repeat/transformer1/model_10.pt'
    args.use_model_path = '/fileserver-gamma/chaoting/ML/cvae-transformer/Experiment/transformer/model_10.pt'
    model = get_model(args, len(SRC.vocab), len(TRG.vocab))

    for name, params in model.named_parameters():
        if name == 'decoder.layers.2.attn_2.out.weight':
            print(params)


def smoothness_plot():
    import matplotlib.pyplot as plt
    import numpy as np

    def arrowed_spines(fig, ax):

        xmin, xmax = ax.get_xlim() 
        ymin, ymax = ax.get_ylim()

        # removing the default axis on all sides:
        for side in ['bottom','right','top','left']:
            ax.spines[side].set_visible(False)

        # removing the axis ticks
        plt.xticks([]) # labels 
        plt.yticks([])
        ax.xaxis.set_ticks_position('none') # tick markers
        ax.yaxis.set_ticks_position('none')

        # get width and height of axes object to compute 
        # matching arrowhead length and width
        dps = fig.dpi_scale_trans.inverted()
        bbox = ax.get_window_extent().transformed(dps)
        width, height = bbox.width, bbox.height

        # manual arrowhead width and length
        hw = 1./20.*(ymax-ymin) 
        hl = 1./20.*(xmax-xmin)
        lw = 1. # axis line width
        ohg = 0.3 # arrow overhang

        # compute matching arrowhead length and width
        yhw = hw/(ymax-ymin)*(xmax-xmin)* height/width 
        yhl = hl/(xmax-xmin)*(ymax-ymin)* width/height

        # draw x and y axis
        ax.arrow(xmin, 0, xmax-xmin, 0., fc='k', ec='k', lw = lw, 
                head_width=hw, head_length=hl, overhang = ohg, 
                length_includes_head= True, clip_on = False) 

        ax.arrow(0, ymin, 0., ymax-ymin, fc='k', ec='k', lw = lw, 
                head_width=yhw, head_length=yhl, overhang = ohg, 
                length_includes_head= True, clip_on = False)

    # plot
    x = np.arange(-2., 10.0, 0.01)
    plt.plot(x, x**2)
    fig = plt.gcf()
    fig.set_facecolor('white') 
    ax = plt.gca()

    arrowed_spines(fig, ax)

    plt.savefig('aa.png')
    
    
def tokenize_smiles():
    from Utils.field import smiles_fields
    
    smiles = 'CC1C2CCC(C2)C1CN(CCO)C(=O)c1ccc(Cl)cc1'
    field_path = '/fileserver-gamma/chaoting/ML/cvae-transformer/molGCT/fields/'
    
    SRC, _ = smiles_fields(field_path)
    tokens = SRC.tokenize(smiles)
    
    print([SRC.vocab.stoi[t] for t in tokens])
    

def test_cross_entropy():
    import torch.nn.functional as F
    
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.randint(5, (3,), dtype=torch.int64)
    
    loss = F.cross_entropy(input, target)
    print(loss)

    loss = F.cross_entropy(F.log_softmax(input, dim=-1), target)
    print(loss)
    

def smiles_to_scaffold():
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    smiles = 'CC1C2CCC(C2)C1CN(CCO)C(=O)c1ccc(Cl)cc1'
    
    m1 = Chem.MolFromSmiles(smiles)
    m2 = MurckoScaffold.GetScaffoldForMol(m1)
    smi_sca = Chem.MolToSmiles(m2)

    print(smiles)
    print(smi_sca)


if __name__ == '__main__':
    # test_intdiv()
    # test_speed_of_open_binaryfiles()
    # memmap_test()
    # merge_np_file()
    # test_float16()
    # file_float32_to_float16()
    # test_if_restart_is_possible()
    # tensor_to_numpy()
    # create_a_dataset()
    # test_rdkit()
    # test_novelty()
    # bar_plot()
    # multihead_attention()
    # test_if_different()
    # smoothness_plot()
    # tokenize_smiles()
    # test_cross_entropy()
    
    smiles_to_scaffold()