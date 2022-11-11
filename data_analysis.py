import os
import argparse
import swifter
from pandarallel import pandarallel
import pandas as pd
import numpy as np
import joblib

from time import time
from Configuration.config import options
from Utils.property import property_prediction, get_mol, tanimoto_similarity, logP, tPSA, QED
import matplotlib.pyplot as plt


def plot_1d_density():
    pass


def plot_similarity_distribution():
    pass


# def sample_random_pairs(df_data, ):


def get_random_pairs(data_path, in_cols, out_cols, n_pairs, n_tests=None):
    df_data = pd.read_csv(data_path)
    if n_tests is not None:
        df_data = df_data[:n_tests]
    np_aug = df_data[in_cols].to_numpy()
    idx = np.random.choice(len(np_aug), n_pairs)
    df_pairs = pd.DataFrame({
        out_cols[0]: np_aug[idx, 0],
        out_cols[1]: np_aug[idx, 1]
    })
    return df_pairs


def compute_tanimoto_similarity(df_data, in_cols=['src', 'trg'], out_col='sim'):
    pandarallel.initialize(progress_bar=True)
    df_data[out_col] = df_data.parallel_apply(lambda x: tanimoto_similarity(
        x[in_cols[0]], x[in_cols[1]]), axis=1)
    return df_data[[out_col]]


def df_plot(df, cols, file_name):
    fig = plt.figure()
    plt.xlabel("X axis label")
    plt.ylabel("Y axis label")
    plt.legend(cols)
    for col in cols:
        df[col].plot.kde(bw_method=0.01)
    fig.savefig(file_name)

def df_bar_plot(df, x, y, file_name, xlabel='x label', ylabel='y label'):
    ax = df.plot.bar(x=x, y=y, rot=0, figsize=(18, 18))
    fig = ax.get_figure()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    fig.savefig(file_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = options(parser)
    args = parser.parse_args()

    scaler = joblib.load(args.scaler_path)

    n_pairs = 100000
    data_type = 'train'
    similarity_choices = (0.70, 0.80, 0.90)
    args.similarity = 0.70

    """correlation between similarity and properties"""

    df_aug = pd.read_csv(os.path.join(
        args.data_path, 'aug', f'data_sim{args.similarity:.2f}', f'{data_type}.csv'))
    
    print('inverse transform')

    df_aug[[f'src_{c}' for c in args.conditions]] = scaler.inverse_transform(
        df_aug[[f'src_{c}' for c in args.conditions]])
    df_aug[[f'trg_{c}' for c in args.conditions]] = scaler.inverse_transform(
        df_aug[[f'trg_{c}' for c in args.conditions]])
    for c in args.conditions:
        df_aug[f'd_{c}'] = df_aug[f'trg_{c}'] - df_aug[f'src_{c}']

    df_aug_sim = compute_tanimoto_similarity(
        df_aug, in_cols=['src', 'trg'], out_col=f'aug_sim{args.similarity:.2f}')
    df_aug[[f'aug_sim{args.similarity:.2f}']] = df_aug_sim[[f'aug_sim{args.similarity:.2f}']]
    print(df_aug.head())

    for c in args.conditions:
        print(df_aug[f'd_{c}'].corr(df_aug[f'aug_sim{args.similarity:.2f}']))
    
    exit()

    df_aug[[f'src_{c}' for c in args.conditions]] = df_aug.apply(lambda x: 
        scaler.inverse_transform(x[[f'src_{c}' for c in args.conditions]], axis=1))        
    df_aug[[f'trg_{c}' for c in args.conditions]] = df_aug.apply(lambda x: 
        scaler.inverse_transform(x[[f'trg_{c}' for c in args.conditions]], axis=1))


    exit(0)

    df_aug = pd.read_csv(os.path.join(
        args.data_path, 'aug', f'data_sim{args.similarity:.2f}', f'{data_type}.csv'))

    df_prop = pd.read_csv(os.path.join(
        args.data_path, 'raw', data_type, 'prop_serial.csv'), index_col='no')
    df_raw = pd.read_csv(os.path.join(
        args.data_path, 'raw', data_type, 'smiles_serial.csv'), index_col='no')

    # df_prop = df_prop.set_index('no')
    # df_raw = df_raw.set_index('no')

    print('Gather properties of source/target SMILES from raw dataset')

    src_prop_dict = { c: f'src_{c}' for c in args.conditions }
    trg_prop_dict = { c: f'trg_{c}' for c in args.conditions }

    src_no, trg_no = df_aug.loc[:, 'src_no'], df_aug.loc[:, 'trg_no']
    df_src_smi = df_raw.loc[src_no, ['smiles']].rename(
        columns={ 'smiles': 'src' }).reset_index(drop=True)
    df_trg_smi = df_raw.loc[trg_no, ['smiles']].rename(
        columns={ 'smiles': 'trg' }).reset_index(drop=True)

    df_src_prop = df_prop.loc[src_no, args.conditions].rename(
        columns=src_prop_dict).reset_index(drop=True)
    df_trg_prop = df_prop.loc[trg_no, args.conditions].rename(
        columns=trg_prop_dict).reset_index(drop=True)

    print('Concatenate SMILES/properties of source/target SMILES')

    df_all = pd.concat([df_src_smi, df_trg_smi, df_src_prop, df_trg_prop], axis=1)

    print('Find the SMILES with the most pairs.')

    df_freq = df_all.groupby(by=['src']).agg({ 'trg': len })
    df_freq = df_freq.sort_values(by=['trg'], ascending=False)
    df_freq['smi_no'] = [i+1 for i in range(len(df_freq))]

    print('Make a bar plot.')

    df_bar_plot(df_freq.iloc[:50], x='smi_no', y='trg', file_name='2.png')

    src_smi1 = df_freq.index.values[0]
    print('src_smi1:', src_smi1)

    src_smi1_pairs = df_all.loc[df_all.src == src_smi1].reset_index(drop=True)
    src_smi1_pairs.to_csv('test.csv')
    # for c in args.conditions:
    #     src_smi1_pairs[f'del_{c}'] = src_smi1_pairs[f'trg_{c}'] - src_smi1_pairs[f'src_{c}']

    print(src_smi1_pairs.describe())
    print(src_smi1_pairs.head())

    exit()

    """property distribution"""

    similarity = 0.80

    df_aug = pd.read_csv(os.path.join(
        args.data_path, 'aug', f'data_sim{similarity:.2f}', f'{data_type}.csv'))
    src_no, trg_no = df_aug.loc[:, 'src_no'], df_aug.loc[:, 'trg_no']

    print(src_no.describe())
    print(trg_no.describe())

    df_raw = pd.read_csv(os.path.join(
        args.data_path, 'raw', data_type, 'smiles_serial.csv'))

    df_raw = df_raw.set_index('no')

    print('df_raw:\n', df_raw.describe())
    
    df_src = df_raw.loc[src_no, ['smiles']].rename(
        columns={ 'smiles': 'src' }).reset_index(drop=True)
    df_trg = df_raw.loc[trg_no, ['smiles']].rename(
        columns={ 'smiles': 'trg' }).reset_index(drop=True)

    print('df_src:\n', df_src.describe())

    df_pairs = pd.concat([df_src, df_trg], axis=1)

    print('df_pairs:\n', df_pairs.describe())

    df_freq = df_pairs.groupby(by=['src']).agg({ 'trg': len })
    df_freq = df_freq.sort_values(by=['trg'], ascending=False)

    print(df_freq.head())

    """
    found:
    CNC(=O)c1cccc(NCC(=O)Nc2cccc(C(=O)NC)c2)c1   31
    COc1ccc(CNC(=O)c2ccc(OC)cc2)cc1              28
    CC(=O)c1cccc(NC(=O)COc2cccc(C(C)=O)c2)c1     28
    COc1cccc(NCC(=O)Nc2cccc(OC)c2)c1             27
    COc1ccc(NC(=O)COc2ccc(OC)cc2)cc1             24
    """


    exit(0)


    logp_p, tpsa_p, qed_p = (property_prediction[c](mol)
                            for c in args.conditions)

    exit(0)

    """similarity analysis"""

    df_raw = pd.read_csv(os.path.join(
        args.data_path, 'raw', data_type, 'smiles_serial.csv'))
    np_raw = df_raw['smiles'].to_numpy()

    src_idx, trg_idx = np.random.choice(len(np_raw), n_pairs), \
        np.random.choice(len(np_raw), n_pairs)

    df_pairs = pd.DataFrame({
        'raw_src': np_raw[src_idx],
        'raw_trg': np_raw[trg_idx]
    })

    raw_similarities = compute_tanimoto_similarity(
        df_pairs, in_cols=['raw_src', 'raw_trg'], out_col='sim_raw')

    aug_similarities = None
    for sim in similarity_choices:
        print(f'similarity = {sim}')
        df_pairs = get_random_pairs(
            data_path=os.path.join(args.data_path, 'aug',
                                   f'data_sim{sim:.2f}', f'{data_type}.csv'),
            in_cols=['src', 'trg'],
            out_cols=[f'src_sim{sim:.2f}', f'trg_sim{sim:.2f}'],
            n_pairs=n_pairs,
            n_tests=None
        )
        similarities = compute_tanimoto_similarity(
            df_pairs,
            in_cols=[f'src_sim{sim:.2f}', f'trg_sim{sim:.2f}'],
            out_col=f'sim{sim:.2f}'
        )
        aug_similarities = pd.concat([aug_similarities, similarities], axis=1)

    aug_similarities = pd.concat([raw_similarities, aug_similarities], axis=1)


    print(aug_similarities.describe())

    plt.xlabel("X axis label")
    plt.ylabel("Y axis label")
    plt.xlim(0, 1)
    plt.ylim(0, 10)
    plt.legend(list(aug_similarities.columns.values))

    for col in list(aug_similarities.columns.values):
        aug_similarities[col].plot.kde(
            bw_method=0.01)
    fig.savefig('a0.png')

    fig = plt.figure()

    plt.xlabel("X axis label")
    plt.ylabel("Y axis label")
    plt.xlim(0, 1)
    plt.ylim(0, 10)
    # plt.legend(list(aug_similarities.columns.values))

    aug_similarities.plot.kde(
        bw_method=0.01, colormap='RdBu_r')
    fig.savefig('a1.png')

    exit(0)