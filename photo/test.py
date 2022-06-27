import os
from math import ceil
import pandas as pd
import numpy as np

from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import plotly.express as px
from moses.metrics import metrics
from moses import get_dataset

logp_lb = 0.03
logp_ub = 4.97
tpsa_lb = 17.92
tpsa_ub = 112.83
qed_lb = 0.58
qed_ub = 0.95


def pearson_correlation_coefficient(properties):
    """
    Pearson correlation coefficient on logP, tPSA, and QED
    """

    logp = properties['logP'].tolist()
    tpsa = properties['tPSA'].tolist()
    qed = properties['QED'].tolist()
  
    print(pearsonr(logp, tpsa)[0])
    print(pearsonr(logp, qed)[0])
    print(pearsonr(qed, tpsa)[0])

    exit()

    corr = properties.corr()
    cmap = sns.diverging_palette(10, 220, as_cmap=True)
    sns.heatmap(corr, vmin=-1.0, vmax=1.0, square=True, cmap=cmap)
    plt.title("Pearson Correlation", fontsize=18)

    plt.savefig('pearson.png')

    s = scatter_matrix(frame=properties[:10000],
                       alpha=0.8, 
                       figsize=(12, 12),
                       s=2,
                       diagonal='kde'
                       )
    plt.savefig('scatter.png')


def error_plot():
    """
    3D error plot on logP, tPSA, and QED
    """
    
    num_samples_each = 5

    logp_values = np.linspace(logp_lb, logp_ub, num=num_samples_each)
    tpsa_values = np.linspace(tpsa_lb, tpsa_ub, num=num_samples_each)
    qed_values = np.linspace(qed_lb, qed_ub, num=num_samples_each)

    header_dict = {
        'logp': [],
        'tpsa': [],
        'qed': [],
        'logp_nmae': [],
        'tpsa_nmae': [],
        'qed_nmae': [],
    }

    for logp in logp_values:
        for tpsa in tpsa_values:
            for qed in qed_values:
                mean_p = os.path.join('molGCT', 'inference',
                        'mean_{:.2f}_{:.2f}_{:.2f}.txt'.format(logp, tpsa, qed))
                mean_df = pd.read_csv(mean_p, sep='\t')

                header_dict['logp'].append(logp)
                header_dict['tpsa'].append(tpsa)
                header_dict['qed'].append(qed)

                header_dict['logp_nmae'].append(mean_df['logp_mae'].values[0] / logp)
                header_dict['tpsa_nmae'].append(mean_df['tpsa_mae'].values[0] / tpsa)
                header_dict['qed_nmae'].append(mean_df['qed_mae'].values[0] / qed)

    data_df = pd.DataFrame.from_dict(header_dict)
    print(data_df.describe())

    fig = px.scatter_3d(data_df, x='logp', y='tpsa', z='qed', color='logp_nmae')
    fig.update_layout(
        width=630, height=500,
        margin=dict(t=40, r=80, l=40, b=40)
    )
    fig.update_coloraxes(cmax=25, cmin=0)
    fig.update_traces(marker_size=6)
    fig.write_image('nmae_logp.png')

    fig = px.scatter_3d(data_df, x='logp', y='tpsa', z='qed', color='tpsa_nmae')
    fig.update_layout(
        width=630, height=500,
        margin=dict(t=40, r=80, l=40, b=40)
    )
    fig.update_coloraxes(cmax=0.5, cmin=0)
    fig.update_traces(marker_size=6)
    fig.write_image('nmae_tpsa.png')

    fig = px.scatter_3d(data_df, x='logp', y='tpsa', z='qed', color='qed_nmae')
    fig.update_layout(
        width=630, height=500,
        margin=dict(t=40, r=80, l=40, b=40)
    )
    fig.update_coloraxes(cmax=0.5, cmin=0)
    fig.update_traces(marker_size=6)
    fig.write_image('nmae_qed.png')


def analyze_data(prop_df):
    """
    Check the percentage of molecules of the
    given property bound over all the molecules 
    """

    print(prop_df.describe())
    print('logP count in the bound:', len(prop_df[(logp_lb < prop_df.logP) & (prop_df.logP < logp_ub)]) / len(prop_df))
    print('tPSA count in the bound:', len(prop_df[(tpsa_lb < prop_df.tPSA) & (prop_df.tPSA < tpsa_ub)]) / len(prop_df))
    print('QED count in the bound:', len(prop_df[(qed_lb < prop_df.QED) & (prop_df.QED < qed_ub)]) / len(prop_df))


    # print('tPSA count in the bound:', prop_df['tPSA' < tpsa_ub].count() / len(prop_df))
    # print('QED count in the bound:', prop_df['QED' < qed_ub].count() / len(prop_df))

import sys
from rdkit import Chem

def numAtomsFromSmiles(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        sys.exit("Invalid SMILES: " + smiles)
    mol = Chem.AddHs(mol)
    return mol.GetNumAtoms()


if __name__ == '__main__':
    # file_path = '/fileserver-gamma/chaoting/ML/data/moses/prop_temp.csv'
    # prop_df = pd.read_csv(file_path)
    # pearson_correlation_coefficient(prop_df)

    # analyze_data(prop_df)
    
    # error_plot()

    n_samples = 100000

    train = get_dataset('train')[:n_samples]
    test = get_dataset('test')[:n_samples]
    test_scaffolds = get_dataset('test_scaffolds')[:n_samples]

    train = list(map(numAtomsFromSmiles, train))
    test = list(map(numAtomsFromSmiles, test))
    test_scaffolds = list(map(numAtomsFromSmiles, test_scaffolds))

    df = pd.DataFrame({
        'train': train,
        'test': test,
        'train_scaffolds': test_scaffolds
    })

    fig = df.plot.kde(bw_method=0.1).get_figure()
    plt.tight_layout()
    fig.savefig('atomnum_dist.png')
    
    exit()

    df = get_dataset('train')
    intDiv = metrics.internal_diversity(df[:100000])
    print("internal diversity >", intDiv)
    exit()

    pearson_correlation_coefficient(prop_df)

    # print(prop_df['logP'].min(), prop_df['logP'].max())
    # print(prop_df['tPSA'].min(), prop_df['tPSA'].max())
    # print(prop_df['QED'].min(), prop_df['QED'].max())

    # example2(file_path)
    # ternary_plot(file_path, fig_path='test.png')