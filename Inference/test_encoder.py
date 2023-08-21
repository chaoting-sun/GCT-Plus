##### deprecated


import os
import pandas as pd
import torch
import numpy as np
import dill as pickle
import seaborn as sns
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from pathos.multiprocessing import ProcessingPool as Pool
from collections import OrderedDict
from scipy.stats import multivariate_normal
import plotly.express as px

from Utils.properties import get_property_fn
from collections import OrderedDict

from Model.build_model import get_generator
from Utils.dataset import SmilesDataset
from Model.modules import get_src_mask
from Inference.utils import prepare_generator
from torchtext.data import Example, Dataset
from Utils.properties import tanimoto_similarity as similarity_fcn
from Utils import DataloaderPreparation
from Model.forward_propagation import forward_propagation
from Utils.smiles import get_mol, murcko_scaffold

from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE
from Utils.analysis import dimension_reduction


# color_map = px.colors.sequential.Plasma
color_map = px.colors.sequential.Turbo

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


def plot_difference_from_stdnorm(x, y):
    # create a 2D histogram of the distribution
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=50)

    # compute the difference between the distribution and a standard Gaussian
    xmesh, ymesh = np.meshgrid(xedges[:-1], yedges[:-1])
    pos = np.empty(xmesh.shape + (2,))
    pos[:, :, 0] = xmesh
    pos[:, :, 1] = ymesh
    gaussian = multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]])
    diff = heatmap - gaussian.pdf(pos)

    # plot the difference
    plt.imshow(diff, cmap='Blues')
    plt.colorbar()
    plt.savefig('1.png')


# from Utils.plot import kde_plot

def kde_plot(df, save_path, xlabel, ylabel, xlimit=None,
             figsize=(6.5, 5), lengend=True):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    df.plot.kde(ax=ax, legend=lengend, xlim=xlimit)
    if lengend:
        ax.legend(fontsize=14)
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    fig.savefig(save_path, bbox_inches="tight")


def get_encoder_output(generator, df_data, use_scaffold,
                       property_list, transform=False,
                       standardize=False):
    """get the density plot on the standardized encoder outputs
    
    The distributions of the mean and logvar from encoder outputs
    are narrow. This function checks if the distributions of all
    dimensions of each sample are similar to normal distributions.
    
    The function firstly gets mean and logvar by a sample, followed
    by standardize them along token dimension. It gathers a lot of
    mean and logvar from several samples, and plot a kde figure.
    """
    mu_results = []
    logvar_results = []
    std_results = []
    mu_mu_results = []
    logvar_mu_results = []
    std_mu_results = []

    for i in range(len(df_data)):
        print(i)
        kwargs = {}
        kwargs['smiles_list'] = [df_data.loc[i, 'smiles']]
        if use_scaffold:
            kwargs['scaffold_list'] = [df_data.loc[i, 'scaffold']]
        if len(property_list) > 0:
            kwargs['econds'] = [df_data.loc[i, property_list]]
            kwargs['transform'] = transform

        _, mu, logvar = generator.encode_smiles(**kwargs)
        std = torch.exp(0.5*logvar)

        mu_mu = mu.mean(dim=1)
        mu_std = mu.std(dim=1)
        logvar_mu = logvar.mean(dim=1)
        logvar_std = logvar.std(dim=1)
        std_mu = std.mean(dim=1)
        std_std = std.std(dim=1)
        
        if standardize:
            mu = (mu - mu_mu) / mu_std
            logvar = (logvar - logvar_mu)/logvar_std
            std = (std - std_mu)/std_std

        mu_results.append(mu.squeeze(0))
        logvar_results.append(logvar.squeeze(0))
        std_results.append(std.squeeze(0))
        mu_mu_results.append(mu_mu)
        logvar_mu_results.append(logvar_mu)
        std_mu_results.append(std_mu)
        
    mu_results = torch.cat(mu_results, dim=0).cpu().numpy()
    logvar_results = torch.cat(logvar_results, dim=0).cpu().numpy()
    std_results = torch.cat(std_results, dim=0).cpu().numpy()
    mu_mu_results = torch.cat(mu_mu_results, dim=0).cpu().numpy()
    logvar_mu_results = torch.cat(logvar_mu_results, dim=0).cpu().numpy()
    std_mu_results = torch.cat(std_mu_results, dim=0).cpu().numpy()

    col_ids = [i for i in range(mu_results.shape[1])]
    
    mu_results = pd.DataFrame(mu_results, columns=col_ids)
    logvar_results = pd.DataFrame(logvar_results, columns=col_ids)
    std_results = pd.DataFrame(std_results, columns=col_ids)
    mu_mu_results = pd.DataFrame(mu_mu_results, columns=col_ids)
    logvar_mu_results = pd.DataFrame(logvar_mu_results, columns=col_ids)
    std_mu_results = pd.DataFrame(std_mu_results, columns=col_ids)

    return ((mu_results, logvar_results, std_results),
            (mu_mu_results, logvar_mu_results, std_mu_results))


@torch.no_grad()
def get_std_of_encoder_outputs(dp, generator, dataset, transform):
    """find the max standard deviation along string length
    
    found: the max. standard deviation is about 0.01, which
    implies that different positions of the same dimension
    get similar values. we may be able to represent a molecule
    by only one vector of size (latent dimension,)
    """
    dataloader = dp.get_dataloader(dataset, batch_size=1,
                                   is_train=True)
    
    mu_std_max = logvar_std_max = -100000

    for i, batch in enumerate(dataloader):
        _, mu, logvar = generator.encode(batch, transform)

        mu = np.squeeze(mu.cpu().numpy(), axis=0)
        mu_std = np.std(mu, axis=0)
        mu_std_max = max(mu_std_max, mu_std.max())
        
        logvar = np.squeeze(logvar.cpu().numpy(), axis=0)
        logvar_std = np.std(logvar, axis=0)
        logvar_std_max = max(logvar_std_max, logvar_std.max())
        
        print('max std of mu:', mu_std_max)
        print('max std of logvar:', logvar_std_max)
        if i == 2000:
            break
    exit()


def sample_mu(generator, df_data, property_list, transform=True,
              n_samples=1000, use_scaffold=False):
    mu_samples = []
    
    for i in range(len(df_data)):
        print(f'# sample: {i}')
        if i == n_samples:
            break
        scaffold_list = [df_data.loc[i, 'scaffold']]
        smiles_list = [df_data.loc[i, 'smiles']]
        econds = df_data.loc[[i], property_list].to_numpy()
        
        if len(property_list) > 0:
            if use_scaffold:
                _, mu, _ = generator.encode_smiles(smiles_list, scaffold_list,
                                                   econds, transform=transform)
            else:
                _, mu, _ = generator.encode_smiles(smiles_list, econds,
                                                   transform=transform)
        else:
            _, mu, _ = generator.encode_smiles(smiles_list)

        mu = mu.mean(dim=1)
        mu_samples.append(mu)
    mu_samples = torch.cat(mu_samples, dim=0)
    return mu_samples.cpu().numpy()


def sample_encoder_outputs(generator, dataloader, transform=True,
                           out_name='mu', n_samples=2000):
    """
    Return:
        outputs: (n_samples, n_dims)
    """
    for i, batch in enumerate(dataloader):
        if i == n_samples:
            break

        z, mu, log_var = generator.encode_batch(batch, transform)
        
        if i == 0:
            dim = z.size(-1)
            out_list = [[] for _ in range(dim)]

        if out_name == 'z':
            out = z
        elif out_name == 'mu':
            out = mu
        elif out_name == 'log_var':
            out = log_var
        
        for d in range(dim):
            out_list[d].append(out[:,0,d].view(-1).cpu().numpy())
        
    sampled_list = []
    for d in range(dim):
        concatenated_out = np.concatenate(out_list[d], axis=0)
        if concatenated_out.shape[0] < n_samples:
            sampled_list.append(concatenated_out)
        else:
            sampled_out = np.random.choice(concatenated_out, n_samples,
                                           replace=False)
            sampled_list.append(sampled_out)
    sampled_list = np.stack(sampled_list, axis=0).T
    return sampled_list


def plot_1d_distribution(outputs, save_folder):
    plt.figure()
    samples = np.random.normal(loc=0, scale=1, size=outputs[0].size(0))
    sns.kdeplot(samples, shade=True)
    for out in outputs:
        sns.kdeplot(out)
    plt.savefig(os.path.join(save_folder, '1d.png'))
    

def plot_2d_distribution(outputs, save_folder):
    for i in range(outputs.size(2)):
        for j in range(i+1, outputs.size(2)):
            plt.figure()
            df = pd.DataFrame({ 'x': outputs[i], 'y': outputs[j] })
            sns.kdeplot(data=df, x="x", y="y", cmap="Blues", shade=True)
            plt.savefig(os.path.join(save_folder, f"2d_{i}-{j}.png"))


def find_top_k(arr, k=3):
    sorted_indices = np.argsort(arr)[::-1]
    top_k_indices = sorted_indices[:k]
    top_k_values = arr[top_k_indices]
    return top_k_indices, top_k_values


def find_bottom_k(arr, k=3):
    sorted_indices = np.argsort(arr)
    bottom_k_indices = sorted_indices[:k]
    bottom_k_values = arr[bottom_k_indices]
    return bottom_k_indices, bottom_k_values


def test_smiles_list_encoder_output(i):
    return [
        'CCc1cccc(OCC(=O)Nc2ccc(C(=O)N(C)C)cc2)c1', 
        'CCc1cccc(OCC(=O)Nc2ccc(C(=O)N(C)C)c(F)c2)c1',
        'Cc1cccc(OCC(=O)NC(C)(C)Cc2ccc3c(c2)OCCO3)c1',
        'CCN1CCCC2(CCN(C(=O)c3cc(C)nc4ccccc34)C2)C1=O',
        'CC1CCCN(C(=O)CN(C)C(=O)NC(C)(C)c2ccccc2F)C1',
        'CC1CCN(C(=O)CN2CCN(C(=O)Nc3cccc(Cl)c3)CC2)CC1',
        'CC1CCN(C(=O)CNC(c2ccc(F)cc2)c2cnn(C)c2)CC1',
        'CC1CN(C(=O)CNC(c2cccc(F)c2)C(C)(C)C)CC(C)O1',
        'CC1CCN(C(=O)CN(C)C(=O)c2ccc3c(c2)CCC3)C(C)C1',
        'CC1CCN(C(=O)CN(C)c2nc3ccccc3s2)CC1'
    ]


def test_can_smiles_list(i):
    smiles_list = {
        0: [
            'C1(C(=O)N2CC3c4ccccc4CCCN3C(=O)C2)CCCCC1',
            'C1CC(C(N2CC(=O)N3CCCc4ccccc4C3C2)=O)CCC1',
            'c1cccc2c1CCCN1C(=O)CN(C(=O)C3CCCCC3)CC12',
            'c1cccc2c1CCCN1C2CN(C(=O)C2CCCCC2)CC1=O',
            'c1cccc2c1CCCN1C2CN(C(C2CCCCC2)=O)CC1=O',
            'c1cccc2c1C1CN(C(=O)C3CCCCC3)CC(=O)N1CCC2',
            'c1ccc2c(c1)CCCN1C2CN(C(C2CCCCC2)=O)CC1=O',
            'C(=O)(N1CC(=O)N2C(c3ccccc3CCC2)C1)C1CCCCC1',
            'C1CCCCC1C(N1CC(=O)N2CCCc3ccccc3C2C1)=O',
            'C12N(CCCc3ccccc31)C(=O)CN(C(=O)C1CCCCC1)C2'
        ],
        1: [
            'C(Oc1c(OC)cccc1)(c1ccccc1OC(C)=O)=O',
            'C(Oc1c(C(=O)Oc2c(OC)cccc2)cccc1)(=O)C',
            'C(=O)(Oc1ccccc1OC)c1c(OC(=O)C)cccc1',
            'O=C(c1c(OC(C)=O)cccc1)Oc1c(OC)cccc1',
            'c1ccc(OC(C)=O)c(C(Oc2ccccc2OC)=O)c1',
            'c1cc(OC(c2ccccc2OC(=O)C)=O)c(OC)cc1',
            'c1c(OC(C)=O)c(C(Oc2ccccc2OC)=O)ccc1',
            'c1(OC)ccccc1OC(=O)c1c(OC(=O)C)cccc1',
            'c1cccc(OC)c1OC(c1c(OC(C)=O)cccc1)=O',
            'c1(OC(C)=O)c(C(=O)Oc2ccccc2OC)cccc1'
        ],
        2: [
            'c12nc(N)nc(N)c1c(C)c(Cc1cc(OC)ccc1OC)cn2',
            'n1c2c(c(N)nc1N)c(C)c(Cc1cc(OC)ccc1OC)cn2',
            'c1c(OC)c(Cc2c(C)c3c(nc2)nc(N)nc3N)cc(OC)c1',
            'Cc1c(Cc2cc(OC)ccc2OC)cnc2nc(N)nc(N)c21',
            'c1c(Cc2c(OC)ccc(OC)c2)c(C)c2c(n1)nc(N)nc2N',
            'c12c(c(N)nc(N)n1)c(C)c(Cc1c(OC)ccc(OC)c1)cn2',
            'C(c1cnc2c(c(N)nc(N)n2)c1C)c1c(OC)ccc(OC)c1',
            'c1(C)c2c(nc(N)nc2N)ncc1Cc1cc(OC)ccc1OC',
            'c1(OC)c(Cc2cnc3nc(N)nc(N)c3c2C)cc(OC)cc1',
            'c1(N)nc(N)c2c(C)c(Cc3cc(OC)ccc3OC)cnc2n1'
        ]
    }
    return smiles_list[i]




def test_smiles_list(i):
    smiles_list = {
        0: [
            'C1CCC(CC1)C(=O)N2CC3C4=CC=CC=C4CCCN3C(=O)C2',
            'CC1CCCC(C1)C(=O)N2CC3N(CCCc4ccccc34)C(=O)C2',
            'CCC1CCCC(C1)C(=O)N2CC3N(CCCc4ccccc34)C(=O)C2',
            'CCCC1CCCC(C1)C(=O)N2CC3N(CCCc4ccccc34)C(=O)C2',
            'CCCCC1CCCC(C1)C(=O)N2CC3N(CCCc4ccccc34)C(=O)C2',
            'CCCCCC1CCCC(C1)C(=O)N2CC3N(CCCc4ccccc34)C(=O)C2',
            'CCCCCCC1CCCC(C1)C(=O)N2CC3N(CCCc4ccccc34)C(=O)C2',
            'CCCCCCCC1CCCC(C1)C(=O)N2CC3N(CCCc4ccccc34)C(=O)C2',
            'CCCCCCCCC1CCCC(C1)C(=O)N2CC3N(CCCc4ccccc34)C(=O)C2',
            'CCCCCCCCCC1CCCC(C1)C(=O)N2CC3N(CCCc4ccccc34)C(=O)C2',
        ],
        1: [
            'C1=CC=C(C=C1)C(=O)OC2=CC=CC=C2',
            'CC1=CC(=CC=C1)OC(=O)C2=CC=CC=C2',
            'CCC1=CC(=CC=C1)OC(=O)C2=CC=CC=C2',
            'CCCC1=CC(=CC=C1)OC(=O)C2=CC=CC=C2',
            'CCCCC1=CC(=CC=C1)OC(=O)C2=CC=CC=C2',
            'CCCCCC1=CC(=CC=C1)OC(=O)C2=CC=CC=C2',
            'CCCCCCC1=CC(=CC=C1)OC(=O)C2=CC=CC=C2',
            'CCCCCCCC1=CC(=CC=C1)OC(=O)C2=CC=CC=C2',
            'CCCCCCCCC1=CC(=CC=C1)OC(=O)C2=CC=CC=C2',
            'CCCCCCCCCC1=CC(=CC=C1)OC(=O)C2=CC=CC=C2'
        ],
        2: [
            'C(c1ccccc1)c2cnc3ncncc3c2',
            'Cc1cccc(Cc2cnc3ncncc3c2)c1',
            'CCc1cccc(Cc2cnc3ncncc3c2)c1',
            'CCCc1cccc(Cc2cnc3ncncc3c2)c1',
            'CCCCc1cccc(Cc2cnc3ncncc3c2)c1',
            'CCCCCc1cccc(Cc2cnc3ncncc3c2)c1',
            'CCCCCCc1cccc(Cc2cnc3ncncc3c2)c1',
            'CCCCCCCc1cccc(Cc2cnc3ncncc3c2)c1',
            'CCCCCCCCc1cccc(Cc2cnc3ncncc3c2)c1',
            'CCCCCCCCCc1cccc(Cc2cnc3ncncc3c2)c1',
        ]
    }
    return smiles_list[i]


def create_dataset(smiles_list, property_list):
    property_fn = get_property_fn(property_list)
    mols = list(map(get_mol, smiles_list))
    murcko_sca = list(map(murcko_scaffold, mols))
    smi = pd.DataFrame({ 'src': smiles_list,
                         'src_scaffold': murcko_sca })
    props = pd.DataFrame({ f'src_{p}': list(map(fn, mols))
                           for p, fn in property_fn.items() })
    return pd.concat([smi, props], axis=1)


def scatter_plot(x, condition, save_path):
    plt.figure()
    plt.scatter(x[:, 0], x[:, 1], c=condition, s=5)
    plt.colorbar()
    plt.savefig(save_path, dpi=300)

from matplotlib.lines import Line2D

@torch.no_grad()
def encoder_outputs_analysis(args, toklen_data, scaler, SRC, TRG, COND, device):
    # make a save directory

    save_folder = os.path.join(args.infer_path, args.benchmark,
                               'test_encoder', args.model_name)
    os.makedirs(save_folder, exist_ok=True)

    # create a generator for sampling
    
    args.model_path = os.path.join(args.train_path, args.benchmark,
                                   args.model_name, f'model_{args.epoch}.pt')
    generator = get_generator(args, SRC, TRG, toklen_data, scaler, device)

    dp = DataloaderPreparation(device, SRC, TRG,
                               model_type=args.model_type,
                               property_list=args.property_list,
                               use_scaffold=args.use_scaffold)

    # get encoder outputs from training set

    if args.model_type == 'cvaetf':
        train = pd.read_csv("/fileserver-gamma/chaoting/ML/dataset/moses/prepared/train_v0_logP-tPSA-QED.csv")
    elif args.model_type == 'scacvaetfv3':
        train = pd.read_csv("/fileserver-gamma/chaoting/ML/dataset/moses/prepared/train_logP-tPSA-QED.csv")
    
    # TASK1: get the standard deviation of mean and logvar from encoder
    if False:
        get_std_of_encoder_outputs(dp, generator, train, transform=False)


    if False:
        df_data = pd.read_csv("/fileserver-gamma/chaoting/ML/dataset/moses/raw/test_scaffolds.csv")
        df_data = df_data.sample(n=100).reset_index(drop=True)

        _, val = get_encoder_output(generator, df_data,
                                    args.use_scaffold,
                                    args.property_list)
        mu, logvar, std = val

        normal = pd.DataFrame(np.random.normal(0, 1, 10000),
                              columns=['normal distribution'])

        def kde_plot(standard_val, xlabel, figname):
            fig, ax = plt.subplots(nrows=1, ncols=1)
            standard_val.plot.kde(ax=ax, legend=False, linewidth=2,
                                xlim=(-8, 8), alpha=0.6)
            normal.plot.kde(ax=ax, legend=True, linewidth=2.5,
                            xlim=(-8, 8), linestyle='--',
                            bw_method=0.5, color='black')
            ax.set_xlabel(xlabel, fontsize=20)
            ax.set_ylabel('Density', fontsize=20)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            fig.savefig(os.path.join(save_folder, f'{figname}.png'),
                        bbox_inches="tight")

        kde_plot(mu, 'Mean', 'each_mean_of_mean')
        kde_plot(logvar, 'Logvar', 'each_mean_of_logvar')
        kde_plot(std, 'Std', 'each_mean_of_std')


    if False:
        df_data = pd.read_csv("/fileserver-gamma/chaoting/ML/dataset/moses/raw/test_scaffolds.csv")
        df_data = df_data.sample(n=100).reset_index(drop=True)

        val, _ = get_encoder_output(generator, df_data,
                                    args.use_scaffold,
                                    args.property_list)
        mu, logvar, std = val

        normal = pd.DataFrame(np.random.normal(0, 1, 10000),
                              columns=['normal distribution'])

        def kde_plot(standard_val, xlabel, figname):
            fig, ax = plt.subplots(nrows=1, ncols=1)
            standard_val.plot.kde(ax=ax, legend=False, linewidth=2,
                                xlim=(-8, 8), alpha=0.6)
            normal.plot.kde(ax=ax, legend=True, linewidth=2.5,
                            xlim=(-8, 8), linestyle='--',
                            bw_method=0.5, color='black')
            ax.set_xlabel(xlabel, fontsize=20)
            ax.set_ylabel('Density', fontsize=20)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            fig.savefig(os.path.join(save_folder, f'{figname}.png'),
                        bbox_inches="tight")

        kde_plot(mu, 'Mean', 'all_mean')
        kde_plot(logvar, 'Logvar', 'all_logvar')
        kde_plot(std, 'Std', 'all_std')

    #TASK2: plot density of standardize mean and logvar from encoder

    if False:
        df_data = pd.read_csv("/fileserver-gamma/chaoting/ML/dataset/moses/raw/test_scaffolds.csv")
        df_data = df_data.sample(n=100).reset_index(drop=True)

        standard_val, _ = get_encoder_output(generator, df_data,
                                             args.use_scaffold,
                                             args.property_list,
                                             standardize=True)
        standard_mu, standard_logvar, standard_std = standard_val

        normal = pd.DataFrame(np.random.normal(0, 1, 10000),
                              columns=['normal distribution'])

        def kde_plot(standard_val, xlabel, figname):
            fig, ax = plt.subplots(nrows=1, ncols=1)
            standard_val.plot.kde(ax=ax, legend=False, linewidth=2,
                                xlim=(-8, 8), alpha=0.6)
            normal.plot.kde(ax=ax, legend=True, linewidth=2.5,
                            xlim=(-8, 8), linestyle='--',
                            bw_method=0.5, color='black')
            ax.set_xlabel(xlabel, fontsize=20)
            ax.set_ylabel('Density', fontsize=20)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            fig.savefig(os.path.join(save_folder, f'{figname}.png'),
                        bbox_inches="tight")

        kde_plot(standard_mu, 'Standardized Mean', 'standardized_mean')
        kde_plot(standard_logvar, 'Standardized Logvar', 'standardized_logvar')
        kde_plot(standard_std, 'Standardized Std', 'standardized_std')


    # TODO: TASK3: cluster molecules with different scaffolds
    
    if False:
        # scacvae 有 cvae 沒
        n_each = 1000
        train = pd.read_csv("/fileserver-gamma/chaoting/ML/dataset/moses/raw/train.csv")
        scaffolds = {
            'c1ccccc1': 0,
            'O=C(Nc1ccccc1)c1ccccc1': 2,
            'O=C(NCc1ccccc1)c1ccccc1': 3,
            'O=C(COc1ccccc1)Nc1ccccc1': 4,
            'c1ccncc1': 5,
            'O=S(=O)(Nc1ccccc1)c1ccccc1': 6,
            'O=C(c1ccccc1)N1CCCCC1': 7,
            'c1ccc(CNc2ccccc2)cc1': 8,
            'c1ccc(-n2cccn2)cc1': 9,
            'O=C(Cc1ccccc1)Nc1ccccc1': 10,
        }

        train_samples = []
        
        for sca in scaffolds:
            train_samples.append(train[train.scaffold == sca].sample(n=n_each))
        train_samples = pd.concat(train_samples, axis=0).reset_index(drop=True)
        print(train_samples)
        train_samples['type'] = train_samples['scaffold'].apply(lambda x: scaffolds[x])
        
        mu_samples = sample_mu(generator, train_samples,
                               property_list=args.property_list,
                               transform=True,
                               n_samples=len(scaffolds)*n_each,
                               use_scaffold=args.use_scaffold)
        print('umap...')
        reduced_mu = dimension_reduction['umap'](mu_samples)

        print('plot figure...')
        scatter_plot(reduced_mu, train_samples['type'],
                     os.path.join(save_folder, 'train_scaffold-distribution.png'))

    # TODO: TASK4: cluster molecules with different properties

    if False:
        n_each = 1000
        train = pd.read_csv("/fileserver-gamma/chaoting/ML/dataset/moses/raw/train.csv")
        train = train.dropna(subset=['scaffold'])
        logP = [(0, 1), (1,2), (3,4)]
        
        train_samples = []
        for v1, v2 in logP:
            sub = train[(v1 <= train.logP) & (train.logP <= v2)]
            print(v1, v2, len(sub))
            train_samples.append(sub.sample(n=n_each))
        train_samples = pd.concat(train_samples, axis=0).reset_index(drop=True)

        def categorize(x):
            for i, (v1, v2) in enumerate(logP):
                if v1 <= x <= v2:
                    return i
            exit(x)
        train_samples['type'] = train_samples['logP'].apply(
            lambda x: categorize(x))

        mu_samples = sample_mu(generator, train_samples,
                               property_list=args.property_list,
                               transform=True,
                               n_samples=len(logP)*n_each,
                               use_scaffold=args.use_scaffold)
        
        print('umap...')
        reduced_mu = dimension_reduction['umap'](mu_samples)

        print('plot figure...')
        scatter_plot(reduced_mu, train_samples['type'],
                     os.path.join(save_folder, 'train_logP-distribution.png'))
        

    # TASK5: observe the 2d plot of smiles by dimensionality reduction
    print(save_folder)

    reduce_method = dimension_reduction['umap']

    train_loader = dp.get_dataloader(train, batch_size=1, is_train=False)
    train_mu = sample_encoder_outputs(generator, train_loader, transform=False, n_samples=1000)
        
    # get encoder outputs from test set
    for i in range(3):
        smiles_list = test_smiles_list(i)
        # smiles_list = test_can_smiles_list(i)
        # smiles_list = test_smiles_list_encoder_output(i)
        test = create_dataset(smiles_list, args.property_list)
        test.to_csv(os.path.join(save_folder, f'src_{i}.csv'))
    
    for i in range(3):
        test = pd.read_csv(os.path.join(save_folder, f'src_{i}.csv'))
        
        test_mu = []
        test_size = []
        
        for j in range(len(test)):
            test_loader = dp.get_dataloader(test.iloc[[j]], batch_size=1, is_train=False)
            test_mu.append(sample_encoder_outputs(generator, test_loader, transform=True))
            test_size.append(1)
        
        test_mu = np.concatenate(test_mu, axis=0)
        all_mu = np.concatenate((train_mu, test_mu), axis=0)
        
        mu_transformed = reduce_method(all_mu)
    
        n_classes = []
        n_classes.extend([0]*train_mu.shape[0])
        for j in range(len(test_size)):
            n_classes.extend([j+1]*test_size[j])
        
        plt.figure()
        plt.scatter(mu_transformed[-sum(test_size):, 0],
                    mu_transformed[-sum(test_size):, 1],
                    c=n_classes[-sum(test_size):],
                    s=3)
        plt.colorbar()
        plt.savefig(os.path.join(save_folder, f'src_{i}_2d.png'), dpi=300)
        print(os.path.join(save_folder, f'src_{i}_2d.png'))
        
        plt.figure()
        plt.scatter(mu_transformed[:, 0],
                    mu_transformed[:, 1],
                    c=n_classes,
                    s=3)
        plt.colorbar()
        plt.savefig(os.path.join(save_folder, f'all_{i}_2d.png'), dpi=300)
        print(os.path.join(save_folder, f'all_{i}_2d.png'))

    # exit()
    

    # plt.scatter(train_mu_pca[:, 0], train_mu_pca[:, 1],
    #             s=0.5, label='train')
    
    # for i in range(len(test)):
    #     dataloader = dp.get_dataloader(test.iloc[[i]], is_train=False)
    #     test_mu = sample_encoder_outputs(generator, dataloader)
    #     test_mu_pca = reduce_tool.transform(test_mu)
    #     print(test_mu_pca.shape)
    #     plt.scatter(test_mu_pca[:, 0], test_mu_pca[:, 1],
    #                 s=0.5, c=color_map[i], label=f'mu{i}')
    
    # legend = plt.legend()
    # for handle in legend.legendHandles:
    #     handle.set_sizes([2])
    # plt.savefig('3.png')

    
    # plot_1d_distribution(zs, save_folder)
    # plot_2d_distribution(zs, save_folder)


def test_encoder(args, toklen_data, scaler, SRC, TRG, COND, device):
    encoder_outputs_analysis(args, toklen_data, scaler, SRC, TRG, COND, device)

    exit()
    
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
    
    
