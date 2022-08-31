import os
import gc
import sys
import joblib
import argparse
import numpy as np
import pandas as pd
from time import time
from torchtext import data
import seaborn as sns
import matplotlib.pyplot as plt
import dill as pickle
import torch
import plotly.graph_objects as go
from plot import density_plot, density_plot_dict

from Utils import allocate_gpu, get_fields
from Utils.seed import set_seed
from Model.build_model import build_model
from Configuration.config import options
from Utils.dataset import to_dataloader
from Model.modules import create_source_mask
from regression import linear_regression


def produce_tiny_data(original_data_folder, tiny_data_folder, random_state=0):
    train = pd.read_csv(os.path.join(original_data_folder, 'train.csv'))
    train = train.sample(n=10000, random_state=random_state)
    train.to_csv(os.path.join(tiny_data_folder, f'train_{random_state}.csv'))

    validation = pd.read_csv(os.path.join(original_data_folder, 'validation.csv'))
    validation = validation.sample(n=1000, random_state=random_state)
    validation.to_csv(os.path.join(tiny_data_folder, f'validation_{random_state}.csv'))


def get_model(args, SRC_vocab_len, TRG_vocab_len, model_type, decode_type):
    if model_type == 'transformer':
        model_path = 'molGCT/molgct.pt'
    elif args.epoch > 0:
        model_path = os.path.join(args.model_directory, f'model_{args.epoch}.pt')
        # model_path = glob.glob(os.path.join(args.model_directory, 'best_*'))[0]
    else:
        model_path = None
    print("Model path:", model_path)
    return build_model(args, SRC_vocab_len, TRG_vocab_len, model_path)


def generate_encoder_outputs(args, nbatches, model, data_iter, device):
    # including mean and variance
    # return matrix with dimension: (latent_dim, n_samples)

    all_z = np.empty((args.batch_size*nbatches, (args.max_strlen+3), args.latent_dim))
    all_mu = np.empty((args.batch_size*nbatches, (args.max_strlen+3), args.latent_dim))
    all_log_var = np.empty((args.batch_size*nbatches, (args.max_strlen+3), args.latent_dim))

    dataloader = to_dataloader(data_iter,
                               args.conditions,
                               TRG.vocab.stoi['<pad>'],
                               args.max_strlen,
                               device)
    all_n = cur_n = 0

    for i, batch in enumerate(dataloader):
        print(f'{(i+1)} / {nbatches}')

        src_mask = create_source_mask(batch.src, batch.econds)
        z, mu, log_var = model(batch.src, batch.econds, src_mask)
        cur_n = len(batch.src)
     
        all_z[all_n:all_n+cur_n, :, :] = z.cpu().numpy()
        all_mu[all_n:all_n+cur_n, :, :] = mu.cpu().numpy()
        all_log_var[all_n:all_n+cur_n, :, :] = log_var.cpu().numpy()

        all_n += cur_n

    return all_z, all_mu, all_log_var


def find_latent_dim_correlation(args, correlation_folder, mu_name,
                                log_var_name, all_mu, all_log_var):
    def get_corr(all_val, save_path):
        if os.path.exists(save_path):
            return pickle.load(open(save_path, 'rb'))
        else:
            all_val = np.reshape(all_val, (-1, args.latent_dim))
            df_all_val = pd.DataFrame(all_val, columns=[i for i in range(args.latent_dim)])
            all_val_corr = df_all_val.corr()
            pickle.dump(all_val_corr, open(save_path, 'wb'))
            return all_val_corr

    mu_corr = get_corr(all_mu, os.path.join(correlation_folder, f'{mu_name}.pkl'))
    log_var_corr = get_corr(all_log_var, os.path.join(correlation_folder, f'{log_var_name}.pkl'))

    save_correlation_plot(mu_corr, os.path.join(correlation_folder, f'{mu_name}.png'))
    save_correlation_plot(log_var_corr, os.path.join(correlation_folder, f'{log_var_name}.png'))


def find_diff_position_correlation(args, correlation_folder, mu_name, 
                                   log_var_name, all_mu, all_log_var, position):
    # 固定意義（mu/log_var），了解 latent_dim 中不同 dim 之間的相關性

    print('Compute correlation of mu...')
    
    def get_corr(all_val, save_path):
        if os.path.exists(save_path):
            return pickle.load(open(save_path, 'rb'))
        else:
            all_val = pd.DataFrame(all_val[:,:,position], columns=[i for i in range(args.max_strlen+3)])
            all_val_corr = all_val.corr()
            pickle.dump(all_val_corr, open(save_path, 'wb'))
            return all_val_corr

    mu_corr = get_corr(all_mu, os.path.join(correlation_folder, 'pkl',f'{mu_name}.pkl'))
    log_var_corr = get_corr(all_log_var, os.path.join(correlation_folder, 'pkl',f'{log_var_name}.pkl'))
   
    save_correlation_plot(mu_corr, os.path.join(correlation_folder, 'png',f'{mu_name}.png'))
    save_correlation_plot(log_var_corr, os.path.join(correlation_folder, 'png',f'{log_var_name}.png'))

    # print(f'mu - highest/lowest corr val: {mu_corr.max().max():.6f}/{mu_corr.min().min():.6f}') 
    # print(f'var - highest/lowest corr val: {log_var_corr.max().max():.6f}/{log_var_corr.min().min():.6f}')
    return (mu_corr.max().max(), mu_corr.min().min()), (log_var_corr.max().max(), log_var_corr.min().min())


def find_mean_sigma_correlation(args, correlation_folder, file_name, all_mu, all_log_var):
    def get_corr(all_mu, all_log_var, save_path):
        if os.path.exists(save_path):
            return pickle.load(open(save_path, 'rb'))
        else:
            correlation_coeff = []
            for i in range(args.max_strlen+3):
                arr = np.empty((2, len(all_mu)*args.latent_dim))
                arr[0, :] = np.reshape(all_mu[:, i, :], (1, -1))
                arr[1, :] = np.reshape(all_log_var[:, i, :], (1, -1))
                res = np.corrcoef(arr)

                print(res)
                coeff = res[0, 1]
                correlation_coeff.append(coeff)
            pickle.dump(correlation_coeff, open(save_path, 'wb'))
            return correlation_coeff
    
    correlation_coeff = get_corr(all_mu, all_log_var, os.path.join(correlation_folder, f'{file_name}.pkl'))
    save_correlation_bar_plot(correlation_coeff, os.path.join(correlation_folder, f'{file_name}.png'))


def compute_pearson_correlation(matrix):
    correlation = np.corrcoef(matrix)
    return correlation


def save_correlation_plot(corr, fig_name):
    # https://towardsdatascience.com/better-heatmaps-and-correlation-matrix-plots-in-python-41445d0f2bec
    plt.figure()
    ax = sns.heatmap(
        corr,
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True
    )

    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    );
    
    plt.savefig(fig_name)


def save_correlation_bar_plot(corr, fig_path):
    df = pd.DataFrame({
        'dimension': [i for i in range(len(corr))],
        'correlation': corr
    })

    df["color"] = np.where(df["correlation"] < 0, 'orange', 'darkcyan')

    fig = go.Figure()
    fig.add_trace(
        go.Bar(name='correlation',
               x=df['dimension'],
               y=df['correlation'],
               marker_color=df['color'])
    )
    fig.update_layout(
        title={
            'text': 'Correlation of Mean and Sigma',
            'x': 0.5,
            'y': 0.9,
            'xanchor': 'center'
        },
        xaxis_title="character position",
        yaxis_title="correlation coefficient",
        font={ 'size': 16 }
    )
    
    fig.update(layout_yaxis_range=[-1,1])
    fig.update_layout(barmode='stack')
    fig.write_image(fig_path)


if __name__ == '__main__':
    set_seed(seed=0)

    parser = argparse.ArgumentParser()
    parser = options(parser)
    args = parser.parse_args()
    args.batch_size = 512

    args.plot_density_of_encoder_outputs = False
    args.compute_latent_dimensional_correlation = False
    args.compute_string_location_correlation = False
    args.compute_mu_log_var_correlation = True

    original_data_folder = os.path.join(args.data_path, 'aug', f'data_sim{args.similarity:.2f}')
    tiny_data_folder = os.path.join(original_data_folder, 'tiny')
    os.makedirs(tiny_data_folder, exist_ok=True)

    print('-------------------------- Settings --------------------------')
    print(' '.join(f'{k}={v}' for k, v in vars(args).items()))
    print('-------------------------- Settings --------------------------')

    device = allocate_gpu()
    scaler = joblib.load(args.scaler_path)
    fields, SRC, TRG = get_fields(args.conditions, args.field_path)
    toklen_data = pd.read_csv(args.toklen_path)

    model = get_model(args, len(SRC.vocab), len(TRG.vocab), 
                      args.model_type, args.decode_type).to(device)

    # state = (0, 1, 2)
    random_states = (0,1,2)

    for state in random_states:
        if os.path.exists(os.path.join(tiny_data_folder, f'train_{state}.csv')):
            continue
        produce_tiny_data(original_data_folder, tiny_data_folder, random_state=state)

    for state in random_states:
        train_data, valid_data = data.TabularDataset.splits(
            path=tiny_data_folder, train=f'train_{state}.csv', validation=f'validation_{state}.csv', test=None,
            format='csv', fields=fields, skip_header=True)

        train_nbatches = int(np.ceil(len(train_data) / args.batch_size))
        valid_nbatches = int(np.ceil(len(valid_data) / args.batch_size))

        train_iter, valid_iter = data.BucketIterator.splits(
            (train_data, valid_data), batch_sizes=(args.batch_size, args.batch_size),
            sort_key=lambda x: (len(x.src), len(x.trg)))

        del train_data
        del valid_data
        gc.collect()

        model.eval()

        with torch.no_grad():
            train_z, train_mu, train_log_var = generate_encoder_outputs(args, 
                                                                        train_nbatches,
                                                                        model.encode_sample,
                                                                        train_iter, device)

        with torch.no_grad():
            valid_z, valid_mu, valid_log_var = generate_encoder_outputs(args, 
                                                                        valid_nbatches,
                                                                        model.encode_sample,
                                                                        valid_iter, device)

        if args.plot_density_of_encoder_outputs:
            plot_dict = density_plot_dict(xlabel='dimension of latent space', ylabel='density', 
                                        nbins=128, figsize=None, xlim=None, ylim=(-8,8))

            _z = np.reshape(train_z, (-1))
            _mu = np.reshape(train_mu, (-1))
            _log_var = np.reshape(train_log_var, (-1))

            lat_dim = np.tile([i for i in range(args.latent_dim)], len(train_z)*(args.max_strlen+3))
            # idx = np.random.choice(np.arange(len(lat_dim)), 50000, replace=False)

            density_plot(lat_dim, _z, fig_path=f'train_density_z_{state}.png', plot_dict=plot_dict)
            density_plot(lat_dim, _mu, fig_path=f'train_density_mu_{state}.png', plot_dict=plot_dict)
            density_plot(lat_dim, _log_var, fig_path=f'train_density_log_var_{state}.png', plot_dict=plot_dict)

            _z = np.reshape(valid_z, (-1))
            _mu = np.reshape(valid_mu, (-1))
            _log_var = np.reshape(valid_log_var, (-1))

            lat_dim = np.tile([i for i in range(args.latent_dim)], len(valid_z)*(args.max_strlen+3))
            # idx = np.random.choice(np.arange(len(lat_dim)), 10000, replace=False)

            density_plot(lat_dim, _z, fig_path=f'valid_density_{state}.png', plot_dict=plot_dict)
            density_plot(lat_dim, _mu, fig_path=f'valid_density_{state}.png', plot_dict=plot_dict)
            density_plot(lat_dim, _log_var, fig_path=f'valid_density_log_var_{state}.png', plot_dict=plot_dict)

        if args.compute_latent_dimensional_correlation:
            print('--- Correlation among different latent dimensions ---')

            correlation_folder = 'latent_dim_correlation'
            os.makedirs(correlation_folder, exist_ok=True)

            find_latent_dim_correlation(args, correlation_folder, f'train_mu_corr_{state}',
                                        f'train_log_var_corr_{state}', train_mu, train_log_var)
            find_latent_dim_correlation(args, correlation_folder, f'valid_mu_corr_{state}',
                                        f'train_log_var_corr_{state}', train_mu, train_log_var)

        if args.compute_string_location_correlation:
            print("--- Correlation between different string positions ---")

            correlation_folder = 'diff_pos_correlation'
            os.makedirs(os.path.join(correlation_folder, 'pkl'), exist_ok=True)
            os.makedirs(os.path.join(correlation_folder, 'png'), exist_ok=True)

            positions = tuple(i for i in range(args.max_strlen+3))
            
            # for pos in positions:


            train_corr = None
            for pos in positions:
                mu_corr, log_var_corr = find_diff_position_correlation(args, correlation_folder,
                                                                      f'train_mu_{pos}_{state}', 
                                                                      f'train_log_var_{pos}_{state}',
                                                                      train_mu, train_log_var, pos)
                print(mu_corr, log_var_corr)
                _corr = pd.DataFrame({
                    'mu_corr_max': [mu_corr[0]], 'mu_corr_min': [mu_corr[1]],
                    'log_var_corr_max': [log_var_corr[0]], 'log_var_corr_min': [log_var_corr[1]]
                })
                train_corr = pd.concat([train_corr, _corr], axis=0)
            train_corr.to_csv(os.path.join(correlation_folder, 'train_corr.csv'))

            valid_corr = None
            for pos in positions:
                mu_corr, log_var_corr = find_diff_position_correlation(args, correlation_folder,
                                                                    f'validation_mu_{pos}_{state}', 
                                                                    f'validation_log_var_{pos}_{state}',
                                                                    valid_mu, valid_log_var, pos)
                _corr = pd.DataFrame({
                    'mu_corr_max': [mu_corr[0]], 'mu_corr_min': [mu_corr[1]],
                    'log_var_corr_max': [log_var_corr[0]], 'log_var_corr_min': [log_var_corr[1]]
                })
                valid_corr = pd.concat([valid_corr, _corr], axis=0)
            valid_corr.to_csv(os.path.join(correlation_folder, 'valid_corr.csv'))

        if args.compute_mu_log_var_correlation:
            print("--- Correlation between mean and std ---")

            correlation_folder = 'mu_log_var_correlation'
            os.makedirs(correlation_folder, exist_ok=True)

            print('train:')
            for i in range(args.max_strlen+3):
                _mu = np.reshape(train_mu[:, i, :], (-1,))
                _log_var = np.reshape(train_log_var[:, i, :], (-1,))
                linear_regression(_mu, _log_var)

            print('validation:')
            for i in range(args.max_strlen+3):
                _mu = np.reshape(valid_mu[:, i, :], (-1,))
                _log_var = np.reshape(valid_log_var[:, i, :], (-1,))
                linear_regression(_mu, _log_var)

            find_mean_sigma_correlation(args, correlation_folder, f'train_corr_{state}', train_mu, train_log_var)
            find_mean_sigma_correlation(args, correlation_folder, f'validation_corr_{state}', valid_mu, valid_log_var)
