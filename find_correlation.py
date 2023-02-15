import os
import gc
from statistics import StatisticsError
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


def sample_and_store_data(data_type, input_path, output_path, n_samples, random_state):
    data = pd.read_csv(os.path.join(input_path, f'{data_type}.csv'))
    data = data.sample(n=n_samples, random_state=random_state)
    data.to_csv(os.path.join(output_path, f'{data_type}_{random_state}.csv'))


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


def generate_encoder_outputs(model, data_iter, conditions, encode_type, batch_size,
                             nbatches, latent_dim, max_strlen, TRG, device):
    # including mean and variance
    # return matrix with dimension: (latent_dim, n_samples)

    all_z = np.empty((batch_size*nbatches, max_strlen, latent_dim))
    all_mu = np.empty((batch_size*nbatches, max_strlen, latent_dim))
    all_log_var = np.empty((batch_size*nbatches, max_strlen, latent_dim))

    encoder = getattr(model, args.encode_type)
    dataloader = to_dataloader(data_iter,
                               conditions,
                               TRG.vocab.stoi['<pad>'],
                               max_strlen,
                               device)
    all_n = cur_n = 0

    for i, batch in enumerate(dataloader):
        print(f'{(i+1)} / {nbatches}')

        src_mask = create_source_mask(batch.src, 
                                      TRG.vocab.stoi['<pad>'],
                                      batch.econds)

        if encode_type == 'encode_sample':
            z, mu, log_var = encoder(batch.src,
                                     batch.econds,
                                     src_mask)
        elif encode_type == 'encode_sample_mlp_sample':
            z, mu, log_var = encoder(batch.src,
                                     batch.econds,
                                     batch.mconds,
                                     src_mask)

        cur_n = len(batch.src)
     
        all_z[all_n:all_n+cur_n, :, :] = z.cpu().numpy()
        all_mu[all_n:all_n+cur_n, :, :] = mu.cpu().numpy()
        all_log_var[all_n:all_n+cur_n, :, :] = log_var.cpu().numpy()

        all_n += cur_n

    return all_z, all_mu, all_log_var


def plot_latent_space_distribution(data_type, mu, log_var, z, n_samples, 
                                   latent_dim, max_strlen, state, correlation_path):
    os.makedirs(correlation_path, exist_ok=True)
    assert mu.shape == log_var.shape and mu.shape == z.shape

    plot_dict = density_plot_dict(xlabel='dimension of latent space', ylabel='position', 
                                  nbins=128, figsize=None, xlim=None, ylim=(-15,15))

    lat_dim = np.tile([i for i in range(latent_dim)], len(z)*max_strlen)
    idx = np.random.choice(np.arange(len(lat_dim)), n_samples, replace=False)

    density_plot(lat_dim[idx], np.reshape(mu, (-1))[idx],
                 fig_path=os.path.join(correlation_path, f'{data_type}_density_mu_{state}.png'), 
                 plot_dict=plot_dict)
    density_plot(lat_dim[idx], np.reshape(log_var, (-1))[idx],
                 fig_path=os.path.join(correlation_path, f'{data_type}_density_log_var_{state}.png'),
                 plot_dict=plot_dict)
    density_plot(lat_dim[idx], np.reshape(z, (-1))[idx],
                 fig_path=os.path.join(correlation_path, f'{data_type}_density_z_{state}.png'),
                 plot_dict=plot_dict)


def save_correlation_plot(corr, fig_name):
    # https://towardsdatascience.com/better-heatmaps-and-correlation-matrix-plots-in-python-41445d0f2bec
    plt.figure()
    ax = sns.heatmap(
        corr,
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True
    )
    ax.set_xlabel('dimension of latent space', fontsize=12)
    ax.set_ylabel('dimension of latent space', fontsize=12)

    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    );
    
    plt.savefig(fig_name, bbox_inches="tight")


def find_latent_dim_correlation(data_type, all_mu, all_log_var,
                                latent_dim, state, correlation_path):
    assert all_mu.shape == all_log_var.shape
    for ext in ('png', 'pkl'):
        os.makedirs(os.path.join(correlation_path, ext), exist_ok=True)

    def get_corr(all_val, save_path):
        if os.path.exists(save_path):
            return pickle.load(open(save_path, 'rb'))
        else:
            all_val = np.reshape(all_val, (-1, latent_dim))
            df_all_val = pd.DataFrame(all_val, columns=[i for i in range(latent_dim)])
            all_val_corr = df_all_val.corr()
            pickle.dump(all_val_corr, open(save_path, 'wb'))
            return all_val_corr

    mu_corr = get_corr(all_mu, os.path.join(correlation_path,
                       'pkl', f'{data_type}_mu_corr_{state}.pkl'))
    log_var_corr = get_corr(all_log_var, os.path.join(correlation_path,
                            'pkl', f'{data_type}_log_var_corr_{state}.pkl'))

    save_correlation_plot(mu_corr, os.path.join(correlation_path,
                         'png', f'{data_type}_mu_corr_{state}.png'))
    save_correlation_plot(log_var_corr, os.path.join(correlation_path, 
                          'png', f'{data_type}_log_var_corr_{state}.png'))


def find_all_position_correlation(data_type, all_mu, all_log_var, 
                                  max_strlen, state, correlation_path):
    for ext in ('png', 'pkl'):
        os.makedirs(os.path.join(correlation_path, ext), exist_ok=True)

    def get_corr(all_values, position, save_path):
        if os.path.exists(save_path):
            return pickle.load(open(save_path, 'rb'))
        else:
            all_values = pd.DataFrame(all_values[:,:,position],  
                                      columns=[i for i in range(max_strlen)])
            corr = all_values.corr()
            pickle.dump(corr, open(save_path, 'wb'))
            return corr

    positions = tuple(i for i in range(max_strlen))
    
    all_corr = None
    
    for p in positions:
        mu_corr = get_corr(all_mu, p, os.path.join(correlation_path, 
                           'pkl', f'{data_type}_mu_{p}_{state}.pkl'))
        log_var_corr = get_corr(all_log_var, p, os.path.join(correlation_path,
                                'pkl', f'{data_type}_log_var_{p}_{state}.pkl'))

        corr = pd.DataFrame({ 'mu_corr_max': [mu_corr.max().max()],
                              'mu_corr_min': [mu_corr.min().min()],
                              'log_var_corr_max': [log_var_corr.max().max()], 
                              'log_var_corr_min': [log_var_corr.min().min()] 
                            })
        all_corr = pd.concat([all_corr, corr], axis=0)

        save_correlation_plot(mu_corr, os.path.join(correlation_path,
                            'png', f'{data_type}_mu_{p}_{state}.png'))
        save_correlation_plot(log_var_corr, os.path.join(correlation_path,
                            'png',f'{data_type}_log_var_{p}_{state}.png'))

    all_corr.to_csv(os.path.join(correlation_path, f'{data_type}_corr_{state}.csv'))


def find_mean_sigma_correlation(data_type, all_mu, all_log_var, latent_dim,
                                max_strlen, state, correlation_path):
    if os.path.exists(os.path.join(correlation_path, 'png', f'{data_type}_corr_{state}.png')):
        return

    for ext in ('png', 'pkl'):
        os.makedirs(os.path.join(correlation_path, ext), exist_ok=True)

    correlation_coeff = []
    for i in range(max_strlen):
        arr = np.empty((2, len(all_mu)*latent_dim))
        arr[0, :] = np.reshape(all_mu[:, i, :], (1, -1))
        arr[1, :] = np.reshape(all_log_var[:, i, :], (1, -1))
        res = np.corrcoef(arr)

        coeff = res[0, 1]
        correlation_coeff.append(coeff)
    pickle.dump(correlation_coeff, open(os.path.join(correlation_path, 
                'pkl', f'{data_type}_corr_{state}.pkl'), 'wb'))

    save_correlation_bar_plot(correlation_coeff, os.path.join(correlation_path,
                              'png', f'{data_type}_corr_{state}.png'))


def compute_pearson_correlation(matrix):
    correlation = np.corrcoef(matrix)
    return correlation


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


def multi_faceted_analysis(args, model, device, fields, TRG):
    ################## Settings ##################
    plot_density_of_encoder_outputs = True
    compute_latent_dimensional_correlation = True
    compute_string_location_correlation = True
    compute_mu_log_var_correlation = True

    random_states = (0,1,2)
    ##############################################

    data_path = os.path.join(args.data_path, 'aug', 'data_sim1.00')
    results_path = os.path.join(data_path, 'tiny')
    os.makedirs(results_path, exist_ok=True)

    for state in random_states:
        sample_and_store_data('train', data_path, results_path, 10000, state)
        sample_and_store_data('validation', data_path, results_path, 1000, state)

    for state in random_states:
        train_data, valid_data = data.TabularDataset.splits(
            path=results_path, train=f'train_{state}.csv', 
            validation=f'validation_{state}.csv', test=None,
            format='csv', fields=fields, skip_header=True
        )

        train_iter, valid_iter = data.BucketIterator.splits(
            (train_data, valid_data), 
            batch_sizes=(args.batch_size, args.batch_size),
            sort_key=lambda x: (len(x.src), len(x.trg))
        )

        train_nbatches = int(np.ceil(len(train_data) / args.batch_size))
        valid_nbatches = int(np.ceil(len(valid_data) / args.batch_size))

        del train_data
        del valid_data
        gc.collect()

        model.eval()

        print("Producing encoder outputs of training/validation set...")
        with torch.no_grad():
            train_z, train_mu, train_log_var = generate_encoder_outputs(model,
                                                                        train_iter,
                                                                        args.conditions,
                                                                        args.encode_type,
                                                                        args.batch_size,
                                                                        train_nbatches,
                                                                        args.latent_dim,
                                                                        args.max_strlen,
                                                                        TRG,
                                                                        device)

            valid_z, valid_mu, valid_log_var = generate_encoder_outputs(model,
                                                                        valid_iter,
                                                                        args.conditions,
                                                                        args.encode_type,
                                                                        args.batch_size,
                                                                        valid_nbatches,
                                                                        args.latent_dim,
                                                                        args.max_strlen,
                                                                        TRG,
                                                                        device)

        if plot_density_of_encoder_outputs:
            print('Plot distribution of the encoder outputs...')
            plot_latent_space_distribution('train', train_mu, train_log_var, train_z, 
                                           100000, args.latent_dim, args.max_strlen, state,
                                           correlation_path=os.path.join(args.storage_path, 'density_distribution'))

            plot_latent_space_distribution('validation', valid_mu, valid_log_var, valid_z,
                                           10000, args.latent_dim, args.max_strlen, state,
                                           correlation_path=os.path.join(args.storage_path, 'density_distribution'))

        if compute_latent_dimensional_correlation:
            print("Find correlation among all latent space dimensions...")
            find_latent_dim_correlation('train', train_mu, train_log_var, args.latent_dim, state, 
                                        correlation_path=os.path.join(args.storage_path, 'latent_dim_correlation'))
            find_latent_dim_correlation('validation', valid_mu, valid_log_var, args.latent_dim, state, 
                                        correlation_path=os.path.join(args.storage_path, 'latent_dim_correlation'))

        if compute_string_location_correlation:
            print("Find correlation among all string positions...")
            find_all_position_correlation('train', train_mu, train_log_var, args.max_strlen, state,
                                          correlation_path=os.path.join(args.storage_path, 'diff_pos_correlation'))
            find_all_position_correlation('validation', valid_mu, valid_log_var, args.max_strlen, state,
                                          correlation_path=os.path.join(args.storage_path, 'diff_pos_correlation'))

        if compute_mu_log_var_correlation:
            print("Find correlation between mean and std...")
            find_mean_sigma_correlation('train', train_mu, train_log_var, args.latent_dim, args.max_strlen, state, 
                                        correlation_path=os.path.join(args.storage_path, 'mu_log_var_correlation'))
            find_mean_sigma_correlation('validation', valid_mu, valid_log_var, args.latent_dim, args.max_strlen, state, 
                                        correlation_path=os.path.join(args.storage_path, 'mu_log_var_correlation'))

            print("Do linear regression between mu and log_var of the training set...")
            for i in range(args.max_strlen):
                _mu = np.reshape(train_mu[:, i, :], (-1,))
                _log_var = np.reshape(train_log_var[:, i, :], (-1,))
                linear_regression(_mu, _log_var)

            print("Do linear regression between mu and log_var of the validation set...")
            for i in range(args.max_strlen):
                _mu = np.reshape(valid_mu[:, i, :], (-1,))
                _log_var = np.reshape(valid_log_var[:, i, :], (-1,))
                linear_regression(_mu, _log_var)


def test_same_len_sequence(args, model, device, fields, TRG):
    data_path = os.path.join(args.data_path, 'aug', 'data_sim1.00')
    results_path = os.path.join(data_path, 'same_length')
    os.makedirs(results_path, exist_ok=True)

    random_states = (0,1,2)
    chosen_str_len = 30
    args.max_strlen = chosen_str_len + len(args.conditions)

    print(f"Sampling data from {data_path}...")
    samples = pd.read_csv(os.path.join(data_path, f'train.csv'))
    samples = samples.loc[samples['src'].str.len() == chosen_str_len]
    for state in random_states:
        if os.path.exists(os.path.join(results_path, f'train_{state}.csv')):
            continue
        samples = samples.sample(n=1000, random_state=state)
        samples.to_csv(os.path.join(results_path, f'train_{state}.csv'))

    for state in random_states:
        print("Getting the iterator...")
        train_data = data.TabularDataset(path=os.path.join(results_path, f'train_{state}.csv'),
                                         format='csv', fields=fields, skip_header=True)
        train_iter = data.BucketIterator(train_data, batch_size=args.batch_size)

        train_nbatches = int(np.ceil(len(train_data) / args.batch_size))

        model.eval()

        with torch.no_grad():
            train_z, train_mu, train_log_var = generate_encoder_outputs(model,
                                                                        train_iter,
                                                                        args.conditions,
                                                                        args.encode_type,
                                                                        args.batch_size,
                                                                        train_nbatches,
                                                                        args.latent_dim,
                                                                        args.max_strlen,
                                                                        TRG,
                                                                        device)
        print('Plot distribution of the encoder outputs...')
        plot_latent_space_distribution('train', train_mu, train_log_var, train_z, 
                                       10000, args.latent_dim, args.max_strlen, state,
                                       correlation_path=os.path.join(args.storage_path, 'density_distribution'))

        print("Find correlation among all latent space dimensions...")
        find_latent_dim_correlation('train', train_mu, train_log_var, args.latent_dim, state, 
                                    correlation_path=os.path.join(args.storage_path, 'latent_dim_correlation'))

        print("Find correlation among all string positions...")
        find_all_position_correlation('train', train_mu, train_log_var, args.max_strlen, state,
                                        correlation_path=os.path.join(args.storage_path, 'diff_pos_correlation'))

        print("Find correlation between mean and std...")
        find_mean_sigma_correlation('train', train_mu, train_log_var, args.latent_dim, args.max_strlen, state, 
                                    correlation_path=os.path.join(args.storage_path, 'mu_log_var_correlation'))

        print("Do linear regression between mu and log_var...")
        for i in range(args.max_strlen):
            _mu = np.reshape(train_mu[:, i, :], (-1,))
            _log_var = np.reshape(train_log_var[:, i, :], (-1,))
            linear_regression(_mu, _log_var)


if __name__ == '__main__':
    set_seed(seed=0)

    parser = argparse.ArgumentParser()
    parser = options(parser)
    args = parser.parse_args()
    args.batch_size = 512

    print('-------------------------- Settings --------------------------')
    print(' '.join(f'{k}={v}' for k, v in vars(args).items()))
    print('-------------------------- Settings --------------------------')

    device = allocate_gpu()
    fields, SRC, TRG = get_fields(args.conditions, args.field_path)
    toklen_data = pd.read_csv(args.toklen_path)

    model = get_model(args, len(SRC.vocab), len(TRG.vocab), 
                      args.model_type, args.decode_type).to(device)

    choice = 1

    if choice == 1:
        multi_faceted_analysis(args, model, device, fields, TRG)
    elif choice == 2:
        test_same_len_sequence(args, model, device, fields, TRG)