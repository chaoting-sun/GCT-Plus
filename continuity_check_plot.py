import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from Evaluation.plot_config import plot_config
import numpy as np


_model_input_to_subpath = {
    'transformer': 'transformer',
    'aug_all-tf': 'transformer_ep25_aug-all',
    'aug-eo-tf': 'transformer_ep25_aug-decoderout',
    'aug-eo-opt0-tf': 'transformer_ep25_aug-decoderout-opt0'
}


def _get_sub_file_folders(query):
    sub_file_folders = []
    sub_path = _model_input_to_subpath[query['model']]
    for ep in query['epoch']:
        sub_file_folders.append(f'{sub_path}_ep{ep}')
    return sub_file_folders
    

def _figure_path(check_on, query_list, method, metric):
    figname = ''
    for i, query in enumerate(query_list):
        if i > 0:
            figname += '-'
        figname += query['model']
        figname += '-'
        if len(query['epoch']) == 1:
            figname += f"{query['epoch'][0]}"
        else:
            figname += f"{query['epoch'][0]}~{query['epoch'][-1]}"
    figname += f'_{check_on}_{method}_{metric}.png'
    
    return os.path.join("./Evaluation", figname)


def _get_legend_name_list(query_list):
    legend_name_list = [] 
    for query in query_list:
        for ep in query['epoch']:
            legend_name_list.append(f"{query['model']} - ep{ep}")
    return legend_name_list


def _get_file_path_list(args, query_list):
    if args.check_on == 'z':
        prop = 'z'
        file_name = 'z1z2_statistics.csv'
    else:
        prop = 'conds'
        file_name = f'{args.check_on}_statistics.csv'
    
    file_path_list = []    
    for i, query in enumerate(query_list):
        sub_file_folders = _get_sub_file_folders(query)
        for each in sub_file_folders:    
            file_path = os.path.join(args.main_folder, each, f"check_{prop}",
                                     f"toklen{args.toklen}", args.decode_algo,
                                     file_name)
            file_path_list.append(file_path)
    return file_path_list
    

def get_metric_results(metric, file_paths, legend_names):
    statistics = {}
    for i, path in enumerate(file_paths):
        if not os.path.exists(path):
            exit(f'File not exists: {path}')
        df = pd.read_csv(path, index_col=[0])
        statistics[legend_names[i]] = df[metric]
    
    return pd.DataFrame(statistics)


def line_plot(dataframe, title, xlabel,
              ylabel, figpath='./test.png'):
    plt.figure(figsize=(10,8))
    
    sns.lineplot(data=dataframe, palette="cubehelix")
    # sns.scatterplot(data=dataframe, palette="cubehelix", s=47)
    
    plt.ylim((0,1))
    plt.title(title, fontsize=30)
    plt.xlabel(xlabel, fontsize=22)
    plt.ylabel(ylabel, fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=18)
    plt.tight_layout()
    plt.savefig(figpath, dpi=200)
    

def plot_on_average(dataframe, title, xlabel, ylabel, figpath):
    print(dataframe)
    
    means = dataframe.mean(axis=0).T
    
    plt.figure(figsize=(10,8))
    means.plot(rot=15)

    plt.ylim((0,1))
    plt.title(title, fontsize=26)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig(figpath, dpi=200)

    
def plot_metric(args, metric, query_list):
    file_path_list = _get_file_path_list(args, query_list)
    legend_name_list = _get_legend_name_list(query_list)
    
    print("get metric results...")
    results = get_metric_results(metric, file_path_list, legend_name_list)
    results['# Steps'] = [i for i in range(len(results))]
    results = results.set_index('# Steps')

    if args.plot_by_steps:
        print("plot by steps...")
        line_plot(dataframe=results,
                  title=f'{metric.capitalize()} by Steps',
                  xlabel='# Steps',
                  ylabel=metric,
                  figpath=_figure_path(args.check_on, query_list, 'line', metric))
    
    if args.plot_on_average:
        print("plot on average...")
        plot_on_average(dataframe=results,
                        title=f'{metric.capitalize()} on Models',
                        xlabel='Models',
                        ylabel=metric,
                        figpath=_figure_path(args.check_on, query_list, 'avg', metric))


def snn_start_value(snn_start): # 越低越好
    product = 1
    for i in range(1, len(snn_start)):
        product *= (snn_start[i] - snn_start[i-1])
    return product**(1.0 / (len(snn_start)-1))
        

def snn_prev_value(snn_prev): # 越高越好
    product = 1
    for i in range(len(snn_prev)):
        product *= snn_prev[i]
    return product**(1.0 / len(snn_prev))
    

# def get_query():
#     # query = [query1, quer2, ...], figname
#     # query_i = { model: ..., epoch: [...] }
#     # model = transformer | aug_all_tf | aug_encoderout_tf

#     query = [
#         # { 'model': 'transformer', 'epoch': np.arange(25, 30) },
#         { 'model': 'transformer', 'epoch': np.arange(21, 26) },
#         { 'model': 'aug-eo-tf', 'epoch': np.arange(26, 36) }
#     ]
#     cond = 'z'

#     config = plot_config(query, figname, cond)
#     return config


def options(parser):
    parser.add_argument('-main_folder', type=str, default='/fileserver-gamma/chaoting/ML/cvae-transformer/Inference')
    parser.add_argument('-decode_algo', type=str, default='greedy')
    parser.add_argument('-toklen', type=str, default=30)
    parser.add_argument('-check_on', type=str, default='logP')

    parser.add_argument('-plot_by_steps', type=bool, default=True)
    parser.add_argument('-plot_on_average', type=bool, default=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    options(parser)
    args = parser.parse_args()
    
    metric = 'snn_previous'
    query = [
        { 'model': 'transformer', 'epoch': np.arange(21, 26) },
        # { 'model': 'transformer', 'epoch': np.arange(21, 26) },
        # { 'model': 'aug-eo-tf', 'epoch': np.arange(26, 36) },
        # { 'model': 'aug-eo-opt0-tf', 'epoch': np.arange(26, 30) }
    ]
    
    # continuity_check_plot(args, config)
    
    plot_metric(args, metric, query)
