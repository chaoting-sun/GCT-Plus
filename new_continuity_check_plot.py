import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from Evaluation.plot_config import plot_config
import numpy as np


_model_input_to_subpath = {
    'tf': 'transformer',
    'all-tf': 'transformer_ep25_aug-all',
    'eo-tf': 'transformer_ep25_aug-decoderout',
    'eo-opt0-tf': 'transformer_ep25_aug-decoderout-opt0',
    'eo-adam-tf': 'transformer_ep25_aug-decoderout-adam',
    'eo-adam_ori-tf': 'transformer_ep25_aug-decoderout',
    'eo-adagrad-tf': 'transformer_ep25_aug-decoderout-adagrad',
    'eo-rmsprop-tf': 'transformer_ep25_aug-decoderout-rmsprop',
}

name_convert = {
    'transformer1': 'CVAE-TF1',
    'transformer2': 'CVAE-TF2',
    'transformer3': 'CVAE-TF3',
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
    print('saved in figname:', figname)
    return os.path.join("./", figname)


def _get_legend_name_list(query_list):
    legend_name_list = [] 
    for query in query_list:
        legend = name_convert[query['model']]
        for ep in query['epoch']:
            legend_name_list.append(f"{legend} {ep}")
    return legend_name_list


def _get_file_path_list(args, query_list):
    if args.check_on == 'z':
        suffix = 'z'
    else:
        suffix = 'conds'
        
    file_path_list = []    
    for i, query in enumerate(query_list):
        model = query['model']
        epochs = query['epoch']
        
        for ep in epochs:
            file_path_list.append(os.path.join(args.main_folder,
                                               f'check_{suffix}',
                                               model, str(ep),
                                               f'{args.check_on}_statistics.csv'))
    return file_path_list
    

def get_metric_results(metric, file_paths, legend_names):
    statistics = {}
    for i, path in enumerate(file_paths):
        if not os.path.exists(path):
            exit(f'File not exists: {path}')
        df = pd.read_csv(path, index_col=[0])
        statistics[legend_names[i]] = df[metric]
    
    statistics = pd.DataFrame(statistics)
    statistics['# Steps'] = [i for i in range(len(statistics))]
    statistics = statistics.set_index('# Steps')
    return statistics


def line_plot(dataframe, title, xlabel,
              ylabel, figpath='./test.png'):
    plt.figure(figsize=(6,5))

    ax = sns.lineplot(data=dataframe, palette="cubehelix")
    ax.get_legend().remove()
    # sns.move_legend(ax, "lower left")
    # sns.scatterplot(data=dataframe, palette="cubehelix", s=47)
    
    plt.ylim((0,1))
    # plt.title(title, fontsize=24)
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=13)
    plt.tight_layout()
    print('line plot:', figpath)
    plt.savefig(figpath, dpi=200)
    

def plot_on_average(dataframe, title, xlabel, ylabel, figpath):
    print(dataframe)
    
    means = dataframe.mean(axis=0).T
    
    plt.figure()
    means.plot(figsize=(6, 5), marker='o', rot=15)

    plt.ylim((0,1))
    # plt.title(title, fontsize=24)
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig(figpath, dpi=200)

    
def plot_metric(args, metric, query_list):
    file_path_list = _get_file_path_list(args, query_list)
    legend_name_list = _get_legend_name_list(query_list)
    
    print("get metric results...")
    results = get_metric_results(metric, file_path_list, legend_name_list)

    if args.plot_by_steps:
        print("plot by steps...")
        
        line_plot(dataframe=results,
                  title=f'{metric.capitalize()} by Steps ({args.check_on})',
                  xlabel='# Steps',
                  ylabel='SNN_start',
                  figpath=_figure_path(args.check_on, query_list, 'line', metric)
                  )
    
    if args.plot_on_average:
        print("plot on average...")
        plot_on_average(dataframe=results,
                        title=f'{metric.capitalize()} on Models ({args.check_on})',
                        xlabel='Models',
                        ylabel='SNN_start',
                        figpath=_figure_path(args.check_on, query_list, 'avg', metric)
                        )

def snn_start_index(snn_start):
    # 越小越好 (smooth)
    slopes = []
    for i in range(len(snn_start)):
        slopes.append(snn_start[i+1] - snn_start[i])
    return np.std(slopes)
    

def snn_prev_index(snn_prev):
    # 越高越好
    return np.average(snn_prev)

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
    parser.add_argument('-main_folder', type=str, default='/fileserver-gamma/chaoting/ML/cvae-transformer/Inference-Results')
    parser.add_argument('-decode_algo', type=str, default='greedy')
    parser.add_argument('-toklen', type=str, default=30)
    parser.add_argument('-check_on', type=str, default='QED')

    parser.add_argument('-plot_by_steps', type=bool, default=True)
    parser.add_argument('-plot_on_average', type=bool, default=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    options(parser)
    args = parser.parse_args()
    
    metric = 'snn_start'
    query = [
        { 'model': 'transformer2', 'epoch': np.arange(16, 25+1) },
        # { 'model': 'tf', 'epoch': np.arange(25, 31) },
        # { 'model': 'all-tf', 'epoch': np.arange(26, 31) },
        # { 'model': 'aug-eo-tf', 'epoch': np.arange(26, 36) },
        # { 'model': 'eo-adam_ori-tf', 'epoch': np.arange(26, 31) }
        # { 'model': 'eo-adagrad-tf', 'epoch': np.arange(26, 31) }
        # { 'model': 'eo-rmsprop-tf', 'epoch': np.arange(26, 31) }
    ]
    
    # continuity_check_plot(args, config)
    plot_metric(args, metric, query)
