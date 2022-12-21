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
    'aug_all-tf': 'transformer_ep25_aug-all',
    'aug-eo-tf-ori': 'transformer_ep25_aug-decoderout',
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
    
    statistics = pd.DataFrame(statistics)
    statistics['# Steps'] = [i for i in range(len(statistics))]
    statistics = statistics.set_index('# Steps')
    return statistics


def line_plot(dataframe, title, xlabel,
              ylabel, figpath='./test.png'):
    plt.figure(figsize=(10,7))
    
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
    

def df_plot(df, title, xlabel, ylabel, figpath):
    plt.figure()
    
    col_names = df.columns.values
    df.plot(figsize=(6, 5), color={ col_names[0]: "#ff004c", col_names[1]: "#235196" })

    plt.ylim((0,1))
    plt.title(title, fontsize=24)
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=18)
    plt.tight_layout()
    plt.savefig(figpath, dpi=200)   

    
# def plot_metric(args, metric, query_list):
#     file_path_list = _get_file_path_list(args, query_list)
#     legend_name_list = _get_legend_name_list(query_list)
    
#     print("get metric results...")
#     results = get_metric_results(metric, file_path_list, legend_name_list)

#     if args.plot_by_steps:
#         print("plot by steps...")
        
#         line_plot(dataframe=results,
#                   title=f'{metric.capitalize()} by Steps ({args.check_on})',
#                   xlabel='# Steps',
#                   ylabel=metric,
#                   figpath=_figure_path(args.check_on, query_list, 'line', metric)
#                   )
    
#     if args.plot_on_average:
#         print("plot on average...")
#         plot_on_average(dataframe=results,
#                         title=f'{metric.capitalize()} on Models ({args.check_on})',
#                         xlabel='Models',
#                         ylabel=metric,
#                         figpath=_figure_path(args.check_on, query_list, 'avg', metric)
#                         )


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


def get_avg(path_list, suffix_file_path, metric):
    avg_metric = []
    for path in path_list:
        df = pd.read_csv(os.path.join(path, suffix_file_path))
        avg_metric.append(df[metric].mean())
    return avg_metric


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


model_folder = "/fileserver-gamma/chaoting/ML/cvae-transformer/Inference/"


model_subpath = {
    'tf'                : 'transformer',
    'aug_all-tf'        : 'transformer_ep25_aug-all',
    'aug-eo-tf-adam_ori': 'transformer_ep25_aug-decoderout',
    'aug-eo-opt0-tf'    : 'transformer_ep25_aug-decoderout-opt0',
    'aug-eo-tf-adagrad' : 'transformer_ep25_aug-decoderout-adagrad',
    'aug-eo-tf-rmsprop' : 'transformer_ep25_aug-decoderout-rmsprop',
}


def get_prefix_path_list(query_model_epoch):
    print('function - get_prefix_path_list')
    prefix_model_path_list = []
    
    for model_epoch in query_model_epoch:
        model = list(model_epoch.keys())[0]
        subpath = model_subpath[model]        
        epoch = model_epoch[model]

        for ep in epoch:
            prefix_model_path_list.append(os.path.join(model_folder, f'{subpath}_ep{ep}'))
    return prefix_model_path_list


def get_suffix_path(check_on, toklen, decode_algo):
    print('function - get_suffix_path')
    if check_on == 'z':
        check_suffix, stat_prefix = 'z', 'z1z2'
    else:
        check_suffix, stat_prefix = 'conds', 'logP'
    suffix_path = os.path.join(f'check_{check_suffix}',
                               f'toklen{toklen}',
                               decode_algo,
                               f'{stat_prefix}_statistics.csv')
    return suffix_path


def get_data_path(query, check_on, toklen, decode_algo):
    print('function - get_data_path')
    prefix_model_path_list = get_prefix_path_list(query)
    suffix_path            = get_suffix_path(check_on, toklen, decode_algo)

    data_path = []
    for prefix_path in prefix_model_path_list:
        data_path.append(os.path.join(prefix_path, suffix_path))
        print(data_path[-1])
    return data_path


def get_legend(query_list):
    legend_list = [] 
    for query in query_list:
        model = list(query.keys())[0]
        epoch = query[model]
        for ep in epoch:
            legend_list.append(f"{model}-ep{ep}")
    return legend_list


def calc_mean(df, metric):
    return df[metric].mean()


def calc_slope_std(df, metric):
    slope_list = []
    for i in range(len(df)-1):
        slope = df[metric].iloc[i+1] - df[metric].iloc[i]
        slope_list.append(slope)
    return np.std(slope_list)


def get_data_metric(model_path, metric, method_fcn):
    metric_list = []
    for path in model_path:
        df = pd.read_csv(path)
        value = method_fcn(df, metric)
        metric_list.append(value)
    return metric_list


def get_epoch_list(query_list):
    epoch_list = []
    for query in query_list:
        epoch_list.extend(list(query.values())[0])
    return epoch_list


def plot_metric_every_epoch(metric_epoch_list, lengend_list, figpath, ylim=(0,1)):
    colors = ["#065cd4", "#0e32c4", "#770cb0", "#ff004c", "#b5070d"]
    markers = ['o', '^', 's', 'x', 'v']

    ep_label, met_label = metric_epoch_list[0].columns

    plt.figure(figsize=(6, 5))
    axes = []
    for i, metric_epoch in enumerate(metric_epoch_list):
        ax = plt.plot(metric_epoch[ep_label], metric_epoch[met_label], color=colors[i], marker=markers[i], linestyle='--')
        axes.append(ax[0])

    plt.ylim(ylim)
    plt.xlabel(ep_label, fontsize=17)
    plt.ylabel(met_label, fontsize=17)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(axes, lengend_list, fontsize=13)

    plt.tight_layout()
    plt.savefig(figpath, dpi=200)
    print("save figure path:", figpath)


import matplotlib.gridspec as gridspec


def plot_metric_every_epoch_one_figure(metric_epoch_list, lengend_list, figpath, ylim=(0,1)):
    colors = ["#065cd4", "#0e32c4", "#770cb0", "#ff004c", "#b5070d"]
    markers = ['o', '^', 's', 'x', 'v']

    ep_label, met_label = metric_epoch_list[0].columns


    # nrows, ncols = 4, 3
    # gs = gridspec.GridSpec(nrows, ncols)
    
    # for r in nrows:
    #     for c in ncols:
    #         ax = plt.subplot(gs[])
    

    fig, axes = plt.subplots(3, 2)
    
    plt.figure(figsize=(6, 5))
    axes = []
    for i, metric_epoch in enumerate(metric_epoch_list):
        ax = plt.plot(metric_epoch[ep_label], metric_epoch[met_label], color=colors[i], marker=markers[i], linestyle='--')
        axes.append(ax[0])

    plt.ylim(ylim)
    plt.xlabel(ep_label, fontsize=17)
    plt.ylabel(met_label, fontsize=17)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(axes, lengend_list, fontsize=13)

    plt.tight_layout()
    plt.savefig("./.png", dpi=200)
    print("save figure path:", figpath)
    

method_dict = {
    'mean'      : calc_mean,
    'slope_std' : calc_slope_std,
}


# def options(parser):
#     parser.add_argument('-main_folder', type=str, default='/fileserver-gamma/chaoting/ML/cvae-transformer/Inference')
#     parser.add_argument('-decode_algo', type=str, default='greedy')
#     parser.add_argument('-toklen', type=str, default=30)
#     parser.add_argument('-check_on', type=str, default='QED')

#     parser.add_argument('-plot_by_steps', type=bool, default=True)
#     parser.add_argument('-plot_on_average', type=bool, default=True)


query1_model_epoch = [
    { 'tf': [21,22,23,24,25,26,27,28,29,30] }
]


query2_model_epoch = [  
    # { 'tf': [] },
    # { 'aug-eo-tf-adam_ori': [26,27,28,29,30] }
    # { 'aug-eo-tf-adagrad': [26,27,28,29,30] }
    { 'aug-eo-tf-rmsprop': [26,27,28,29,30] }
]



def get_data_path_metric(query, check_on, toklen, decode_algo, method, metric):
    method_fcn = method_dict[method]
    
    model_path = get_data_path(query, check_on, toklen, decode_algo)
    model_metric = get_data_metric(model_path, metric, method_fcn)
    epoch_list = get_epoch_list(query)

    epoch_metric = pd.DataFrame({
        '# Epoch'                                        : epoch_list,
        f'{method.capitalize()} of {metric} ({check_on})': model_metric
    })
    return epoch_metric


def plot_all_figures(args):
    metric_method = {
        "snn_start" : 'slope_std',
        "uniqueness": 'mean',
        "int_div"   : 'mean'
    }

    check_property = {
        'check_z'    : ['z'],
        'check_conds': ['logP', 'tPSA', 'QED'],
    }

    def get_epoch_metric(model_name, model_epoch, metric, method):
        method_fcn = method_dict[method]
        model_path = []
        for epoch in model_epoch:
            model_path.append(os.path.join(args.main_folder,
                                           check_on,
                                           model_name,
                                           str(epoch),
                                           f'{prop}_statistics.csv'
                                         ))
        metric_list = get_data_metric(model_path, metric, method_fcn)

        epoch_metric = pd.DataFrame({
            "# Epoch": model_epoch,
            f"{method.capitalize()} of {metric} ({prop})": metric_list
        })
        return epoch_metric

    for check_on in ('check_z', 'check_conds'):
        for metric, method in metric_method.items():
            method_fcn = method_dict[method]

            for prop in check_property[check_on]:
                ref_epoch_metric = get_epoch_metric(args.ref_model_name,
                                                    args.ref_model_epoch,
                                                    metric, method)
                epoch_metric_list = [ref_epoch_metric]
                for model_name in args.new_model_name:
                    print('calculate:', model_name)
                    new_epoch_metric = get_epoch_metric(model_name,
                                                        args.new_model_epoch,
                                                        metric, method)
                    epoch_metric_list.append(new_epoch_metric)

                legend_list = [args.ref_model_name]
                legend_list.extend(args.new_model_name)

                print("plot the figure...")
                # ylimit = (0,0.4) if method == "slope_std" else (0,1)
                ylimit = (0, 0.4) if metric == "snn_start" else (0,0.8)
                plot_metric_every_epoch(epoch_metric_list,
                                        legend_list,
                                        os.path.join(args.main_folder, check_on, f"{prop}_{metric}_{method}.png"),
                                        ylimit)



""" model settings """

NEW_MODEL_LIST1 = [
    "transformer_ep25_aug-s0.80-t0.10",
    "transformer_ep25_aug-s0.70-t0.10",
    "transformer_ep25_aug-s0.60-t0.10",
    "transformer_ep25_aug-s0.50-t0.10",
]

NEW_MODEL_LIST2 = [
    "transformer_ep25_aug-s0.80-t0.20",
    "transformer_ep25_aug-s0.70-t0.20",
    "transformer_ep25_aug-s0.60-t0.20",
    # "transformer_ep25_aug-s0.50-t0.20",
]


REF_MODEL_NAME = "transformer"
NEW_MODEL_NAME = NEW_MODEL_LIST1
REF_MODEL_EPOCH = [21,22,23,24,25,26,27,28,29,30]
NEW_MODEL_EPOCH = [26,27,28,29,30]


def add_args(parser):
    # soft constraints
    parser.add_argument('-ref_model_name', type=str, default=REF_MODEL_NAME)
    parser.add_argument('-new_model_name', nargs='+', type=str, default=NEW_MODEL_NAME)

    parser.add_argument('-ref_model_epoch', nargs='+', type=int, default=REF_MODEL_EPOCH)
    parser.add_argument('-new_model_epoch', nargs='+', type=int, default=NEW_MODEL_EPOCH)
    
    # hard constraints
    parser.add_argument('-main_folder', type=str, default='/fileserver-gamma/chaoting/ML/cvae-transformer/Inference-Results')
    parser.add_argument('-toklen', type=str, default=30)
    parser.add_argument('-decode_algo', type=str, default='greedy')


if __name__ == '__main__':
    # plot(prop='QED', metric='uniqueness')
    
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    # metric = 'snn_start'
    # method = 'slope_std'

    # metric = 'uniqueness'
    # method = 'mean'
    # plot_metric(args, metric, method)

    plot_all_figures(args)