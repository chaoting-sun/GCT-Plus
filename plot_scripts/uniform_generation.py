import os
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from collections import OrderedDict
from scipy.stats import gaussian_kde
import seaborn as sns


property_bounds = {
    'logP': [ 0.03,   4.97],
    'tPSA': [17.92, 112.83],
    'QED' : [ 0.58,   0.95]
}


def get_prop_bounds():
    target_props = np.array(np.meshgrid(
        np.linspace(property_bounds['logP'][0], property_bounds['logP'][1], num=5),
        np.linspace(property_bounds['tPSA'][0], property_bounds['tPSA'][1], num=5),
        np.linspace(property_bounds['QED'][0], property_bounds['QED'][1], num=5))) \
        .T.reshape(-1, 3)
    prop_intervals = [
        (property_bounds['logP'][1]-property_bounds['logP'][0])/4,
        (property_bounds['tPSA'][1]-property_bounds['tPSA'][0])/4,
        (property_bounds['QED'][1]-property_bounds['QED'][0])/4,
    ]
    return target_props, prop_intervals


def get_training_set_number():
    data_path = '/fileserver-gamma/chaoting/ML/dataset/moses/raw/train/prop_serial.csv'

    train = pd.read_csv(data_path)
    target_props, prop_intervals = get_prop_bounds()
    p1_range, p2_range, p3_range = prop_intervals[0]/2, prop_intervals[1]/2, prop_intervals[2]/2
    
    n_trains = []
    n_trains = np.zeros((len(target_props),), dtype=np.int32)
   
    for i, (c_logP, c_tPSA, c_QED) in enumerate(target_props):
        f = train.loc[(c_logP-p1_range <= train.logP) & (train.logP <= c_logP+p1_range) &
                      (c_tPSA-p2_range <= train.tPSA) & (train.tPSA <= c_tPSA+p2_range) &
                      (c_QED-p3_range <= train.QED) & (train.QED <= c_QED+p3_range)]
        # n_trains.append(len(f))
        n_trains[i] = len(f)
    return n_trains

# plot selections
colors = ["#065cd4", "#0e32c4", "#770cb0", "#ff004c", "#b5070d", "#750509"]
markers = ['o', '^', 's', 'x', 'v', '|']

# how many lines in a figure
infig_settings = {
    1: {
        'color': ["#065cd4"],
        'marker': ['o'],
    },
    2: {
        'color': ["#065cd4", "#ff004c"],
        'marker': ['o', '^'],
    },    
    3: {
        'color': ["blue", "green", "orange"],
        # 'color': ["#065cd4", "#ff004c", "#750509"],
        'marker': ['o', '^', 's'],
    },
}

# how many figures
outfig_settings = {
    1: {
        'nrows': 1,
        'ncols': 1,
        'figsize': (6, 5.5),
        'space': { }
    },
    2: { 
        'nrows': 1,
        'ncols': 2,
        'figsize': (25, 10),
        'space': { }
    },
    3: {
        'nrows': 1,
        'ncols': 3,
        'figsize': (15, 4.3),
        'space': { 'left': 0.10, 'right': 0.98, 'top': 0.95, 'bottom': 0.35, 'wspace': 0.24 }
    },
    4: {
        'nrows': 2,
        'ncols': 2,
        'figsize': (12, 12),
        'space': { 'left': 0.10, 'right': 0.98, 'top': 0.95, 'bottom': 0.20, 'wspace': 0.24 }        
    },
    6: {
        'nrows': 2,
        'ncols': 3,
        'figsize': (16, 12),
        'space': { 'left': 0.80, 'right': 0.98, 'top': 0.95, 'bottom': 0.20, 'wspace': 0.36 }                
    }
}

model_to_legend = {
    'transformer1': 'CVAE-TF1',    
    'transformer2': 'CVAE-TF2',    
    'transformer3': 'CVAE-TF3',    
}

def get_legend(model_name, epoch):
    return f'{model_to_legend[model_name]} ep{epoch}'

# folder names
main_folder  = '/fileserver-gamma/chaoting/ML/cvae-transformer/Inference-Results/'
task_folder  = [ 'src_generation', 'check_conds', 'check_z', 'uniform_generation' ]
model_list   = ['transformer1', 'transformer2', 'transformer3']
check_choice = {
    'z'   : 'check_z',
    'logP': 'check_conds',
    'tPSA': 'check_conds',
    'QED' : 'check_conds',
}

# map from a metric to its computational function
def get_mean(data):
    print(data)
    return sum(data) / len(data)

def get_Ksmooth(data):
    slopes = []
    for i in range(len(data)-1):
        slopes.append(data[i+1] - data[i])
    return np.std(slopes)

metric_function = {
    "valid"       : get_mean,
    "unique"      : get_mean,
    "novel"       : get_mean,
    "intDiv"      : get_mean,
    "snn_previous": get_mean,
    "snn_start"   : get_Ksmooth,
    "logpAARD"    : get_mean,
    "tpsaAARD"    : get_mean,
    "qedAARD"     : get_mean,
    "logpAMSD"    : get_mean,
    "tpsaAMSD"    : get_mean,
    "qedAMSD"     : get_mean,
}

def return_value(val):
    return val

def return_percentage(val):
    return val*100

metric_to_figure = {
    "valid"     : {
        'xlabel': '# Property set',
        'ylabel': 'Validity (%)',
        'ylimit': (0, 100),
        'process': return_percentage,
    },
    "unique"    : {
        'xlabel': '# Property set',
        'ylabel': 'Uniqueness (%)',
        'ylimit': (0, 100),
        'process': return_percentage,
    },
    "novel"     : {
        'xlabel': '#Property set',
        'ylabel': 'Novelty (%)',
        'ylimit': (0, 100),
        'process': return_percentage,
    },
    "intDiv"    : {
        'xlabel': '#Property set',
        'ylabel': 'Internal diversity',
        'ylimit': (0, 1),
        'process': return_value,
    },
    "snn_previous"    : {
        'xlabel': '#Property set',
        'ylabel': 'SNNprev',
        'ylimit': (0, 1),
        'process': return_value,
    },
    "snn_start"    : {
        'xlabel': 'Property set',
        'ylabel': 'SNNstart',
        'ylimit': (0, 1),
        'process': return_value,
    },
    "logpAARD"    : {
        'xlabel': 'Property set',
        'ylabel': 'AARD (%)',
        'ylimit': (0, 100),
        'process': return_percentage,
    },
    "tpsaAARD"    : {
        'xlabel': 'Property set',
        'ylabel': 'AARD (%)',
        'ylimit': (0, 100),
        'process': return_percentage,
    },
    "qedAARD"    : {
        'xlabel': 'Property set',
        'ylabel': 'AARD (%)',
        'ylimit': (0, 100),
        'process': return_percentage,
    },
    "logpAMSD"    : {
        'xlabel': 'Property set',
        'ylabel': 'AMSD (%)',
        'ylimit': (0, 100),
        'process': return_percentage,
    },
    "tpsaAMSD"    : {
        'xlabel': 'Property set',
        'ylabel': 'AMSD (%)',
        'ylimit': (0, 100),
        'process': return_percentage,
    },
    "qedAMSD"    : {
        'xlabel': 'Property set',
        'ylabel': 'AMSD (%)',
        'ylimit': (0, 100),
        'process': return_percentage,
    },
}


def get_full_metrics(data_dict, model_epoch, experiment, file_name):
    for model_name, epoch_list in model_epoch.items():
        for epoch in epoch_list:
            data_path = os.path.join(main_folder,
                                     experiment,
                                     model_name,
                                     str(epoch),
                                     file_name)
            legend_name = get_legend(model_name, epoch)
            data_dict[legend_name] = {}
            data_dict[legend_name]['data'] = pd.read_csv(data_path, index_col=[0])


def get_one_model_full_metric(data_dict, experiment, model_name, epoch_list, file_name):
    """
    uniform generation
    """    
    for i, epoch in enumerate(epoch_list):
        data_path = os.path.join(main_folder,
                                 experiment,
                                 model_name,
                                 str(epoch),
                                 file_name)
        legend_name = get_legend(model_name, epoch)
        data_dict[legend_name] = {}
        data_dict[legend_name]['data'] = pd.read_csv(data_path, index_col=[0])
        data_dict[legend_name]['legend'] = get_legend(model_name, epoch)
    return data_dict


def add_plot_features(data_dict, metric):
    target_props, _ = get_prop_bounds()
    train_number = get_training_set_number()
    
    for _, data in data_dict.items():
        
        vals = []
        for i in range(len(data['data'][metric])):
            if train_number[i] == 0:
                continue
            vals.append(data['data'][metric][i])

        data['value'] = metric_to_figure[metric]['process'](
            metric_function[metric](vals))
        
        # data['value'] = metric_to_figure[metric]['process'](
        #     metric_function[metric](data['data'][metric]))
        
        # if metric == 'logpAARD':
        #     print('metric:', metric)
        #     for i in range(len(data['data'][metric])):
        #         print(data['data'][metric][i])
        
        # for i in range(len(data['data'][metric])):
        #     if train_number[i] != 0:
        #         continue
        #     print(f'{i},{target_props[i][0]},{target_props[i][1]},{target_props[i][2]},'
        #             f'{train_number[i]},{data["data"][metric][i]}')
    
        
        data['x'] = data['data'].index
        data['y'] = metric_to_figure[metric]['process'](data['data'][metric])
        data['xlabel'] = metric_to_figure[metric]['xlabel']
        data['ylabel'] = metric_to_figure[metric]['ylabel']
        data['ylimit'] = metric_to_figure[metric]['ylimit']


def plot_validity(model_epoch,
                  main_folder,
                  metric_name='valid',
                  experiment='uniform_generation',
                  figname='ug-valid.png'
                  ):
    """
    data_dict: metric_name -> legend_name -> data
    """

    full_data_dict = OrderedDict()
    full_data_dict[metric_name] = OrderedDict()
    
    get_full_metrics(
        data_dict=full_data_dict[metric_name],
        model_epoch=model_epoch,
        experiment=experiment,
        file_name='all_stat.csv'
    )
    
    add_plot_features(
        data_dict=full_data_dict[metric_name],
        metric=metric_name
    )
    
    one_figure_plot(full_data_dict, metric_name,
                    figpath=os.path.join(main_folder, figname))
        

def one_figure_plot(full_data_dict, metric_name, figpath='./aaa.png'):
    train_number = get_training_set_number()    

    settings_out = outfig_settings[len(full_data_dict)]
    settings_in = infig_settings[len(full_data_dict[metric_name])]
    
    legend_list = []
    avg_values = []

    fig, ax1 = plt.subplots(1, 1, figsize=(8, 6.8))
    ax2 = ax1.twinx()
    
    for i, (legend, data) in enumerate(full_data_dict[metric_name].items()):
        x_values = data['x']

        ax1.plot(
            data['x'], data['y'], 'ro',
            color=settings_in['color'][i],
            marker=settings_in['marker'][i],
            # linestyle='--'
        )
        ax1.set_ylim(data['ylimit'])
        ax1.set_xlabel(data['xlabel'], fontsize=22)
        ax1.set_ylabel(data['ylabel'], fontsize=22)
        
        avg_values.append(data["value"])
        legend_list.append(legend)

    ax2.plot(x_values, train_number/10000)
    ax2.set_ylabel("# Training set ("+r'$\times 10^4$'+")", fontsize=22)

    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(2)

    ax1.xaxis.set_tick_params(width=2, labelsize=18)
    ax1.yaxis.set_tick_params(width=2, labelsize=18)
    ax2.yaxis.set_tick_params(width=2, labelsize=18)

    plt.title(f'AVG: {sum(avg_values)/len(avg_values):.1f}%', fontsize=26)
    plt.tight_layout()
    print('save to:', figpath)
    plt.savefig(figpath, dpi=200)


def plot_molecular_diversity(model_epoch,
                             main_folder,
                             experiment='uniform_generation'
                             ):
    metric_list = ['unique', 'novel', 'intDiv']
    full_data_dict = OrderedDict()

    for metric_name in metric_list:
        full_data_dict[metric_name] = OrderedDict()
            
        get_full_metrics(
            data_dict=full_data_dict[metric_name],
            model_epoch=model_epoch,
            experiment=experiment,
            file_name='all_stat.csv'
        )
        
        add_plot_features(
            data_dict=full_data_dict[metric_name],
            metric=metric_name
        )
    
    out_params = outfig_settings[len(full_data_dict)]    
    multi_figure_plot(full_data_dict, out_params,
                      figpath=os.path.join(main_folder, 'ug-diversity.png'))


def multi_figure_plot(full_data_dict, out_params, figpath):
    train_number = get_training_set_number()
    
    mpl.rcParams['axes.linewidth'] = 2
    mpl.rcParams['xtick.major.width'] = 2
    mpl.rcParams['ytick.major.width'] = 2
    
    fig = plt.figure(figsize=out_params['figsize'])
    gs = gridspec.GridSpec(nrows=out_params['nrows'], ncols=out_params['ncols'])
    # gs.update(**out_params['space'])
    
    met = list(full_data_dict.keys())
    met_id = 0
    
    for r in range(out_params['nrows']):
        for c in range(out_params['ncols']):
            metric_name = met[met_id]
            met_id += 1
            
            data_dict = full_data_dict[metric_name]
            in_params = infig_settings[len(data_dict)]
            
            avg_values = []
            legend_list = []

            subplt_ax = plt.subplot(gs[r, c])

            for i, (legend, data) in enumerate(data_dict.items()):
                ax = subplt_ax.plot(
                    data['x'], data['y'], 'ro',
                    color=in_params['color'][i],
                    marker=in_params['marker'][i],
                    # linestyle='--'
                )
                
                subplt_ax.set_ylim(data['ylimit'])
                subplt_ax.tick_params(axis='both', labelsize=16)

                avg_values.append(data['value'])
                legend_list.append(legend)
                x_values = data['x']

            subplt_ax.set_xlabel(data['xlabel'], fontsize=22)
            subplt_ax.set_ylabel(data['ylabel'], fontsize=22)
            
            ax2 = subplt_ax.twinx()
            ax2.plot(x_values, train_number/10000)
            ax2.yaxis.set_tick_params(width=2, labelsize=18)
            if c == out_params['ncols'] - 1:
                ax2.set_ylabel("# Training set ("+r'$\times 10^4$'+")", fontsize=22)

            avg_values = sum(avg_values)/len(avg_values)
            if metric_name == 'intDiv':
                val_str = f'AVG: {avg_values:.2f}'
            else:
                val_str = f'AVG: {avg_values:.1f}%'
            subplt_ax.set_title(val_str, fontsize=22)
            
            plt.tight_layout()

    # fig.legend(
    #     legend_list,
    #     fontsize=18,
    #     loc='lower center',
    #     bbox_to_anchor=(0.5, 0)
    # )
    plt.savefig(figpath, dpi=200)


def multi_figure_plot2(full_data_dict, out_params, figpath):
    train_number = get_training_set_number()
    
    mpl.rcParams['axes.linewidth'] = 2
    mpl.rcParams['xtick.major.width'] = 2
    mpl.rcParams['ytick.major.width'] = 2
    
    fig = plt.figure(figsize=out_params['figsize'])
    gs = gridspec.GridSpec(nrows=out_params['nrows'], ncols=out_params['ncols'])
    # gs.update(**out_params['space'])
    
    met = list(full_data_dict.keys())
    met_id = 0
    
    for r in range(out_params['nrows']):
        for c in range(out_params['ncols']):
            metric_name = met[met_id]
            met_id += 1
            
            data_dict = full_data_dict[metric_name]
            in_params = infig_settings[len(data_dict)]
            
            avg_values = []
            legend_list = []

            subplt_ax = plt.subplot(gs[r, c])
            
            for i, (legend, data) in enumerate(data_dict.items()):
                ax = subplt_ax.plot(
                    data['x'], data['y'], 'ro',
                    color=in_params['color'][i],
                    marker=in_params['marker'][i],
                    # linestyle='--'
                )
                
                subplt_ax.set_ylim(data['ylimit'])
                subplt_ax.tick_params(axis='both', labelsize=16)

                avg_values.append(data['value'])
                legend_list.append(legend)
                x_values = data['x']

            if r == out_params['nrows']-1:
                subplt_ax.set_xlabel(data['xlabel'], fontsize=22)
            if c == 0:
                subplt_ax.set_ylabel(data['ylabel'], fontsize=22)
            
            avg_values = sum(avg_values)/len(avg_values)
            if metric_name == 'intDiv':
                val_str = f'AVG: {avg_values:.2f}'
            else:
                val_str = f'AVG: {avg_values:.1f}%'
            subplt_ax.set_title(val_str, fontsize=22)
            
            ax2 = subplt_ax.twinx()
            ax2.plot(x_values, train_number/10000)
            ax2.yaxis.set_tick_params(width=2, labelsize=18)
            
            if c == out_params['ncols'] - 1:
                ax2.set_ylabel("# Training set ("+r'$\times 10^4$'+")", fontsize=22)

            plt.tight_layout()

    # fig.legend(
    #     legend_list,
    #     fontsize=18,
    #     loc='lower center',
    #     bbox_to_anchor=(0.5, 0)
    # )
    plt.savefig(figpath, dpi=200)


def plot_property_errors(model_epoch,
                         main_folder,
                         experiment='uniform_generation'
                        ):
    metric_list = ['logpAARD', 'tpsaAARD', 'qedAARD', 'logpAMSD', 'tpsaAMSD', 'qedAMSD']
    full_data_dict = OrderedDict()

    for metric_name in metric_list:
        full_data_dict[metric_name] = OrderedDict()
            
        get_full_metrics(
            data_dict=full_data_dict[metric_name],
            model_epoch=model_epoch,
            experiment=experiment,
            file_name='all_stat.csv'
        )
        
        add_plot_features(
            data_dict=full_data_dict[metric_name],
            metric=metric_name
        )

    out_params = outfig_settings[len(full_data_dict)]
    out_params['figsize'] = (16, 9)
    multi_figure_plot2(full_data_dict, out_params,
                      figpath=os.path.join(main_folder, 'ug-prop_error.png'))



# model_epoch = OrderedDict({
#     'transformer1': [20],
#     'transformer2': [20],
#     'transformer3': [20],        
# })

# get_multi_metric_data(model_epoch)