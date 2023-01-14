import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from collections import OrderedDict
from Configuration.config import props_opts


def 


target_props = np.array(np.meshgrid(
    np.linspace(args.logp_lb, args.logp_ub, num=args.n_each_prop),
    np.linspace(args.tpsa_lb, args.tpsa_ub, num=args.n_each_prop),
    np.linspace(args.qed_lb, args.qed_ub, num=args.n_each_prop))) \
    .T.reshape(-1, 3)


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
        'figsize': (15, 5.5),
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
        # 'xticks': ,
        # 'yticks': ,
    },
    "unique"    : {
        'xlabel': '# Property set',
        'ylabel': 'Uniqueness (%)',
        'ylimit': (0, 100),
        'process': return_percentage,
        # 'xticks': ,
        # 'yticks': ,
    },
    "novel"     : {
        'xlabel': '#Property set',
        'ylabel': 'Novelty (%)',
        'ylimit': (0, 100),
        'process': return_percentage,
        # 'xticks': ,
        # 'yticks': ,
    },
    "intDiv"    : {
        'xlabel': '#Property set',
        'ylabel': 'Internal diversity',
        'ylimit': (0, 1),
        'process': return_value,
        # 'xticks': ,
        # 'yticks': ,
    },
    "snn_previous"    : {
        'xlabel': '#Property set',
        'ylabel': 'SNNprev',
        'ylimit': (0, 1),
        'process': return_value,
        # 'xticks': ,
        # 'yticks': ,
    },
    "snn_start"    : {
        'xlabel': 'Property set',
        'ylabel': 'SNNstart',
        'ylimit': (0, 1),
        'process': return_value,
        # 'xticks': ,
        # 'yticks': ,
    },
}


def get_full_metrics(data_dict, model_epoch, exper_name, file_name):
    for model_name, epoch_list in model_epoch.items():
        for epoch in epoch_list:
            data_path = os.path.join(main_folder,
                                     exper_name,
                                     model_name,
                                     str(epoch),
                                     file_name)
            legend_name = get_legend(model_name, epoch)
            data_dict[legend_name] = {}
            data_dict[legend_name]['data'] = pd.read_csv(data_path, index_col=[0])


def get_one_model_full_metric(data_dict, exper_name, model_name, epoch_list, file_name):
    """
    uniform generation
    """    
    for i, epoch in enumerate(epoch_list):
        data_path = os.path.join(main_folder,
                                 exper_name,
                                 model_name,
                                 str(epoch),
                                 file_name)
        legend_name = get_legend(model_name, epoch)
        data_dict[legend_name] = {}
        data_dict[legend_name]['data'] = pd.read_csv(data_path, index_col=[0])
        data_dict[legend_name]['legend'] = get_legend(model_name, epoch)
    return data_dict


def add_plot_features(data_dict, metric):
    for _, data in data_dict.items(): 
        data['value'] = metric_to_figure[metric]['process'](
            metric_function[metric](data['data'][metric]))
        data['x'] = data['data'].index
        data['y'] = metric_to_figure[metric]['process'](data['data'][metric])
        data['xlabel'] = metric_to_figure[metric]['xlabel']
        data['ylabel'] = metric_to_figure[metric]['ylabel']
        data['ylimit'] = metric_to_figure[metric]['ylimit']


def plot_data():
    """
    data_dict: metric_name -> legend_name -> data
    """
    
    metric_name = 'valid'

    full_data_dict = OrderedDict()
    full_data_dict[metric_name] = OrderedDict()
    
    model_epoch = OrderedDict({
        'transformer1': [20],
        'transformer2': [20],
        'transformer3': [20],        
    })
    
    get_full_metrics(
        data_dict=full_data_dict[metric_name],
        model_epoch=model_epoch,
        exper_name='uniform_generation',
        file_name='all_stat.csv'
    )
    
    add_plot_features(
        data_dict=full_data_dict[metric_name],
        metric=metric_name
    )
    
    one_figure_plot(full_data_dict, metric_name, figname='./aaa.png')
        

def get_multi_metric_data(model_epoch):
    metric_list = ['unique', 'novel', 'intDiv']
    full_data_dict = OrderedDict()

    for metric_name in metric_list:
        full_data_dict[metric_name] = OrderedDict()
            
        get_full_metrics(
            data_dict=full_data_dict[metric_name],
            model_epoch=model_epoch,
            exper_name='uniform_generation',
            file_name='all_stat.csv'
        )
        
        add_plot_features(
            data_dict=full_data_dict[metric_name],
            metric=metric_name
        )
    
    multi_figure_plot(full_data_dict, figpath='aaa.png')


def multi_figure_plot(full_data_dict, figpath):
    out_params = outfig_settings[len(full_data_dict)]
    
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

            subplt_ax.set_xlabel(data['xlabel'], fontsize=22)
            subplt_ax.set_ylabel(data['ylabel'], fontsize=22)
            
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


def one_figure_plot(full_data_dict, metric_name, figname='./aaa.png'):
    settings_out = outfig_settings[len(full_data_dict)]
    settings_in = infig_settings[len(full_data_dict[metric_name])]
    
    legend_list = []
    avg_values = []

    fig, ax = plt.subplots(1, 1, figsize=settings_out['figsize'])
    for i, (legend, data) in enumerate(full_data_dict[metric_name].items()):
        plt.plot(
            data['x'], data['y'], 'ro',
            color=settings_in['color'][i],
            marker=settings_in['marker'][i],
            # linestyle='--'
        )
        plt.ylim(data['ylimit'])
        plt.xlabel(data['xlabel'], fontsize=22)
        plt.ylabel(data['ylabel'], fontsize=22)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        
        avg_values.append(data["value"])
        legend_list.append(legend)

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2)
    ax.xaxis.set_tick_params(width=2)
    ax.yaxis.set_tick_params(width=2)

    plt.title(f'AVG: {sum(avg_values)/len(avg_values):.1f}%', fontsize=26)
    plt.tight_layout()
    # fig.legend(
    #     legend_list,
    #     fontsize=18,
    #     loc='lower left',
    #     bbox_to_anchor=(0.18, 0.15)
    # )
    plt.savefig(figname, dpi=200)



if __name__ == '__main__':
    plot_data()
    
    # model_epoch = OrderedDict({
    #     'transformer1': [20],
    #     'transformer2': [20],
    #     'transformer3': [20],        
    # })

    # get_multi_metric_data(model_epoch)