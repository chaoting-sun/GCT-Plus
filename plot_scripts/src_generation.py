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
        'figsize': (10, 5.1),
        'space': { }
    },
    3: {
        'nrows': 1,
        'ncols': 3,
        'figsize': (15, 5.1),
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
        'figsize': (14, 9.2),
        'space': { 'left': 0.80, 'right': 0.98, 'top': 0.95, 'bottom': 0.20, 'wspace': 0.36 }                
    },
    9: {
        'nrows': 3,
        'ncols': 3,
        'figsize': (14, 13),
        'space': { 'left': 0.80, 'right': 0.98, 'top': 0.95, 'bottom': 0.20, 'wspace': 0.36 }                
    }
}

model_to_legend = {
    'transformer1': 'CVAE-TF1',    
    'transformer2': 'CVAE-TF2',    
    'transformer3': 'CVAE-TF3',
    'transformer_ep25_aug-all-s0.60-t0.10': 'CVAE-TF1_AUG-S0.6T0.1',
    'transformer_ep25_aug-all-s0.70-t0.10': 'CVAE-TF1_AUG-S0.7T0.1',
    'transformer_ep25_aug-all-s0.80-t0.10': 'CVAE-TF1_AUG-S0.8T0.1'
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
    return sum(data) / len(data)

def get_Ksmooth(data):
    slopes = []
    for i in range(len(data)-1):
        slopes.append(data[i+1] - data[i])
    return np.std(slopes)

metric_function = {
    "validity"    : get_mean,
    "uniqueness"  : get_mean,
    "novelty"     : get_mean,
    "int_div"     : get_mean,
    "snn_previous": get_mean,
    "snn_start"   : get_Ksmooth,
}

def return_value(val):
    return val

def return_percentage(val):
    return val*100

metric_to_figure = {
    "validity"     : {
        'xlabel': '# Step',
        'ylabel': 'Validity (%)',
        'ylimit': (0, 100),
        'process': return_percentage,
    },
    "uniqueness"  : {
        'xlabel': '# Step',
        'ylabel': 'Uniqueness (%)',
        'ylimit': (0, 100),
        'process': return_percentage,
    },
    "novelty"    : {
        'xlabel': '# Step',
        'ylabel': 'Novelty (%)',
        'ylimit': (0, 100),
        'process': return_percentage,
    },
    "int_div"    : {
        'xlabel': '# Step',
        'ylabel': 'Internal diversity',
        'ylimit': (0, 1),
        'process': return_value,
    },
    "snn_start"    : {
        'xlabel': '# Step',
        'ylabel': r'$SNN_{start}$',
        'ylimit': (0, 1),
        'process': return_value,
    },
    "snn_previous"    : {
        'xlabel': '# Step',
        'ylabel': r'$SNN_{prev}$',
        'ylimit': (0, 1),
        'process': return_value,
    },
    "not_intersect"    : {
        'xlabel': '# Step',
        'ylabel': 'No intersection (%)',
        'ylimit': (0, 100),
        'process': return_percentage,
    },    
}


def get_full_metrics(data_dict, model_epoch, experiment, property_name):
    for model_name, epoch_list in model_epoch.items():
        for epoch in epoch_list:
                data_name = f'{property_name}_statistics.csv'
                data_path = os.path.join(main_folder,
                                        experiment,
                                        model_name,
                                        str(epoch),
                                        data_name)
                legend_name = get_legend(model_name, epoch, property_name)
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
    for _, data in data_dict.items():    
        # vals = []
        # for i in range(len(data['data'][metric])):
        #     if train_number[i] == 0:
        #         continue
        #     vals.append(data['data'][metric][i])

        # data['value'] = metric_to_figure[metric]['process'](
        #     metric_function[metric](vals))
        
        data['value'] = metric_to_figure[metric]['process'](
            metric_function[metric](data['data'][metric]))
        
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

    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(2)

    ax1.xaxis.set_tick_params(width=2, labelsize=18)
    ax1.yaxis.set_tick_params(width=2, labelsize=18)

    plt.title(f'AVG: {sum(avg_values)/len(avg_values):.1f}%', fontsize=26)
    plt.tight_layout()
    print('save to:', figpath)
    plt.savefig(figpath, dpi=200)


# def plot_molecular_diversity(model_epoch,
#                              main_folder,
#                              experiment='check_conds'
#                              ):
#     metric_list = ['uniqueness', 'novelty', 'int_div']
#     full_data_dict = OrderedDict()

#     for metric_name in metric_list:
#         full_data_dict[metric_name] = OrderedDict()

            
#         get_full_metrics(
#             data_dict=full_data_dict[metric_name],
#             model_epoch=model_epoch,
#             experiment=experiment,
#             property_name='logP_statistics.csv' 
#         )
        
#         add_plot_features(
#             data_dict=full_data_dict[metric_name],
#             metric=metric_name
#         )
    
#     out_params = outfig_settings[len(full_data_dict)]    
#     multi_figure_plot(full_data_dict, out_params,
#                       figpath=os.path.join(main_folder, 'ccz-diversity.png'))


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
            
            avg_values = sum(avg_values)/len(avg_values)
            if metric_name == 'int_div':
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


def multi_figure_plot2(full_data_dict, out_params, metric_list, figpath):
    mpl.rcParams['axes.linewidth'] = 2
    mpl.rcParams['xtick.major.width'] = 2
    mpl.rcParams['ytick.major.width'] = 2
    
    fig = plt.figure(figsize=out_params['figsize'])
    gs = gridspec.GridSpec(nrows=out_params['nrows'], ncols=out_params['ncols'])
    # gs.update(**out_params['space'])
    
    met = list(full_data_dict.keys())
    met_id = 0
    
    property_list = ['logP', 'tPSA', 'QED']
    
    for r, metric in enumerate(metric_list):
        for c, prop in enumerate(property_list):
            data_dict = full_data_dict[metric][prop]
            in_params = infig_settings[len(data_dict)]

            subplt_ax = plt.subplot(gs[r, c])
            avg_values = []

            for i, (legend, data) in enumerate(data_dict.items()):
                ax = subplt_ax.plot(
                    data['x'], data['y'],
                    color=in_params['color'][i],
                    # marker=in_params['marker'][i],
                    # linestyle='--'
                )

                subplt_ax.set_ylim(data['ylimit'])
                subplt_ax.tick_params(axis='both', labelsize=16)
                avg_values.append(data['value'])

            # if r == 0:
            #     subplt_ax.set_title(prop, fontsize=25)
            
            if r == len(metric_list)-1:
                subplt_ax.set_xlabel(data['xlabel'], fontsize=22)
            if c == 0:
                subplt_ax.set_ylabel(data['ylabel'], fontsize=22)
            
            avg_values = sum(avg_values)/len(avg_values)
            if metric == 'int_div':
                val_str = f'AVG: {avg_values:.2f}'
            else:
                val_str = f'AVG: {avg_values:.1f}%'
            subplt_ax.set_title(val_str, fontsize=22)

            plt.tight_layout()

    plt.savefig(figpath, dpi=200)
    

def plot_snn(model_epoch,
             save_folder,
             experiment='check_conds'
            ):
    
    prop_list = ('logP','tPSA', 'QED')
    metric_list = ['snn_start', 'snn_previous']
    
    full_data_dict = OrderedDict()


    for metric in metric_list:
        full_data_dict[metric] = OrderedDict()

        for prop in prop_list:
            full_data_dict[metric][prop] = OrderedDict()

            for model_name, epoch_list in model_epoch.items():
                for epoch in epoch_list:
                    data_name = f'{prop}_statistics.csv'
                    data_path = os.path.join(main_folder,
                                             experiment,
                                             model_name,
                                             str(epoch),
                                             data_name)
                    legend_name = get_legend(model_name, epoch)

                    data_dict = {}
                    data_dict['data'] = pd.read_csv(data_path, index_col=[0])
                    data_dict['value'] = metric_to_figure[metric]['process'](
                        metric_function[metric](data_dict['data'][metric]))
                    data_dict['x'] = data_dict['data'].index
                    data_dict['y'] = metric_to_figure[metric]['process'](data_dict['data'][metric])
                    data_dict['xlabel'] = metric_to_figure[metric]['xlabel']
                    data_dict['ylabel'] = metric_to_figure[metric]['ylabel']
                    data_dict['ylimit'] = metric_to_figure[metric]['ylimit']
                                        
                    full_data_dict[metric][prop][legend_name] = data_dict
    
    print(full_data_dict)

    out_params = outfig_settings[6]
    multi_figure_plot2(full_data_dict, out_params, metric_list,
                      figpath=os.path.join(save_folder, 'ccc-snn.png'))


def plot_molecular_diversity(model_epoch,
             save_folder,
             experiment='check_conds'
            ):
    
    prop_list = ['logP','tPSA', 'QED']
    metric_list = ['uniqueness', 'novelty', 'int_div']
    
    full_data_dict = OrderedDict()

    for metric in metric_list:
        full_data_dict[metric] = OrderedDict()

        for prop in prop_list:
            full_data_dict[metric][prop] = OrderedDict()

            for model_name, epoch_list in model_epoch.items():
                for epoch in epoch_list:
                    data_name = f'{prop}_statistics.csv'
                    data_path = os.path.join(main_folder,
                                             experiment,
                                             model_name,
                                             str(epoch),
                                             data_name)
                    legend_name = get_legend(model_name, epoch)

                    data_dict = {}
                    data_dict['data'] = pd.read_csv(data_path, index_col=[0])
                    data_dict['value'] = metric_to_figure[metric]['process'](
                        metric_function[metric](data_dict['data'][metric]))
                    data_dict['x'] = data_dict['data'].index
                    data_dict['y'] = metric_to_figure[metric]['process'](data_dict['data'][metric])
                    data_dict['xlabel'] = metric_to_figure[metric]['xlabel']
                    data_dict['ylabel'] = metric_to_figure[metric]['ylabel']
                    data_dict['ylimit'] = metric_to_figure[metric]['ylimit']
                                        
                    full_data_dict[metric][prop][legend_name] = data_dict
                    
    print(full_data_dict)

    out_params = outfig_settings[len(prop_list)*len(metric_list)]
    multi_figure_plot2(full_data_dict, out_params, metric_list,
                      figpath=os.path.join(save_folder, 'ccc-diversity.png'))


def plot_similarity(save_folder, model_epoch, smiles_list, prop_list):
    inference_path = '/fileserver-gamma/chaoting/ML/cvae-transformer/Inference-Results/'

    data_dict = OrderedDict()

    for i in range(len(smiles_list)):
        smiles = smiles_list[i]
        props = prop_list[i]

        data_dict[smiles] = OrderedDict()

        for model_name, epoch_list in model_epoch.items():
            for epoch in epoch_list:
                data_folder = os.path.join(inference_path,
                                            'src_generation', 
                                            model_name,
                                            str(epoch),
                                            smiles_list[i],
                                            '1_step', '1.csv'
                                        )
            df = pd.read_csv(data_folder)
            df2 = df.drop_duplicates(subset="gen")
            
            legend_name = model_to_legend[model_name]
            
            data_dict[smiles][legend_name] = {}

            data_dict[smiles][legend_name]['unique'] = len(df2) / len(df) * 100
            data_dict[smiles][legend_name]['x'] = np.arange(0, 1, 0.1)
            data_dict[smiles][legend_name]['y'] = df2['similarity']

    mpl.rcParams['axes.linewidth'] = 2
    mpl.rcParams['xtick.major.width'] = 2
    mpl.rcParams['ytick.major.width'] = 2

    for i, (smiles, model_result) in enumerate(data_dict.items()):
        fig, ax = plt.subplots(1, 1, figsize=(8, 6.8))
        sns.set_style('white')
        plt.rc('legend',fontsize=21)
        ax.set_xlim((0,1))
        
        unique = []
        
        all_data = {}
        
        for j, (legend, data) in enumerate(model_result.items()):
            unique.append(data['unique'])
            all_data[legend] = data['y']
            # sns.kdeplot(data['y'], bw=0.1)
        df = pd.DataFrame(all_data)
        # sns.lineplot(data=df)
        # df.plot.kde(bw_method=0.02)
        sns.histplot(df, binwidth=0.02)
        
        ax.set_title(f"S{i+1}', uniqueness = {sum(unique)/len(unique):.1f}%", fontsize=26)
        ax.set_xlabel("Tanimoto similarity", fontsize=26)
        ax.set_ylabel("Number", fontsize=26)
        ax.xaxis.set_tick_params(width=2, labelsize=22)
        ax.yaxis.set_tick_params(width=2, labelsize=22)
        
        fig.tight_layout()
        fig.savefig(f'{save_folder}/{smiles_list[i]}-deltaP=0.png')
    

def one_figure_plot(full_data_dict, metric_name, figpath='./aaa.png'):
    settings_out = outfig_settings[len(full_data_dict)]
    settings_in = infig_settings[len(full_data_dict[metric_name])]
    
    legend_list = []
    avg_values = []

    fig, ax1 = plt.subplots(1, 1, figsize=(8, 6.8))
    
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

    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(2)

    ax1.xaxis.set_tick_params(width=2, labelsize=18)
    ax1.yaxis.set_tick_params(width=2, labelsize=18)

    plt.title(f'AVG: {sum(avg_values)/len(avg_values):.1f}%', fontsize=26)
    plt.tight_layout()
    print('save to:', figpath)
    plt.savefig(figpath, dpi=200)
