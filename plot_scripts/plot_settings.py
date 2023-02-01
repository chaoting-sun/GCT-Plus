from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl


config_in = {
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


config_out = {
    1: { 'nrows': 1, 'ncols': 1, 'figsize': (6, 5.5),
         'space': { }
    },
    2: { 'nrows': 1, 'ncols': 2, 'figsize': (10, 5.1),
         'space': { }
    },
    3: { 'nrows': 1, 'ncols': 3, 'figsize': (15, 5.1),
         'space': { 'left': 0.10, 'right': 0.98, 'top': 0.95, 'bottom': 0.35, 'wspace': 0.24 }
    },
    4: { 'nrows': 2, 'ncols': 2, 'figsize': (12, 12),
         'space': { 'left': 0.10, 'right': 0.98, 'top': 0.95, 'bottom': 0.20, 'wspace': 0.24 }        
    },
    6: { 'nrows': 2, 'ncols': 3, 'figsize': (14, 9.2),
         'space': { 'left': 0.80, 'right': 0.98, 'top': 0.95, 'bottom': 0.20, 'wspace': 0.36 }                
    },
    9: { 'nrows': 3, 'ncols': 3, 'figsize': (14, 13),
        'space': { 'left': 0.80, 'right': 0.98, 'top': 0.95, 'bottom': 0.20, 'wspace': 0.36 }                
    }
}


"""
figure_dict:
- figure_name
    - legend_name
        - marker
        - color
    - xlabel
    - ylabel
"""


def multi_figure_plot(figure_dict):
    mpl.rcParams['axes.linewidth'] = 2
    mpl.rcParams['xtick.major.width'] = 2
    mpl.rcParams['ytick.major.width'] = 2

    params_out = config_out[len(figure_dict)]
    
    fig = plt.figure(figsize=params_out['figsize'])
    gs = gridspec.GridSpec(nrows=params_out['nrows'],
                           ncols=params_out['ncols'])
    
    for i, (onefig_name, onefig_dict) in enumerate(figure_dict.items()):
        row = i / params_out['ncols']
        col = i % params_out['ncols']
        
        subplt_ax = plt.subplot(gs[row, col])

        if 'title' in onefig_dict:
            subplt_ax.set_title(onefig_dict['title'], fontsize=22)
        if 'xlabel' in onefig_dict:
            subplt_ax.set_xlabel(onefig_dict['xlabel'], fontsize=22)            
        if 'ylabel' in onefig_dict:
            subplt_ax.set_xlabel(onefig_dict['ylabel'], fontsize=22)        
        if 'ylimit' in onefig_dict:
            subplt_ax.set_ylim(onefig_dict['ylimit'])            
        
        params_in = config_in[len(onefig_dict)]
        for d, (legend, data) in enumerate(onefig_dict.items()):
            ax = subplt_ax.plot(
                data['x'], data['y'], 'ro',
                color=params_in['color'][d],
                marker=params_in['marker'][d],
                # linestyle='--'
            )

        subplt_ax.tick_params(axis='both', labelsize=16) 
    
    plt.tight_layout()        
    plt.savefig(figure_dict['figpath'], dpi=200)