import os
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from collections import OrderedDict
from Configuration.config import property_bounds
from scipy.stats import gaussian_kde
import seaborn as sns

from plot_scripts import uniform_generation as ug
from plot_scripts import conticheckz as ccz
from plot_scripts import conticheckc as ccc
from plot_scripts import src_generation as sg

property_bounds = {
    'logP': [ 0.03,   4.97],
    'tPSA': [17.92, 112.83],
    'QED' : [ 0.58,   0.95]
}

save_folder = "./plot_scripts"


def uniform_generation():
    model_epoch = OrderedDict({
        'transformer1': [20],
        'transformer2': [20],
        'transformer3': [20],
    })
    
    folder = 'garbage'
    os.makedirs(folder, exist_ok=True)
    ug.plot_validity(model_epoch, folder)
    ug.plot_molecular_diversity(model_epoch, folder)
    ug.plot_property_errors(model_epoch, folder)
    
    # ug.plot_validity(model_epoch, main_folder)    
    # ug.plot_molecular_diversity(model_epoch, main_folder)
    # ug.plot_property_errors(model_epoch, main_folder)

# uniform_generation()


def conticheck_z():
    model_epoch = OrderedDict({
        'transformer1': [20],
        'transformer2': [20],
        'transformer3': [20],
    })
    
    ccz.plot_molecular_diversity(model_epoch, save_folder)
    # ccz.plot_snn(model_epoch, save_folder)
    
# conticheck_z()

def conticheck_c():
    model_epoch = OrderedDict({
        'transformer1': [20],
        'transformer2': [20],
        'transformer3': [20],
    })
    
    # ccc.plot_snn(model_epoch, save_folder)
    ccc.plot_molecular_diversity(model_epoch, save_folder)

# conticheck_c()

"""
molecules with properties in the range of higher density
"""
# smiles_list = [
#     'Cn1ncc(Br)c1NC(=O)Nc1ccccc1', 
#     'CCN1C(=O)C(O)(CC(=O)c2ccc(C)cc2)c2ccccc21',
#     'O=C(NC1CCc2cc(F)ccc21)c1[nH]nc2c1CCCC2'
# ]
# prop_list = [
#     [2.82660, 58.95, 0.894693],
#     [2.82212, 57.61, 0.883604],
#     [2.84490, 57.78, 0.895593]
# ]

"""
molecules with properties in the range of lower density
"""
smiles_list = [
    'N#Cc1c(Br)cnc(N)c1Br', 
    'CC(=O)Nc1cccc(-c2nc3cc(C)ccc3[nH]c2=O)c1',
    'CC(NC(=O)OC(C)(C)C)c1nc(CO)nn1Cc1ccccc1'
]


prop_list = [
    [2.06048, 62.70, 0.788971],
    [2.85692, 74.85, 0.762590],
    [2.40440, 89.27, 0.877442]
]

def src_generation():
    # model_epoch = OrderedDict({
    #     'transformer1': [20],
    #     'transformer2': [20],
    #     'transformer3': [20],
    # })
    # sg.plot_similarity(save_folder, model_epoch, smiles_list, prop_list)

    model_epoch = OrderedDict({
        'transformer1': [20],
        'transformer_ep25_aug-all-s0.60-t0.10': [26],
        'transformer_ep25_aug-all-s0.70-t0.10': [26],
        'transformer_ep25_aug-all-s0.80-t0.10': [26],
    })
    sg.plot_similarity(save_folder, model_epoch, smiles_list, prop_list)
    
src_generation()