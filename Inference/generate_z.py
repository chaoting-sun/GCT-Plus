"""
Observe the distance of random 2 latent spaces on
training/validation set.
"""

import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
from torchtext import data
# from multiprocessing import Pool

from Utils.dataset import to_dataloader
import dill as pickle


def distance(z1, z2):
    return torch.sqrt(torch.sum((z2-z1)**2)).item()


def get_dataloader(cond_list, file_path, fields,
                   pad_idx, max_strlen, device, batch_size):
    dataset = data.TabularDataset(path=file_path, format='csv',
                                  fields=fields, skip_header=True)
    nbatches = int(np.ceil(len(dataset) / batch_size))
    data_iter = data.BucketIterator(dataset, batch_size=batch_size)
    dataloader = to_dataloader(data_iter, cond_list,
                               pad_idx, max_strlen, device)
    return dataloader, nbatches


def get_z_from_smiles(args, LOG, smiles_generator,
                      data_path, fields, device, TRG, nz=100):
    LOG.info('Get dataloader')
    batch_size = 128
    dataloader, nbatches = get_dataloader(args.conditions, data_path,
                                          fields, TRG.vocab.stoi["<pad>"],
                                          args.max_strlen, device, batch_size)

    LOG.info('Start to generate')
    zs_container = torch.empty((nz, args.max_strlen, args.latent_dim),
                               dtype=torch.float32)

    # ERROR - RuntimeError: CUDA out of memory.
    # https://blog.csdn.net/weixin_43760844/article/details/113462431
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            print(f"{(i+1)*batch_size:<5} / {nz}")

            batch.src.requires_grad, batch.econds.requires_grad = False, False
            zs, _, _ = smiles_generator.get_z_from_src(batch.src, batch.econds)

            for j, z in enumerate(zs):
                if i*batch_size+j >= nz:
                    break
                zs_container[i*batch_size+j] = z.cpu()

            torch.cuda.empty_cache()
            if (i+1)*batch_size >= nz:
                break

    return zs_container


def plot_dist_figure(data_1d, title, xlabel, ylabel, fig_path):
    # seaborn histogram
    sns.histplot(data_1d, kde=True, bins=int(data_1d.max()/2), color='blue')
    # Add labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(fig_path)


def generate_z(args, smiles_generator, fields, device, TRG, logger):
    np.random.seed(0)

    ########################## SETTINGS ##########################
    nz = 10000
    n_pairs = 100000
    # nz = 100
    # n_pairs = 10
    data_type = "train"  # or train

    data_folder = os.path.join(args.data_path, 'aug', 'data_sim1.00')
    fig_folder = os.path.join(args.data_path, "DistOfRand2Zs")
    figure_name = f"{data_type}-{args.model_type}.png"
    ##############################################################

    data_path = os.path.join(data_folder, f'{data_type}.csv')
    if not os.path.exists(data_path):
        exit(f'File path not exists: {data_path}')

    LOG = logger('generate_z', log_path=os.path.join(
        fig_folder, "generate_z.log"))

    zs_container = get_z_from_smiles(args, LOG, smiles_generator,
                                     os.path.join(
                                         data_folder, f'{data_type}.csv'),
                                     fields, device, TRG, nz)

    print(f"Generate {n_pairs} random pairs.")
    rand_pairs = [tuple(np.random.choice(range(nz), 2, replace=False))
                  for _ in range(n_pairs)]

    print("Compute distance of random two latent space.")
    collected_distance = np.empty([n_pairs, ], dtype=np.float32)
    for i, p in enumerate(rand_pairs):
        z1, z2 = zs_container[p[0]], zs_container[p[1]]
        collected_distance[i] = distance(z1, z2)

    print("Collected distance:", collected_distance)

    os.makedirs(fig_folder, exist_ok=True)
    plot_dist_figure(collected_distance,
                     title=f"Distance of 2 random Zs ({data_type})",
                     xlabel="Distance",
                     ylabel="Distribution",
                     fig_path=os.path.join(fig_folder, figure_name))

    print("Store the values")
    pickle.dump(collected_distance, open(os.path.join(fig_folder,
                f"dist_{data_type}-{args.model_type}.pkl"), "wb"))
