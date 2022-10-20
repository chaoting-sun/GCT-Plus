import os
import numpy as np
import torch
from torchtext import data

from Utils.dataset import to_dataloader


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


def get_z_from_train_smiles(args, logger, smiles_generator,
                            fields, device, TRG, nz=100):
    source_file = os.path.join(
        args.data_path, 'aug', f'data_sim1.00', 'validation.csv')
    if not os.path.exists(source_file):
        exit(f'File path not exists: {source_file}')

    logger.info('Get dataloader')
    batch_size = 128
    dataloader, nbatches = get_dataloader(args.conditions, source_file,
                                          fields, TRG.vocab.stoi["<pad>"],
                                          args.max_strlen, device, batch_size)

    logger.info('Start to generate')
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


# Import the libraries
import matplotlib.pyplot as plt
import seaborn as sns


def plot_dist_figure(data_1d, title, xlabel, ylabel, fig_path):
    # seaborn histogram
    sns.histplot(data_1d, kde=True, bins=int(data_1d.max()/2), color = 'blue')
    # Add labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(fig_path)


def generate_z(args, logger, smiles_generator, fields, device, TRG):
    np.random.seed(0)

    nz = 10000
    n_pairs = 100000

    zs_container = get_z_from_train_smiles(
        args, logger, smiles_generator, fields, device, TRG, nz)
    
    rand_pairs = [np.random.choice(range(nz), 2, replace=False)
                  for _ in range(n_pairs)]

    collected_distance = np.empty([n_pairs,], dtype=np.float32)
    for i, p in enumerate(rand_pairs):
        z1, z2 = zs_container[p[0]], zs_container[p[1]]
        collected_distance[i] = distance(z1, z2)

    print("collected distance:", collected_distance)
    plot_dist_figure(collected_distance,
                     title="dist",
                     xlabel="Distance of random two latent space",
                     ylabel="y",
                     fig_path="./k.png")


