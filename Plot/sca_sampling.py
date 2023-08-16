# import pandas as pd
# from Utils.smiles import murcko_scaffold, plot_highlighted_smiles_group


# df1 = pd.read_csv('/fileserver-gamma/chaoting/ML/cvae-transformer/Inference-Dataset/moses/sca_sampling/scavaetf1-warmup15000-15/test_scaffolds/s9_gen.csv', index_col=[0])
# df2 = pd.read_csv('/fileserver-gamma/chaoting/ML/cvae-transformer/Inference-Dataset/moses/sca_sampling/scavaetf2-warmup15000-16/test_scaffolds/s9_gen.csv', index_col=[0])
# df3 = pd.read_csv('/fileserver-gamma/chaoting/ML/cvae-transformer/Inference-Dataset/moses/sca_sampling/scavaetf3-warmup15000-16/test_scaffolds/s9_gen.csv', index_col=[0])
# df = pd.concat([df1, df2, df3], axis=0).reset_index(drop=True)

# substructure1 = 'O=C(NCC1CCc2ccccc2O1)c1cnc(-c2cccnc2)nc1'

# df = df.drop_duplicates(subset='smiles')
# df['scaffold'] = df['smiles'].apply(lambda x: murcko_scaffold(x))
# left1 = df[df.scaffold == substructure1]


# plot_highlighted_smiles_group(
#     smiles=left1['smiles'].sample(n=15),
#     img_size=(440, 270),
#     substructure_smiles=substructure1,
#     save_path='./uc-big.png',
#     n_per_mol=5
# )

# ########################################

# df1 = pd.read_csv('/fileserver-gamma/chaoting/ML/cvae-transformer/Inference-Dataset/moses/sca_sampling/scavaetf1-warmup15000-15/test_scaffolds/s67_gen.csv', index_col=[0])
# df2 = pd.read_csv('/fileserver-gamma/chaoting/ML/cvae-transformer/Inference-Dataset/moses/sca_sampling/scavaetf2-warmup15000-16/test_scaffolds/s67_gen.csv', index_col=[0])
# df3 = pd.read_csv('/fileserver-gamma/chaoting/ML/cvae-transformer/Inference-Dataset/moses/sca_sampling/scavaetf3-warmup15000-16/test_scaffolds/s67_gen.csv', index_col=[0])
# df = pd.concat([df1, df2, df3], axis=0).reset_index(drop=True)

# substructure2 = 'c1ccc2c(c1)CCS2'

# df = df.drop_duplicates(subset='smiles')
# df['scaffold'] = df['smiles'].apply(lambda x: murcko_scaffold(x))
# left2 = df[df.scaffold == substructure2]


# plot_highlighted_smiles_group(
#     smiles=left2['smiles'].sample(n=15),
#     img_size=(440, 270),
#     substructure_smiles=substructure2,
#     save_path='./uc-small.png',
#     n_per_mol=5
# )

import os
import pandas as pd
import seaborn as sns
from functools import partial
import matplotlib.pyplot as plt
from collections import OrderedDict

from Utils import mapper, get_mol, murcko_scaffold_similarity


n = 100

inference_path = '/fileserver-gamma/chaoting/ML/cvae-transformer/Inference-Dataset/moses/sca-sampling/'
model_name_list = ['scavaetf1-15', 'scavaetf2-16', 'scavaetf3-16']


def get_scaffold_similarity(scaffold_source='train', n=100, n_jobs=16):
    scaffold_sim = OrderedDict()
    
    scaffold_sample = pd.read_csv(os.path.join(inference_path,
                      f'{scaffold_source}_sample.csv'), index_col=[0])
    
    for sid in range(n):
        print('sid =', sid)
        scaffold = scaffold_sample.loc[sid, 'scaffold']
        scaffold_mol = get_mol(scaffold)
        
        gen = []
        for model_name in model_name_list:
            _gen = pd.read_csv(os.path.join(inference_path, model_name,
                               scaffold_source, f'gen{sid}.csv'), index_col=[0])
            gen.append(_gen)
        gen = pd.concat(gen, axis=0)
        gen = gen.dropna(subset=['smiles'])
        
        # gen = gen.sample(n=50)
        
        gen = gen.reset_index(drop=True)
        
        gen['mol'] = mapper(get_mol, gen['smiles'], n_jobs)
        valid = gen.dropna(subset='mol').copy()        
        similarity_fn = partial(murcko_scaffold_similarity, smi_or_mol2=scaffold_mol)
        valid['scaffold_sim'] = mapper(similarity_fn, valid['mol'], n_jobs)
        scaffold_sim[sid] = valid['scaffold_sim']
    
    scaffold_sim = pd.DataFrame(scaffold_sim)
    return scaffold_sim


scasim_train = get_scaffold_similarity(scaffold_source='train', n=n)
scasim_testsca = get_scaffold_similarity(scaffold_source='test_scaffolds', n=n)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), dpi=150)

for sid in range(n):
    murcko_scaffold_sim = scasim_train[sid].dropna()
    g = sns.kdeplot(data=murcko_scaffold_sim, ax=axes[0],
                    shade=True, linewidth=2, legend=False)
    g.set(xlabel=None)
    g.set(ylabel=None)
    axes[0].set_title(r'$\mathdefault{Scaffold}_{\mathdefault{seen}}$', fontsize=14)
    axes[0].tick_params(axis="both", which="major", labelsize=12)
    axes[0].set_xlabel(xlabel='Murcko scaffold similarity', fontsize=14)
    axes[0].set_ylabel(ylabel='Density', fontsize=14)

axes[0].set_xlim(0, 1.)

for sid in range(n):
    murcko_scaffold_sim = scasim_testsca[sid].dropna()
    g = sns.kdeplot(data=murcko_scaffold_sim, ax=axes[1],
                    shade=True, linewidth=2, legend=False)
    g.set(xlabel=None)
    g.set(ylabel=None)
    axes[1].set_title(r'$\mathdefault{Scaffold}_{\mathdefault{unseen}}$', fontsize=14)
    axes[1].tick_params(axis="both", which="major", labelsize=12)
    axes[1].set_xlabel(xlabel='Murcko scaffold similarity', fontsize=14)

axes[1].set_xlim(0, 1.)

fig.savefig(os.path.join(inference_path, 'scaffold_sim.png'), bbox_inches="tight")