import torch
import pickle
import pandas as pd
from Inference.toklen_sampling import tokenlen_gen_from_data_distribution


TRG = pickle.load(open('/fileserver-gamma/chaoting/ML/dataset/moses/utils/TRG.pkl', 'rb'))

df = pd.read_csv('/fileserver-gamma/chaoting/ML/cvae-transformer/Inference-Results/uniform_generation/transformer1/21/2.50_112.83_0.95.csv')
df = df.dropna(subset=['smiles'])
df['toklen'] = df['smiles'].apply(lambda x: len(TRG.tokenize((x))))

toklen_data = pd.read_csv('/fileserver-gamma/chaoting/ML/dataset/moses/raw/train/toklen_list.csv')

s = pd.DataFrame({
    'toklen': df['toklen'].tolist(),
    'train': toklen_data[toklen_data.columns[0]][:len(df)].tolist()
})

s.to_csv('./gen_len.csv')

exit()



sample = pd.read_csv('/fileserver-gamma/chaoting/ML/cvae-transformer/Inference/transformer_ep24/check_z/toklen30/greedy/z1z2_42.csv')


sample = sample.dropna(subset=['smiles'])
sample['toklen'] = sample['smiles'].apply(lambda x: len(TRG.tokenize((x))))


# n_bin = int(toklen_data.max() - toklen_data.min())

# print('t1')
# toklen1 = tokenlen_gen_from_data_distribution(data=toklen_data, size=1000,
#                                               nBins=n_bin).reshape((-1,))
# print('t2')
# toklen2 = [tokenlen_gen_from_data_distribution(data=toklen_data, size=1,
#            nBins=n_bin)[0][0] for i in range(1000)]

df = sample[['toklen']]

import seaborn as sns
import matplotlib.pyplot as plt

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,5.5))

for c in df.columns:
    sns.kdeplot(df[c], ax=ax, shade=True, label=c, linewidth=3)

# df.plot.kde(ax=ax, legend=True, xlim=xlimit)
ax.legend(fontsize=14)
# ax.set_xlabel(xlabel, fontsize=20)
# ax.set_ylabel(ylabel, fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
fig.savefig('./2.png', bbox_inches="tight") 