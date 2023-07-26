import pandas as pd
from Utils.smiles import murcko_scaffold, plot_highlighted_smiles_group


df1 = pd.read_csv('/fileserver-gamma/chaoting/ML/cvae-transformer/Inference-Dataset/moses/sca_sampling/scavaetf1-warmup15000-15/test_scaffolds/s9_gen.csv', index_col=[0])
df2 = pd.read_csv('/fileserver-gamma/chaoting/ML/cvae-transformer/Inference-Dataset/moses/sca_sampling/scavaetf2-warmup15000-16/test_scaffolds/s9_gen.csv', index_col=[0])
df3 = pd.read_csv('/fileserver-gamma/chaoting/ML/cvae-transformer/Inference-Dataset/moses/sca_sampling/scavaetf3-warmup15000-16/test_scaffolds/s9_gen.csv', index_col=[0])
df = pd.concat([df1, df2, df3], axis=0).reset_index(drop=True)

substructure1 = 'O=C(NCC1CCc2ccccc2O1)c1cnc(-c2cccnc2)nc1'

df = df.drop_duplicates(subset='smiles')
df['scaffold'] = df['smiles'].apply(lambda x: murcko_scaffold(x))
left1 = df[df.scaffold == substructure1]


plot_highlighted_smiles_group(
    smiles=left1['smiles'].sample(n=15),
    img_size=(440, 270),
    substructure_smiles=substructure1,
    save_path='./uc-big.png',
    n_per_mol=5
)

########################################

df1 = pd.read_csv('/fileserver-gamma/chaoting/ML/cvae-transformer/Inference-Dataset/moses/sca_sampling/scavaetf1-warmup15000-15/test_scaffolds/s67_gen.csv', index_col=[0])
df2 = pd.read_csv('/fileserver-gamma/chaoting/ML/cvae-transformer/Inference-Dataset/moses/sca_sampling/scavaetf2-warmup15000-16/test_scaffolds/s67_gen.csv', index_col=[0])
df3 = pd.read_csv('/fileserver-gamma/chaoting/ML/cvae-transformer/Inference-Dataset/moses/sca_sampling/scavaetf3-warmup15000-16/test_scaffolds/s67_gen.csv', index_col=[0])
df = pd.concat([df1, df2, df3], axis=0).reset_index(drop=True)

substructure2 = 'c1ccc2c(c1)CCS2'

df = df.drop_duplicates(subset='smiles')
df['scaffold'] = df['smiles'].apply(lambda x: murcko_scaffold(x))
left2 = df[df.scaffold == substructure2]


plot_highlighted_smiles_group(
    smiles=left2['smiles'].sample(n=15),
    img_size=(440, 270),
    substructure_smiles=substructure2,
    save_path='./uc-small.png',
    n_per_mol=5
)