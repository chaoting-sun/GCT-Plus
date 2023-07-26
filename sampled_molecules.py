import os
import pandas as pd
from rdkit.Chem import Draw
import cairosvg
import xml.etree.ElementTree as ET
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import MolDrawOptions


def plot_smiles(smiles, save_path, size=(500,500)):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        exit('Not valid!')
    else:
        img = Draw.MolToImage(mol, size=size)
        img.save(save_path)


def plot_smiles_group(smiles, save_path, n_per_mol=None, img_size=None,
                      descriptions=None, n_jobs=1):
    """plot a list of smiles
    
    Args:
        smiles (List): a list of SMILES
        save_path (str): save path
        n_per_mol (int): number of molecules per row
        img_size (Tuple(height, width)): image size

    Returns:
        None
    """
    kwargs = {}
    kwargs['useSVG'] = True
    # kwargs['vspace'] = 2
    kwargs['mols'] = list(map(Chem.MolFromSmiles, smiles))
    draw_options = Draw.MolDrawOptions()
    # draw_options.legendFraction = 0.2
    draw_options.legendFontSize = 24
    # draw_options.rowSpacing = 0.2
    # kwargs['subImgSize'] = draw_options
    
    if n_per_mol is not None:
        kwargs['molsPerRow'] = n_per_mol
    if img_size is not None:
        kwargs['subImgSize'] = img_size
    if descriptions is not None:
        kwargs['legends'] = descriptions
    kwargs['drawOptions'] = draw_options

    svg_data = Draw.MolsToGridImage(**kwargs)

    # Parse the SVG XML data using ElementTree
    root = ET.fromstring(svg_data)
    # Extract the width and height attributes from the root element
    width = int(root.attrib['width'].strip('px'))
    height = int(root.attrib['height'].strip('px'))

    png_data = cairosvg.svg2png(bytestring=svg_data,
                                output_width=width*1.6,
                                output_height=height*1.6)
        
    with open(save_path, 'wb') as f:
        f.write(png_data)


def plot_highlighted_smiles_group(
        smiles,
        substructure_smiles,
        save_path,
        img_size,
        n_per_mol=None,
        n_jobs=1,
        highlight_color=(0, 1, 0),
        descriptions=None
    ):
    substructure = Chem.MolFromSmiles(substructure_smiles)
    molecules = list(map(Chem.MolFromSmiles, smiles))
    
    for mol in molecules:
        rdDepictor.Compute2DCoords(mol)

    highlights = []
    for mol in molecules:
        match = mol.GetSubstructMatch(substructure)
        atom_highlights = set(match)
        bond_highlights = set()
        for atom_idx in match:
            atom_bonds = mol.GetAtomWithIdx(atom_idx).GetBonds()
            for bond in atom_bonds:
                if bond.GetBeginAtomIdx() in match and bond.GetEndAtomIdx() in match:
                    bond_highlights.add(bond.GetIdx())
        highlights.append((atom_highlights, bond_highlights))

    # Create a custom MolDrawOptions object
    draw_options = MolDrawOptions()
    draw_options.highlightColour = highlight_color
    draw_options.legendFontSize = 20

    # img = Draw.MolsToGridImage(molecules, molsPerRow=n_per_mol, subImgSize=img_size,
    #                             highlightAtomLists=[hl_atoms for hl_atoms, _ in highlights],
    #                             highlightBondLists=[hl_bonds for _, hl_bonds in highlights],
    #                             drawOptions=draw_options, legends=descriptions
    #                             )
    # img.save(save_path)

    img = Draw.MolsToGridImage(molecules, molsPerRow=n_per_mol, subImgSize=img_size,
                                highlightAtomLists=[hl_atoms for hl_atoms, _ in highlights],
                                highlightBondLists=[hl_bonds for _, hl_bonds in highlights],
                                drawOptions=draw_options, useSVG=True,
                                legends=descriptions
                                )    
    root = ET.fromstring(img)
    width = int(root.attrib['width'].strip('px'))
    height = int(root.attrib['height'].strip('px'))

    png_data = cairosvg.svg2png(bytestring=img,
                                output_width=width*1.6,
                                output_height=height*1.6)
    
    with open(save_path, 'wb') as f:
        f.write(png_data)


property_list = ['logP', 'tPSA', 'QED']

tol = [0.4, 8, 0.03]
trg_prop_settings = {
    'logP': [ 1.0,   2.0,  3.0],
    'tPSA': [30.0,  60.0, 90.0],
    'QED' : [ 0.6, 0.725, 0.85],
}


if False:
    file_folder = '/fileserver-gamma/chaoting/ML/cvae-transformer/Inference-Dataset/moses/prop_sampling/'

    prop1 = pd.read_csv(os.path.join(file_folder, 'cvaetf1_15/cvaetf1-15_prop_0.csv'), index_col=[0])
    prop2 = pd.read_csv(os.path.join(file_folder, 'cvaetf2_15/cvaetf2-15_prop_1.csv'), index_col=[0])
    prop3 = pd.read_csv(os.path.join(file_folder, 'cvaetf3_15/cvaetf3-15_prop_2.csv'), index_col=[0])
    prop = pd.concat([prop1, prop2, prop3], axis=0)

    # gen_smiles1 = prop1['SMILES'].tolist()
    # gen_smiles2 = prop2['SMILES'].tolist()
    # gen_smiles3 = prop3['SMILES'].tolist()

    good_set = prop[(1 - 0.3 <= prop.logP) & (prop.logP <= 1 + 0.3) &
                    (30 - 5 <= prop.tPSA) & (prop.tPSA <= 30) &
                    (0.60 - 0.02 <= prop.QED) & (prop.QED <= 0.60 + 0.02)]
    good_set = good_set.drop_duplicates(subset='SMILES')
    n = 5
    descriptions = []
    good_set = good_set.sample(n=n).reset_index(drop=True)
    for i in range(n):
        descriptions.append(f'logP={good_set.loc[i, "logP"]:.1f} '
                            f'tPSA={good_set.loc[i, "tPSA"]:.0f} '
                            f'QED={good_set.loc[i, "QED"]:.2f}')

    print(good_set['SMILES'])


    plot_smiles_group(good_set['SMILES'], save_path='./1.png', n_per_mol=n, img_size=(350,250), descriptions=descriptions)


if False:
    from Utils.smiles import murcko_scaffold

    def plot(s, n, save_path):
        df1 = pd.read_csv(f'/fileserver-gamma/chaoting/ML/cvae-transformer/Inference-Dataset/moses/sca_sampling/scavaetf1-warmup15000-15/test_scaffolds/s{s}_gen.csv', index_col=[0])
        df2 = pd.read_csv(f'/fileserver-gamma/chaoting/ML/cvae-transformer/Inference-Dataset/moses/sca_sampling/scavaetf2-warmup15000-16/test_scaffolds/s{s}_gen.csv', index_col=[0])
        df3 = pd.read_csv(f'/fileserver-gamma/chaoting/ML/cvae-transformer/Inference-Dataset/moses/sca_sampling/scavaetf3-warmup15000-16/test_scaffolds/s{s}_gen.csv', index_col=[0])
        df = pd.concat([df1, df2, df3], axis=0).reset_index(drop=True)

        scaffold_sample = pd.read_csv('/fileserver-gamma/chaoting/ML/cvae-transformer/Inference-Dataset/moses/sca_sampling/test_scaffolds_sample.csv')

        df = df.sample(n=1000)
        df['scaffold'] = df['smiles'].apply(lambda x: murcko_scaffold(x))
        scaffold = scaffold_sample.loc[s, 'scaffold']
        df = df[df.scaffold == scaffold]
        df = df.drop_duplicates(subset='smiles')
        
        plot_highlighted_smiles_group(
            smiles=df['smiles'][:n],
            img_size=(500, 300),
            substructure_smiles=scaffold,
            save_path=save_path,
            n_per_mol=int(n/2)
        )

    plot(s=67, n=10, save_path='./1.png')
    plot(s=9, n=10, save_path='./2.png')


if True:
    s = 74
    # s = 31
    n = 16

    p1 = 3
    p2 = 60
    p3 = 0.85

    prop = pd.read_csv(f'/fileserver-gamma/chaoting/ML/cvae-transformer/Inference-Dataset/moses/psca_sampling'
                     f'/scacvaetfv31-beta0.01-warmup15000-17/test_scaffolds/s{s}_p3.0-60.0-0.725_prop.csv', index_col=[0])

    prop = prop[(p1 - 0.3 <= prop.logP) & (prop.logP <= p1 + 0.3) &
                (p2 - 6 <= prop.tPSA) & (prop.tPSA <= p2 + 6) &
                (p3 - 0.03 <= prop.QED) & (prop.QED <= p3 + 0.03)]
    prop = prop.drop_duplicates(subset='smiles')

    scaffold_sample = pd.read_csv('/fileserver-gamma/chaoting/ML/cvae-transformer/Inference-Dataset/moses/sca_sampling/test_scaffolds_sample.csv')
    scaffold = scaffold_sample.loc[s, 'scaffold']
    print(scaffold)

    prop = prop[prop.scaffold == scaffold]
    prop = prop.sample(n=n).reset_index(drop=True)
    print(prop)

    descriptions = []
    for i in range(n):
        descriptions.append(f'logP={prop.loc[i, "logP"]:.1f} '
                            f'tPSA={prop.loc[i, "tPSA"]:.0f} '
                            f'QED={prop.loc[i, "QED"]:.2f}')

    plot_highlighted_smiles_group(
        smiles=prop['smiles'],
        img_size=(390,280),
        substructure_smiles=scaffold,
        save_path='./1.png',
        n_per_mol=4,
        descriptions=descriptions
    )
    plot_smiles(scaffold, save_path='./2.png', size=(500, 300))


if False:
    descriptions = ['Start', 'I1', 'I2', 'I3', 'I4',
                    'I5', 'I6', 'I7', 'I8', 'End']
    df = pd.read_csv(f'/fileserver-gamma/chaoting/ML/cvae-transformer/Inference-Dataset/moses'
                     f'/test_decoder/vaetf1_-37/prediction29.csv', index_col=[0])
    descriptions = ['Start', 'I1', 'I2', 'I3', 'I4',
                    'I5', 'I6', 'I7', 'I8', 'End']
    plot_smiles_group(df['smiles'], save_path='./1.png', n_per_mol=5, img_size=(420,190),
                      descriptions=descriptions)
    
    df = pd.read_csv(f'/fileserver-gamma/chaoting/ML/cvae-transformer/Inference-Dataset/moses'
                     f'/test_decoder/ctf1-30/prediction29.csv', index_col=[0])
    plot_smiles_group(df['smiles'], save_path='./2.png', n_per_mol=5, img_size=(420,190),
                      descriptions=descriptions)
    


if False:
    n = 8
    prop = pd.read_csv("/fileserver-gamma/chaoting/ML/cvae-transformer/Inference-Dataset/moses/prop_sampling/case-study/evaluation.csv")

    descriptions = []
    for i in range(n):
        descriptions.append(f'logP={prop.loc[i, "logP"]:.1f} '
                            f'tPSA={prop.loc[i, "tPSA"]:.0f} '
                            f'QED={prop.loc[i, "QED"]:.2f}')
    
    plot_smiles_group(prop['SMILES'], save_path='./1.png', n_per_mol=4, img_size=(350,250), descriptions=descriptions)
