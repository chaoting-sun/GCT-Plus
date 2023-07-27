import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from pathos.multiprocessing import ProcessingPool as Pool
import cairosvg
import xml.etree.ElementTree as ET
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit.Chem.Fingerprints.FingerprintMols import FingerprintMol
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
import cairosvg
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import MolDrawOptions

# def to_mol(smi):
#     if isinstance(smi, str) and smi and len(smi)>0 and smi != 'nan':
#         return Chem.MolFromSmiles(smi)


def MolFromSmiles(smi):
    if isinstance(smi, str) and smi and len(smi)>0 and smi != 'nan':
        return Chem.MolFromSmiles(smi)


def MolToSmiles(mol):
    return Chem.MolToSmiles(mol) if mol is not None else mol


def get_mol(smi_or_mol):
    """convert smiles to mol. (copied from molgpt)
    """
    if isinstance(smi_or_mol, str):
        if len(smi_or_mol) == 0:
            return None
        mol = Chem.MolFromSmiles(smi_or_mol)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
        return mol
    return smi_or_mol


def get_canonical(smi_or_mol):
    if isinstance(smi_or_mol, str):
        if len(smi_or_mol) == 0:
            return None
        mol = Chem.MolFromSmiles(smi_or_mol)
    elif isinstance(smi_or_mol, Chem.Mol):
        mol = smi_or_mol
    else:
        return None

    if mol is None:
        return None

    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return None

    return Chem.MolToSmiles(mol, canonical=True)


def to_fp_ECFP(smi):
    if smi:
        mol = MolFromSmiles(smi)
        if mol is None:
            return None
        return GetMorganFingerprintAsBitVect(mol, 2, 1024)


def tanimoto_similarity(smi1, smi2):
    fp1, fp2 = None, None
    if smi1 and type(smi1)==str and len(smi1)>0:
        fp1 = to_fp_ECFP(smi1)
    if smi2 and type(smi2)==str and len(smi2)>0:
        fp2 = to_fp_ECFP(smi2)

    if fp1 is not None and fp2 is not None:
        return TanimotoSimilarity(fp1, fp2)
    else:
        return None


# def get_murcko_scaffold(smi_or_mol):
#     try:
#         mol = get_mol(smi_or_mol)
#         scaffold = MurckoScaffoldSmiles(mol=mol)
#         return scaffold
#     except Exception as e:
#         print(f"Error: {e}")
#         return None


def smi_to_mol(smiles, include_smi=False, n_jobs=1):
    """convert smiles to mol object
    
    Args:
        smiles (List[str]): SMILES
        n_jobs (int): number of cpus
         
    Returns:
        valid_smi (List[str]): a list of valid SMILES
        valid_mol (List[mol]): a list of valid mols
    """
    with Pool(n_jobs) as pool:
        mols = np.array(pool.map(get_mol, smiles))
    valid_mask = (mols != None)
    valid_mol = np.array(mols)[valid_mask]
    
    if include_smi:
        valid_smi = np.array(smiles)[valid_mask]
        return valid_mol, valid_smi
    return valid_mol


def mol_to_smi(mol, n_jobs=1):
    with Pool(n_jobs) as pool:
        smiles_list = np.array(pool.map(MolToSmiles, mol))
    return smiles_list


def is_valid(smi):
    return 1 if smi_to_mol(smi) else 0


def canonical_smi(smi):
    if smi is None:
        return None
    mol = get_mol(smi)
    if mol is not None:
        return Chem.MolToSmiles(mol, canonical=True, doRandom=False, isomericSmiles=False)
    return None


def murcko_scaffold(smi_or_mol):
    mol = get_mol(smi_or_mol)
    if mol is None:
        return None
    return MurckoScaffoldSmiles(mol=mol)
    
    
def murcko_scaffold_similarity(smi_or_mol1, smi_or_mol2):
    scaffold1 = murcko_scaffold(smi_or_mol1)
    scaffold2 = murcko_scaffold(smi_or_mol2)
    if scaffold1 is None or scaffold2 is None:
        return None
    mol1 = get_mol(scaffold1)
    mol2 = get_mol(scaffold2)
    if mol1 is None or mol2 is None:
        return None
    fp1 = FingerprintMol(mol1)
    fp2 = FingerprintMol(mol2)
    return TanimotoSimilarity(fp1, fp2)

    
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
    molecules = smi_to_mol(smiles, n_jobs=n_jobs)
    
    print(molecules)
    
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


# def MurckoScaffoldSimilarity(smi1, smi2):
#     """refer to the implementation in molgpt"""
#     mol1, mol2 = MolFromSmiles(smi1), MolFromSmiles(smi2)
#     if mol1 is None or mol2 is None:
#         return None
#     try:
#         ms1 = MurckoScaffoldSmiles(smi1)
#     except:
#         print(f'Cannot convert into scaffold: {smi1}')
#         return None
#     try:
#         ms2 = MurckoScaffoldSmiles(smi2)
#     except:
#         print(f'Cannot convert into scaffold: {smi2}')
#         return None

#     fp1, fp2 = FingerprintMol(MolFromSmiles(ms1)), FingerprintMol(MolFromSmiles(ms2))
#     return TanimotoSimilarity(fp1, fp2)
    

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
    kwargs['mols'] = smi_to_mol(smiles, n_jobs=n_jobs)
    draw_options = Draw.MolDrawOptions()
    # draw_options.rowSpacing = 0.2
    # kwargs['subImgSize'] = draw_options
    
    if n_per_mol is not None:
        kwargs['molsPerRow'] = n_per_mol
    if img_size is not None:
        kwargs['subImgSize'] = img_size
    if descriptions is not None:
        kwargs['legends'] = descriptions

    # Create a MolDrawOptions object to set the drawing options
    # draw_options = Draw.MolDrawOptions()
    # draw_options.rowSpacing = 1
    # draw_options.scale = 1
    # draw_options.padding = 0
    # draw_options.clearBackground = False
    # draw_options.useBWAtomPalette()
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


# def plot_smiles_group(smiles_group, save_path=None, n_per_mol=4, molSize=(200, 200),
#                       kekulize=True,  legends=[], highlightAtoms=None,
#                       highlightAtomColors=None, highlightAtomRadii=None):
#     """
#     Plot a list of SMILES strings as a grid of RDKit molecules. (GPT-3.5)

#     Args:
#         smiles_group (List[str]): A list of SMILES strings to plot.
#         save_path (str): The file path to save the output image to. If None, the image will not be saved. Defaults to None.
#         n_per_mol (int): The number of molecules to plot per row in the image. Defaults to 4.
#         molSize (Tuple[int, int]): The size of each molecule in the grid, in pixels. Defaults to (200, 200).
#         kekulize (bool): Whether to kekulize the SMILES strings before plotting. Defaults to True.
#         legends (List[str]): A list of labels to add as captions for each molecule in the grid. Defaults to None.
#         highlightAtoms (List[List[int]]): A list of atom indices to highlight in each molecule. Defaults to None.
#         highlightAtomColors (List[str]): A list of colors to use for highlighting each atom. Defaults to None.
#         highlightAtomRadii (List[int]): A list of radii to use for highlighting each atom. Defaults to None.

#     Returns:
#         None
#     """

#     # Create RDKit molecules from the input SMILES strings
#     mols = [smi_to_mol(smi) for smi in smiles_group]

#     # Create a MolDrawOptions object to set the drawing options
#     draw_options = Draw.MolDrawOptions()
#     draw_options.useBWAtomPalette()

#     # Set the highlight options if provided
#     if highlightAtoms is not None:
#         draw_options.highlightAtoms = highlightAtoms
#         draw_options.highlightAtomColors = highlightAtomColors
#         draw_options.highlightAtomRadii = highlightAtomRadii

#     # Generate the grid image using MolsToGridImage
#     svg_data = Draw.MolsToGridImage(mols=list(mols),
#                                     molsPerRow=n_per_mol,
#                                     subImgSize=molSize,
#                                     kekulize=kekulize,
#                                     drawOptions=draw_options,
#                                     legends=legends, 
#                                     useSVG=True)

#     # Convert the SVG data to a PNG image using cairosvg
#     png_data = cairosvg.svg2png(bytestring=svg_data)

#     # Save the image to disk if a file path is provided
#     if save_path is not None:
#         with open(save_path, 'wb') as f:
#             f.write(png_data)


def get_substructure_smiles(smiles, min_ratio=0.1, max_ratio=0.5):
    mol = Chem.MolFromSmiles(smiles)
    heavy_atom_count = mol.GetNumHeavyAtoms()
    substructures = []

    for bond in mol.GetBonds():
        bt = bond.GetBondType()
        if bt == Chem.rdchem.BondType.SINGLE:
            emol = Chem.EditableMol(mol)
            emol.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            newmol = emol.GetMol()
            fragments = Chem.GetMolFrags(newmol, asMols=True)

            for frag in fragments:
                frag_heavy_atom_count = frag.GetNumHeavyAtoms()
                ratio = frag_heavy_atom_count / heavy_atom_count

                if min_ratio <= ratio <= max_ratio:
                    frag_smiles = Chem.MolToSmiles(frag)
                    substructures.append(frag_smiles)
    
    # Add Murcko scaffold to substructures
    murcko_scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    murcko_scaffold_smiles = Chem.MolToSmiles(murcko_scaffold)
    substructures.append(murcko_scaffold_smiles)
    
    return list(set(substructures))


# def get_substructure_smiles(smiles, min_ratio=0.1, max_ratio=0.5):
#     mol = Chem.MolFromSmiles(smiles)
#     heavy_atom_count = mol.GetNumHeavyAtoms()
#     substructures = []

#     for bond in mol.GetBonds():
#         bt = bond.GetBondType()
#         if bt == Chem.rdchem.BondType.SINGLE:
#             emol = Chem.EditableMol(mol)
#             emol.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
#             newmol = emol.GetMol()
#             fragments = Chem.GetMolFrags(newmol, asMols=True)
            
#             for idx, frag in enumerate(fragments):
#                 frag_heavy_atom_count = frag.GetNumHeavyAtoms()
#                 ratio = frag_heavy_atom_count / heavy_atom_count
#                 if min_ratio <= ratio <= max_ratio:
#                     # Check if the fragment is connected to at least one other fragment
#                     for other_idx, other_frag in enumerate(fragments):
#                         if idx != other_idx and Chem.CanonicalRankAtoms(frag, breakTies=True) != Chem.CanonicalRankAtoms(other_frag, breakTies=True):
#                             substructures.append(Chem.MolToSmiles(frag))
#                             break

#     # Remove duplicates and return the list
#     return list(set(substructures))


# def get_substructure_smiles(smiles, min_ratio=0.1, max_ratio=0.5):
#     mol = Chem.MolFromSmiles(smiles)
#     heavy_atom_count = mol.GetNumHeavyAtoms()
#     substructures = []

#     # Add Murcko scaffold to substructures
#     # murcko_scaffold = MurckoScaffold.GetScaffoldForMol(mol)
#     # murcko_scaffold_smiles = Chem.MolToSmiles(murcko_scaffold)
#     # substructures.append(murcko_scaffold_smiles)

#     for bond in mol.GetBonds():
#         bt = bond.GetBondType()
#         if bt == Chem.rdchem.BondType.SINGLE:
#             emol = Chem.EditableMol(mol)
#             emol.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
#             newmol = emol.GetMol()
#             fragments = Chem.GetMolFrags(newmol, asMols=True)

#             # Find the central substructure as the largest fragment by the number of heavy atoms
#             central_substructure = max(fragments, key=lambda x: x.GetNumHeavyAtoms())
#             central_heavy_atom_count = central_substructure.GetNumHeavyAtoms()
#             ratio = central_heavy_atom_count / heavy_atom_count

#             if min_ratio <= ratio <= max_ratio:
#                 substructures.append(Chem.MolToSmiles(central_substructure))

#     # Remove duplicates and return the list
#     return list(set(substructures))


def is_substructure(smiles, subst):
    mol = Chem.MolFromSmiles(smiles)
    subst_mol = Chem.MolFromSmiles(subst)
    return mol.HasSubstructMatch(subst_mol)


def generate_substructures(mol):
    substructures = set()
    for atom in mol.GetAtoms():
        for neighbors in atom.GetNeighbors():
            bond = mol.GetBondBetweenAtoms(atom.GetIdx(), neighbors.GetIdx())
            substructure = Chem.MolFragmentToSmiles(mol, [atom.GetIdx(), neighbors.GetIdx()], [bond.GetIdx()], canonical=True)
            substructures.add(substructure)
    return substructures

from itertools import combinations
from collections import deque

def generate_substructures_within_ratio(mol, min_ratio, max_ratio):
    num_atoms_mol = mol.GetNumAtoms()
    substructures = set()
    
    queue = deque()
    for atom in mol.GetAtoms():
        queue.append(([atom.GetIdx()], set()))

    while queue:
        atom_indices, bond_indices = queue.popleft()
        num_atoms_substructure = len(atom_indices)
        ratio = num_atoms_substructure / num_atoms_mol

        if min_ratio <= ratio <= max_ratio:
            env = Chem.FindAtomEnvironmentOfRadiusN(mol, 0, atom_indices)
            substructure_mol = Chem.PathToSubmol(mol, env)
            substructure_smiles = Chem.MolToSmiles(substructure_mol, canonical=True)
            substructures.add(substructure_smiles)

        if ratio < max_ratio:
            last_atom_idx = atom_indices[-1]
            last_atom = mol.GetAtomWithIdx(last_atom_idx)
            for neighbor in last_atom.GetNeighbors():
                neighbor_idx = neighbor.GetIdx()
                if neighbor_idx not in atom_indices:
                    bond_idx = mol.GetBondBetweenAtoms(last_atom_idx, neighbor_idx).GetIdx()
                    if bond_idx not in bond_indices:
                        new_atom_indices = atom_indices + [neighbor_idx]
                        new_bond_indices = bond_indices.union({bond_idx})
                        queue.append((new_atom_indices, new_bond_indices))
                        
    return substructures


def randomize_smiles(smiles):
    """from molegpt"""
    """Perform a randomization of a SMILES string
    must be RDKit sanitizable"""
    m = Chem.MolFromSmiles(smiles)
    ans = list(range(m.GetNumAtoms()))
    np.random.shuffle(ans)
    nm = Chem.RenumberAtoms(m,ans)
    return Chem.MolToSmiles(nm, canonical=False, isomericSmiles=False)




if __name__ == '__main__':
    smiles = 'COc1ccc(OC)c(Cc2cnc3nc(N)nc(N)c3c2C)c1'
    
    for i in range(100):
        print(randomize_smiles(smiles))


    exit()
    
    mol = Chem.MolFromSmiles(smiles)
    subs = generate_substructures_within_ratio(mol, 0.1, 1)
    print(subs)
    exit()
    substructure = get_substructure_smiles(smiles)
    
    for sub in substructure:
        print(sub)
        if is_substructure(sub, smiles):
            print(f'>> substructure of {smiles}')