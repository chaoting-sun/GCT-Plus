from rdkit import Chem


def to_mol(smi):
    if isinstance(smi, str) and smi and len(smi)>0 and smi != 'nan':
        return Chem.MolFromSmiles(smi)


def is_valid(smiles):
    return 1 if to_mol(smiles) else 0


def get_canonical_smile(smile):
    if smile != 'None':
        mol = Chem.MolFromSmiles(smile)
        if mol is not None:
            smi = Chem.MolToSmiles(mol, canonical=True, doRandom=False, isomericSmiles=False)
            return smi
        else:
            return None
    else:
        return None
