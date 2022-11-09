import numpy as np

from rdkit import RDLogger, rdBase
from rdkit.DataStructs import TanimotoSimilarity
from rdkit.Chem.rdchem import AtomValenceException
from rdkit.Chem import MolFromSmiles, MolToSmiles, \
    Descriptors, Mol, AllChem, SanitizeMol

from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect

from moses.metrics.SA_Score import sascorer
from moses.metrics.NP_Score import npscorer


def disable_rdkit_logging():
    """
    Disables RDKit whiny logging.
    """
    logger = RDLogger.logger()
    logger.setLevel(RDLogger.ERROR)
    rdBase.DisableLog('rdApp.error')

disable_rdkit_logging()

"""
property functions
"""

def logP(mol: Mol) -> float:
    """ RDKit's partition coefficient """
    return Descriptors.MolLogP(mol)


def tPSA(mol: Mol) -> float:
    """ RDKit's topological polar surface area """
    return Descriptors.TPSA(mol)


def QED(mol: Mol) -> float:
    """ RDKit's Quantitative Estimates of Drug-likeness """
    try:
        return Descriptors.qed(mol)
    except AtomValenceException:
        print('QED - invalid smiles:', MolToSmiles(mol))
        return np.nan


def SA(mol) -> float:
    """ RDKit's Synthetic Accessibility score """
    return sascorer.calculateScore(mol)


def NP(mol) -> float:
    """ RDKit's Natural Product-likeness score """
    return npscorer.scoreMol(mol)

property_prediction = {
    "logP": logP,
    "tPSA": tPSA,
    "QED": QED,
    "SA": SA,
    "NP": NP
}


def to_fp_ECFP(smi):
    if smi:
        mol = MolFromSmiles(smi)
        if mol is None:
            return None
        return GetMorganFingerprintAsBitVect(mol, 2, 1024)
        # return AllChem.GetMorganFingerprint(mol, radius=2)


def tanimoto_similarity_pool(args):
    return tanimoto_similarity(*args)


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


def is_valid(smi):
    return 1 if to_mol(smi) else 0


def props_predictor_wrapper(conditions):
    def props_predictor(smiles):
        mol = MolFromSmiles(smiles)
        if mol is not None:
            valid = 1
            props = [property_prediction[c](mol)
                     for c in conditions]
        else:
            valid = 0
            props = [np.nan]*len(conditions)
        return valid, props
    return props_predictor


def to_mol(smi):
    """
    Creates a Mol object from a SMILES string.
    :param smi: SMILES string.
    :return: A Mol object or None if it's not valid.
    """
    if isinstance(smi, str) and smi and len(smi)>0 and smi != 'nan':
        return MolFromSmiles(smi)


# https://github.com/molecularsets/moses/blob/master/moses/metrics/metrics.py
def get_mol(smiles_or_mol):
    '''
    Loads SMILES/molecule into RDKit's object
    '''
    if isinstance(smiles_or_mol, str):
        if len(smiles_or_mol) == 0:
            return None
        mol = MolFromSmiles(smiles_or_mol)
        if mol is None:
            return None
        try:
            SanitizeMol(mol)
        except ValueError:
            return None
        return mol
    return smiles_or_mol


def get_smiles(mol):
    if mol is not None:
        return MolToSmiles(mol)
    return mol


def get_canonical_smile(smile):
    if smile != 'None':
        mol = MolFromSmiles(smile)
        if mol is not None:
            smi = MolToSmiles(mol, canonical=True, doRandom=False, isomericSmiles=False)
            return smi
        else:
            return None
    else:
        return None
    

