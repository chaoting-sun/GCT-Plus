import numpy as np
import pandas as pd
from collections import OrderedDict
from pathos.multiprocessing import ProcessingPool as Pool

from rdkit import RDLogger, rdBase
from rdkit.Chem.Fingerprints.FingerprintMols import FingerprintMol
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

from rdkit.Chem.rdchem import AtomValenceException
from rdkit.Chem import MolFromSmiles, MolToSmiles, \
    Descriptors, Mol, AllChem, SanitizeMol

from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect

from moses.metrics.SA_Score import sascorer
from moses.metrics.NP_Score import npscorer


def disable_rdkit_logging():
    """Disable RDKit whiny logging"""
    logger = RDLogger.logger()
    logger.setLevel(RDLogger.ERROR)
    rdBase.DisableLog('rdApp.error')

disable_rdkit_logging()

"""Property prediction function"""

def logP(mol):
    return Descriptors.MolLogP(mol)


def tPSA(mol):
    return Descriptors.TPSA(mol)


def QED(mol):
    try:
        return Descriptors.qed(mol)
    except AtomValenceException:
        return np.nan


def SAS(mol) -> float:
    return sascorer.calculateScore(mol)


def NP(mol) -> float:
    return npscorer.scoreMol(mol)


# property_prediction = {
#     "logP": logP,
#     "tPSA": tPSA,
#     "QED" : QED,
#     "SA"  : SAS,
#     "NP"  : NP
# }

property_fcn = {
    "logP": logP,
    "tPSA": tPSA,
    "QED" : QED,
    "SAS"  : SAS,
    "NP"  : NP
}


def predict_properties(smiles_list, property_list, n_jobs=1):
    with Pool(n_jobs) as pool:
        mol_list = list(pool.map(to_mol, smiles_list))

    calculated_properties = OrderedDict()
    for prop in property_list:
        with Pool(n_jobs) as pool:
            calculated_properties[prop] = list(pool.map(property_fcn[prop], mol_list))
    return pd.DataFrame(calculated_properties)


def to_fp_ECFP(smi):
    if smi:
        mol = MolFromSmiles(smi)
        if mol is None:
            return None
        return GetMorganFingerprintAsBitVect(mol, 2, 1024)


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


def MurckoScaffoldSimilarity(smi1, smi2):
    """refer to the implementation in molgpt"""
    
    mol1, mol2 = MolFromSmiles(smi1), MolFromSmiles(smi2)
    if mol1 is None or mol2 is None:
        return None

    ms1, ms2 = MurckoScaffoldSmiles(smi1), MurckoScaffoldSmiles(smi2)    
    fp1, fp2 = FingerprintMol(MolFromSmiles(ms1)), FingerprintMol(MolFromSmiles(ms2))
    return TanimotoSimilarity(fp1, fp2)


def get_similarity(smiles_list1, smiles_list2, similarity_fcn, n_jobs):
    with Pool(n_jobs) as pool:
        res = pool.amap(similarity_fcn, smiles_list1, smiles_list2)
    similarity_list = res.get()
    return similarity_list


def is_valid(smi):
    return 1 if to_mol(smi) else 0

# old
def predict_props(smiles, conditions=['logP', 'tPSA', 'QED']):
    mol = MolFromSmiles(smiles)
    if mol:
        return [property_fcn[c](mol)
                for c in conditions]
    return [np.nan]*len(conditions)


def props_predictor_wrapper(conditions):
    def props_predictor(smiles):
        mol = MolFromSmiles(smiles)
        if mol is not None:
            valid = 1
            props = [property_fcn[c](mol)
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
    