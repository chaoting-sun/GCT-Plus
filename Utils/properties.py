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
from rdkit.Chem import rdmolops

from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect, \
    CalcNumHBA, CalcNumHBD, CalcNumRotatableBonds, CalcNumAliphaticRings, \
    CalcNumAromaticRings

from moses.metrics.SA_Score import sascorer
from moses.metrics.NP_Score import npscorer
from Utils.mapper import mapper


def disable_rdkit_logging():
    """Disable RDKit whiny logging"""
    logger = RDLogger.logger()
    logger.setLevel(RDLogger.ERROR)
    rdBase.DisableLog('rdApp.error')
disable_rdkit_logging()


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


def MW(mol):
    return Descriptors.MolWt(mol)


def HAC(mol):
    return mol.GetNumHeavyAtoms()


def HBA(mol):
    """returns the number of H-bond acceptors for a molecule"""
    return CalcNumHBA(mol)


def HBD(mol):
    """returns the number of H-bond donors for a molecule"""
    return CalcNumHBD(mol)


def RBN(mol):
    """return the number of rotatable bonds for a molecule"""
    return CalcNumRotatableBonds(mol)


def AIRN(mol):
    """return the number of aliphatic rings"""
    return CalcNumAliphaticRings(mol)


def ARRN(mol):
    return CalcNumAromaticRings(mol)


property_fn = {
    "logP": logP,
    "tPSA": tPSA,
    "QED" : QED,
    "SAS" : SAS,
    "NP"  : NP,
    "MW"  : MW,
    'HAC' : HAC,
    'HBA' : HBA,
    'HBD' : HBD,
    'RBN' : RBN,
    'AIRN': AIRN,
    'ARRN': ARRN
}


property_type = {
    "logP": 'density',
    "tPSA": 'density',
    "QED" : 'density',
    "SAS" : 'density',
    "NP"  : 'density',
    "MW"  : 'number',
    'HAC' : 'number',
    'HBA' : 'number',
    'HBD' : 'number',
    'RBN' : 'number',
    'AIRN': 'number',
    'ARRN': 'number'
}


def get_property_fn(props):
    property_fn = {
        "logP": logP,
        "tPSA": tPSA,
        "QED" : QED,
        "SAS" : SAS,
        "NP"  : NP,
        "MW"  : MW,
        'HAC' : HAC,
        'HBA' : HBA,
        'HBD' : HBD,
        'RBN' : RBN,
        'AIRN': AIRN,
        'ARRN': ARRN,
    }
    return { p: property_fn[p] for p in props }


def mols_to_props(mols, property_fns, n_jobs=1, col_names=None):
    """compute properties of mols by a list of property functions
    
    Args:
        mols (List[mol]): a list of mols
        property_fns (Dict[str, callable]): a list of property functions
        col_names (List[str]): a list of column names of the
            output pd.DataFrame
        n_jobs (int): number of available cpus
        
    Returns:
        props (pd.DataFrame): a dataframe with molecular properties
            computed by the given property functions    
    """
    props = OrderedDict()
    for i, (p, fn) in enumerate(property_fns.items()):
        name = p if col_names is None else col_names[i]
        with Pool(n_jobs) as pool:
            print('property fn:', fn.__name__)
            props[name] = list(pool.map(fn, mols))
    return pd.DataFrame(props)
    

# def smiles_list_to_props(smiles_list, prop_fns, col_names=None, n_jobs=1):
#     mols = smiles_list_to_mols(smiles_list, n_jobs)
#     assert sum([1 if m == None else 0 for m in mols]) == 0
#     props = mols_to_props(mols, prop_fns, col_names, n_jobs)
#     return props


# def predict_properties(mol_or_smi, property_list, n_jobs=1):
#     if isinstance(mol_or_smi[0], str):
#         with Pool(n_jobs) as pool:
#             mol_list = list(pool.map(to_mol, mol_or_smi))
#     else:
#         mol_list = mol_or_smi

#     calculated_properties = OrderedDict()
#     for prop in property_list:
#         with Pool(n_jobs) as pool:
#             calculated_properties[prop] = pool.map(property_fcn[prop], mol_list)
#     return pd.DataFrame(calculated_properties)


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


def compute_molecular_similarity(smiles_list1, smiles_list2,
                                 similarity_fn, n_jobs=1):
    """
    Computes the element-wise molecular similarity of two lists of SMILES strings.

    Args:
        smiles_list1 (List[str]): A list of SMILES strings.
        smiles_list2 (List[str]): A list of SMILES strings.
        similarity_fn (Callable): A function that computes the similarity between two molecules.
        n_jobs (int): The number of parallel jobs to use for the similarity computation. Set to -1 to use all available CPUs. Defaults to -1.

    Returns:
        List[float]: A list of molecular similarity values between each pair of molecules in the input lists.
    """
    with Pool(n_jobs) as pool:
        res = pool.amap(similarity_fn, smiles_list1, smiles_list2)
    similarity_list = res.get()
    return similarity_list


# def props_predictor_wrapper(conditions):
#     def props_predictor(smiles):
#         mol = MolFromSmiles(smiles)
#         if mol is not None:
#             valid = 1
#             props = [property_fcn[c](mol)
#                      for c in conditions]
#         else:
#             valid = 0
#             props = [np.nan]*len(conditions)
#         return valid, props
#     return props_predictor


def get_similarity(similarity_fn, smi_list1, smi_list2, n_jobs):
    with Pool(n_jobs) as pool:
        res = pool.amap(similarity_fn, smi_list1, smi_list2)
    similarity = res.get()
    return similarity