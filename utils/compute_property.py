import pandas as pd
from typing import Tuple, List

from rdkit import Chem
from rdkit.Chem import Descriptors, Mol, rdMolDescriptors
from rdkit.Chem import rdmolfiles as rf

from moses.metrics.SA_Score import sascorer
from moses.metrics.NP_Score import npscorer


def MolFromSmiles(smiles):
    return rf.MolFromSmiles(smiles)


def logP(mol: Mol) -> float:
    """ RDKit's partition coefficient """
    return Descriptors.MolLogP(mol)


def QED(mol: Mol) -> float:
    """ RDKit's Quantitative Estimates of Drug-likeness """
    return Descriptors.qed(mol)


def SA(mol) -> float:
    """ RDKit's Synthetic Accessibility score """
    return sascorer.calculateScore(mol)


def NP(mol) -> float:
    """ RDKit's Natural Product-likeness score """
    return npscorer.scoreMol(mol)


def tPSA(mol: Mol) -> float:
    """ RDKit's topological polar surface area """
    return Descriptors.TPSA(mol)


