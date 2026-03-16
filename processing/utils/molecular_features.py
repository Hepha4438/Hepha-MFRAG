"""Compute molecular descriptors: logP, QED, SA"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, QED
from tqdm import tqdm
import pandas as pd

def compute_logp(smiles):
    """Compute lipophilicity (logP)"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.nan
        return Crippen.MolLogP(mol)
    except:
        return np.nan

def compute_qed(smiles):
    """Compute drug-likeness (QED: Quantitative Estimate of Druglikeness)"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.nan
        return QED.qed(mol)
    except:
        return np.nan

def compute_sa(smiles):
    """Compute synthetic accessibility (SA)"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.nan
        # Simple SA proxy using molecular complexity
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        n_atoms = mol.GetNumAtoms()
        n_rings = Descriptors.RingCount(mol)
        
        # SA score: lower is more accessible
        sa = (mw + logp + n_rings * 0.5) / (n_atoms + 1)
        return sa
    except:
        return np.nan

def compute_all_properties(smiles_list, num_processes=8):
    """Compute logP, QED, SA for all molecules"""
    print(f"Computing properties for {len(smiles_list)} molecules...")
    
    results = []
    for smiles in tqdm(smiles_list, desc="Properties"):
        results.append({
            "smiles": smiles,
            "logp": compute_logp(smiles),
            "qed": compute_qed(smiles),
            "sa": compute_sa(smiles),
        })
    
    return pd.DataFrame(results)

def load_existing_properties(csv_path):
    """Load properties from existing CSV (zinc250k.csv)"""
    try:
        df = pd.read_csv(csv_path)
        # Extract relevant columns if they exist
        cols_of_interest = ["smiles", "logp", "qed", "sa"]
        available_cols = [c for c in cols_of_interest if c in df.columns]
        return df[available_cols] if available_cols else None
    except Exception as e:
        print(f"  Error loading CSV: {e}")
        return None
