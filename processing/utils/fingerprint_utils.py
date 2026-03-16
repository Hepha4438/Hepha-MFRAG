"""Fingerprint utilities for shape/motif ECFP computation"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import pickle

def compute_ecfp(smiles, radius=2, nbits=2048):
    """Compute ECFP fingerprint for molecule/fragment"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
        # Convert to numpy array
        arr = np.zeros(nbits, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except:
        return None

def ecfp_similarity(fp1, fp2):
    """Tanimoto similarity between two ECFP fingerprints (numpy arrays)"""
    if fp1 is None or fp2 is None:
        return 0.0
    try:
        # Convert to bit vectors
        fp1_bv = DataStructs.CreateFromBitString(''.join(map(str, fp1.astype(int))))
        fp2_bv = DataStructs.CreateFromBitString(''.join(map(str, fp2.astype(int))))
        return DataStructs.TanimotoSimilarity(fp1_bv, fp2_bv)
    except:
        return 0.0

def average_ecfp(fingerprints):
    """Average fingerprints (for shape ECFP from motif ECFPs)"""
    valid_fps = [fp for fp in fingerprints if fp is not None]
    if not valid_fps:
        return None
    # Average across all fingerprints
    return np.mean(valid_fps, axis=0)

def save_fingerprint_dict(fp_dict, save_path):
    """Save fingerprints dictionary"""
    with open(save_path, 'wb') as f:
        pickle.dump(fp_dict, f)

def load_fingerprint_dict(load_path):
    """Load fingerprints dictionary"""
    with open(load_path, 'rb') as f:
        return pickle.load(f)
