"""Fingerprint utilities for shape/motif ECFP computation"""
import warnings
import logging
import sys
import os
from io import StringIO
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import pickle

# Suppress RDKit deprecation warnings about MorganGenerator
warnings.filterwarnings('ignore', category=DeprecationWarning, module='rdkit')

# Suppress RDKit's internal logger (it uses its own logging system)
logging.getLogger('rdkit').setLevel(logging.ERROR)
logging.getLogger('rdkit.Chem').setLevel(logging.ERROR)

# Suppress stderr output from RDKit
class SuppressRDKitStderr:
    """Context manager to suppress RDKit stderr at file descriptor level (C++ warnings)"""
    def __enter__(self):
        # Save current stderr file descriptor
        self._original_fd = os.dup(2)
        # Open /dev/null
        self._devnull_fd = os.open(os.devnull, os.O_WRONLY)
        # Redirect stderr (file descriptor 2) to /dev/null
        os.dup2(self._devnull_fd, 2)
        # Also redirect Python's sys.stderr
        self._original_stderr = sys.stderr
        sys.stderr = StringIO()
        return self
    
    def __exit__(self, *args):
        # Restore stderr file descriptor
        os.dup2(self._original_fd, 2)
        os.close(self._original_fd)
        os.close(self._devnull_fd)
        # Restore Python's sys.stderr
        sys.stderr = self._original_stderr

def compute_ecfp(smiles, radius=2, nbits=2048):
    """Compute ECFP fingerprint for molecule/fragment"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        # Suppress both RDKit warnings and stderr output
        with warnings.catch_warnings(), SuppressRDKitStderr():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            warnings.filterwarnings('ignore', message='.*MorganGenerator.*')
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
