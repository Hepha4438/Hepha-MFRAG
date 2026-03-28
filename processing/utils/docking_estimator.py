"""Docking score estimation using Surrogate Random Forest models"""
import os
import warnings
import logging
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from .fingerprint_utils import compute_ecfp

# Suppress RDKit deprecation warnings and logger
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*MorganGenerator.*')
logging.getLogger('rdkit').setLevel(logging.ERROR)

_MODELS = {}

def get_proxy_docking_scores(smiles_list, targets, num_processes=1):
    """
    Estimate docking score using Scikit-Learn Random Forest surrogates trained on AutoDock Vina.
    Returns binding affinity DataFrame.
    """
    global _MODELS
    
    # Lazy-load models if not already loaded
    if not _MODELS:
        # Assuming script is run from project root, but let's be robust
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        models_dir = os.path.join(project_root, 'processing', 'output', 'models')
        
        for target in targets:
            model_path = os.path.join(models_dir, f"rf_{target}.joblib")
            if os.path.exists(model_path):
                _MODELS[target] = joblib.load(model_path)
            else:
                print(f"Warning: Surrogate model for {target} not found at {model_path}. Yielding default score.")
                _MODELS[target] = None

    results = []
    
    print(f"Computing Proxy Vina scores for {len(smiles_list)} molecules...")
    for smiles in tqdm(smiles_list, desc="Proxy docking scores", leave=False):
        row = {"smiles": smiles}
        fp = compute_ecfp(smiles)
        
        for target in targets:
            if fp is not None and _MODELS.get(target) is not None:
                try:
                    # Input to predict is a 2D array [n_samples, n_features]
                    pred = _MODELS[target].predict([fp])[0]
                    row[f"docking_{target}"] = float(pred)
                except Exception as e:
                    row[f"docking_{target}"] = -6.0 # Fallback on error
            else:
                row[f"docking_{target}"] = -6.0 # Default fallback if fp fails or model missing
                
        results.append(row)
        
    return pd.DataFrame(results)
