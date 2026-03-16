"""Step 2: Build motif vocabulary using decomposition
Decompose molecules into motifs and compute ECFP fingerprints
Independent implementation without MAGNet dependency
"""
import pickle
import sys
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from utils.fingerprint_utils import compute_ecfp
from utils.magnet_decomposition import MolDecomposition
from config import DEFAULT_CONFIG
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def extract_motifs_magnet(smiles_list, max_molecules=1000):
    """
    Extract molecular motifs using MAGNet decomposition.
    Each molecule is decomposed into chemical fragments (motifs).
    
    Args:
        smiles_list: list of SMILES strings  
        max_molecules: max molecules to process
        
    Returns:
        dict mapping motif SMILES to ECFP array
    """
    motif_dict = {}
    motif_counter = defaultdict(int)
    
    smiles_list = smiles_list[:max_molecules]
    
    print(f"Extracting motifs using MAGNet decomposition from {len(smiles_list)} molecules...")
    
    for smiles in tqdm(smiles_list, desc="Motif extraction"):
        try:
            # Decompose molecule into motifs
            decomp = MolDecomposition(smiles)
            motifs = decomp.get_motifs()
            
            # Add each motif to dictionary
            for motif_smiles in motifs:
                if motif_smiles not in motif_counter:
                    # Compute ECFP once per unique motif
                    ecfp = compute_ecfp(motif_smiles)
                    if ecfp is not None:
                        motif_dict[motif_smiles] = ecfp
                
                motif_counter[motif_smiles] += 1
            
        except Exception as e:
            # Skip molecules that can't be decomposed
            continue
    
    print(f"Extracted {len(motif_dict)} unique motifs from {len(smiles_list)} molecules")
    return motif_dict


def extract_motifs_scaffold(smiles_list, max_molecules=1000):
    """
    Fallback: Extract Murcko scaffolds as motifs
    
    Args:
        smiles_list: list of SMILES strings
        max_molecules: max molecules to process
        
    Returns:
        dict mapping scaffold SMILES to ECFP array
    """
    motif_dict = {}
    motif_counter = defaultdict(int)
    
    smiles_list = smiles_list[:max_molecules]
    
    print(f"Extracting Murcko scaffolds from {len(smiles_list)} molecules...")
    
    for smiles in tqdm(smiles_list, desc="Motif extraction"):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            
            # Get Murcko scaffold as motif
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            motif_smiles = Chem.MolToSmiles(scaffold)
            
            if motif_smiles not in motif_counter:
                # Compute ECFP once per unique motif
                ecfp = compute_ecfp(motif_smiles)
                if ecfp is not None:
                    motif_dict[motif_smiles] = ecfp
            
            motif_counter[motif_smiles] += 1
            
        except Exception:
            continue
    
    print(f"Extracted {len(motif_dict)} unique Murcko scaffolds")
    return motif_dict


def main():
    config = DEFAULT_CONFIG
    
    print("=" * 80)
    print("STEP 2: Build Motif Vocabulary (MAGNet Decomposition)")
    print("=" * 80)
    
    # Create output directory
    output_dir = config.output_dir / "vocabularies"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load SMILES
    print(f"[1/2] Loading SMILES from {config.input_smiles}...")
    with open(config.input_smiles, 'r') as f:
        smiles_list = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(smiles_list)} SMILES")
    
    # Extract motifs using local decomposition implementation
    max_mols = config.max_motif_molecules if config.max_motif_molecules is not None else len(smiles_list)
    print(f"[2/2] Extracting motifs (all {max_mols} molecules)...")
    print("  Using local decomposition...")
    motif_vocab = extract_motifs_magnet(smiles_list, max_molecules=max_mols)
    
    # Save vocabulary
    output_path = output_dir / "motif_vocab.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(motif_vocab, f)
    
    print("=" * 80)
    print("Step 2 Complete!")
    print("=" * 80)
    print(f"Output: {output_path}")
    print(f"Motifs: {len(motif_vocab)}")
    print()


if __name__ == "__main__":
    main()
