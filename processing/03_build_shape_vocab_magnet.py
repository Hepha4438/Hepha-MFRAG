"""Step 3: Build shape vocabulary by grouping motifs by topology
Uses Weisfeiler-Lehman graph hashing for topology matching
Independent implementation without external dependencies
"""
import pickle
import sys
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from utils.fingerprint_utils import compute_ecfp, average_ecfp
from config import DEFAULT_CONFIG
from rdkit import Chem
import networkx as nx
import numpy as np


def get_motif_topology_hash(motif_smiles):
    """
    Compute Weisfeiler-Lehman graph hash for motif topology.
    This represents the shape/topology of the molecule.
    
    Args:
        motif_smiles: SMILES string of motif
        
    Returns:
        WL graph hash string
    """
    try:
        mol = Chem.MolFromSmiles(motif_smiles)
        if mol is None:
            return None
        
        # Sanitize
        Chem.SanitizeMol(mol)
        
        # Get adjacency matrix
        adjacency = Chem.GetAdjacencyMatrix(mol)
        
        # Create NetworkX graph from adjacency
        graph = nx.from_numpy_array(np.triu(adjacency), create_using=nx.Graph)
        
        # Compute Weisfeiler-Lehman graph hash
        graph_hash = nx.weisfeiler_lehman_graph_hash(graph)
        
        return graph_hash
    except Exception:
        return None


def build_shape_vocabulary_from_motifs(motif_vocab, max_motifs=None):
    """
    Build shape vocabulary by grouping motifs by topology (Weisfeiler-Lehman hash).
    
    This takes the motif vocabulary from Step 2 and groups similar motifs together.
    
    Args:
        motif_vocab: dict mapping motif_smiles to ecfp_array from Step 2
        max_motifs: optional max number of motifs to process
        
    Returns:
        dict mapping topology_hash to {'motifs': [...], 'count': N, 'avg_ecfp': [...]}
    """
    shape_groups = defaultdict(list)
    shape_ecfps = defaultdict(list)
    
    motif_smiles_list = list(motif_vocab.keys())
    if max_motifs:
        motif_smiles_list = motif_smiles_list[:max_motifs]
    
    print(f"Building shape vocabulary from {len(motif_smiles_list)} motifs...")
    
    for motif_smiles in tqdm(motif_smiles_list, desc="Grouping motifs by shape"):
        try:
            # Get topology hash for this motif
            shape_hash = get_motif_topology_hash(motif_smiles)
            if shape_hash is None:
                continue
            
            # Add motif to this shape group
            shape_groups[shape_hash].append(motif_smiles)
            
            # Get pre-computed ECFP from motif vocab
            ecfp = motif_vocab[motif_smiles]
            shape_ecfps[shape_hash].append(ecfp)
            
        except Exception:
            continue
    
    # Compute average ECFP per shape
    shape_vocab = {}
    for shape_hash, motifs in shape_groups.items():
        unique_motifs = list(set(motifs))
        
        ecfps = shape_ecfps.get(shape_hash, [])
        if ecfps:
            avg_ecfp = average_ecfp(ecfps)
        else:
            avg_ecfp = None
        
        shape_vocab[shape_hash] = {
            'motifs': unique_motifs,
            'count': len(motifs),
            'avg_ecfp': avg_ecfp
        }
    
    print(f"Built shape vocabulary with {len(shape_vocab)} unique shapes")
    return shape_vocab



def main():
    config = DEFAULT_CONFIG
    
    print("=" * 80)
    print("STEP 3: Build Shape Vocabulary (from Motif Vocabulary)")
    print("=" * 80)
    
    # Create output directory
    output_dir = config.output_dir / "vocabularies"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load motif vocabulary from Step 2
    print(f"[1/2] Loading motif vocabulary from Step 2...")
    motif_vocab_path = output_dir / "motif_vocab.pkl"
    if not motif_vocab_path.exists():
        print(f"ERROR: {motif_vocab_path} not found!")
        print("Please run Step 2 first: python 02_build_motif_vocab_magnet.py")
        return
    
    with open(motif_vocab_path, 'rb') as f:
        motif_vocab = pickle.load(f)
    print(f"Loaded motif vocabulary with {len(motif_vocab)} motifs")
    
    # Build shape vocabulary from motifs
    print(f"[2/2] Building shape vocabulary by grouping motifs by topology...")
    shape_vocab = build_shape_vocabulary_from_motifs(motif_vocab)
    
    
    # Save vocabulary
    output_path = output_dir / "shape_vocab.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(shape_vocab, f)
    
    print("=" * 80)
    print("Step 3 Complete!")
    print("=" * 80)
    print(f"Output: {output_path}")
    print(f"Shapes: {len(shape_vocab)}")
    print()


if __name__ == "__main__":
    main()
