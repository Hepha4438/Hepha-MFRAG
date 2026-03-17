"""
HES Data Generation: Prepare molecules for hierarchical drug generation model.

This script takes the output from Stages 1-4 and generates HES data by:
1. Loading pre-computed molecular graphs
2. Extracting scaffold graphs using MAGNet decomposition
3. Mapping motifs and shapes to vocabulary IDs
4. Creating PyTorch Geometric Data objects with all required fields
"""
import sys
from pathlib import Path
from tqdm import tqdm
import pickle
import torch
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from utils.magnet_decomposition import MolDecomposition
from utils.scaffold_extractor import get_scaffold_from_decomposition, get_motif_decomposition
from utils.vocab_matcher import VocabularyMatcher
from utils.graph_builder import load_graph, mol_to_pyg_data, create_scaffold_graph_data
from utils.hes_dataset import HESDataset, HESDataLoader, create_hes_dataset
from config import DEFAULT_CONFIG


def process_molecule(smiles, graph_path, matcher):
    """
    Process a single molecule to create HES data.
    
    Args:
        smiles: SMILES string
        graph_path: Path to saved .pt graph file
        matcher: VocabularyMatcher instance
    
    Returns:
        Dictionary with all required fields, or None if processing fails
    """
    try:
        # Load pre-computed atomic graph
        data_g = load_graph(graph_path)
        if data_g is None:
            return None
        
        # Decompose molecule into motifs
        try:
            mol_decomp = MolDecomposition(smiles)
        except Exception as e:
            print(f"  [WARN] Decomposition failed for {smiles[:50]}: {e}")
            return None
        
        # Extract scaffold
        scaffold_mol, _, _ = get_scaffold_from_decomposition(mol_decomp)
        
        # Create scaffold graph
        if scaffold_mol is not None:
            data_sc = mol_to_pyg_data(scaffold_mol, add_smiles=False)
        else:
            # If no scaffold (edge case), create empty scaffold
            data_sc = None
        
        # Get motif vocabulary IDs
        motif_ids = []
        for i in sorted(mol_decomp.id_to_fragment.keys()):
            if i >= 0:  # Skip leaf atoms
                motif_smiles = mol_decomp.id_to_fragment[i]
                vocab_id = matcher.get_motif_id(motif_smiles)
                if vocab_id >= 0:
                    motif_ids.append(vocab_id)
        
        if not motif_ids:
            # No valid motifs found
            return None
        
        # Get shape vocabulary IDs
        shape_ids = []
        seen_hashes = set()
        for i in sorted(mol_decomp.id_to_hash.keys()):
            if i >= 0:  # Skip leaf atoms
                topo_hash = mol_decomp.id_to_hash[i]
                if topo_hash not in seen_hashes:
                    seen_hashes.add(topo_hash)
                    shape_id = matcher.get_shape_id(topo_hash)
                    if shape_id >= 0:
                        shape_ids.append(shape_id)
        
        if not shape_ids:
            # No valid shapes found
            return None
        
        # Create data dictionary
        data_dict = {
            'smiles': smiles,
            'x_g': data_g.x if hasattr(data_g, 'x') else None,
            'edge_index_g': data_g.edge_index if hasattr(data_g, 'edge_index') else torch.zeros((2, 0), dtype=torch.long),
            'edge_attr_g': data_g.edge_attr if hasattr(data_g, 'edge_attr') else torch.zeros(0, dtype=torch.long),
            'x_sc': data_sc.x if data_sc is not None and hasattr(data_sc, 'x') else None,
            'edge_index_sc': data_sc.edge_index if data_sc is not None and hasattr(data_sc, 'edge_index') else torch.zeros((2, 0), dtype=torch.long),
            'edge_attr_sc': data_sc.edge_attr if data_sc is not None and hasattr(data_sc, 'edge_attr') else torch.zeros(0, dtype=torch.long),
            'motif_indices': motif_ids,
            'shape_indices': shape_ids,
        }
        
        return data_dict
    
    except Exception as e:
        return None


def main():
    config = DEFAULT_CONFIG
    
    print("=" * 80)
    print("STEP 5: Generate HES Data for Drug Generation Model")
    print("=" * 80)
    
    # [1/5] Load vocabularies
    print(f"\n[1/5] Loading vocabularies...")
    
    motif_vocab_path = config.motif_vocab_path
    shape_vocab_path = config.shape_vocab_path
    
    if not motif_vocab_path.exists() or not shape_vocab_path.exists():
        print(f"  [ERROR] Vocabulary files not found!")
        print(f"    - motif_vocab.pkl: {motif_vocab_path.exists()}")
        print(f"    - shape_vocab.pkl: {shape_vocab_path.exists()}")
        return None
    
    matcher = VocabularyMatcher(motif_vocab_path, shape_vocab_path)
    vocab_stats = matcher.get_stats()
    print(f"  ✓ Loaded vocabularies:")
    print(f"    - Motifs: {vocab_stats['num_motifs']}")
    print(f"    - Shapes: {vocab_stats['num_shapes']}")
    
    # [2/5] Load SMILES and graph files
    print(f"\n[2/5] Loading SMILES and graph paths...")
    
    if not config.input_smiles.exists():
        print(f"  [ERROR] SMILES file not found: {config.input_smiles}")
        return None
    
    with open(config.input_smiles) as f:
        smiles_list = [line.strip() for line in f if line.strip()]
    print(f"  ✓ Loaded {len(smiles_list)} SMILES")
    
    # Collect graph file paths
    graph_files = sorted(config.graphs_dir.glob("graph_*.pt"))
    if len(graph_files) < len(smiles_list):
        print(f"  [WARN] Only {len(graph_files)} graph files found for {len(smiles_list)} molecules")
        smiles_list = smiles_list[:len(graph_files)]
    
    # [3/5] Load properties
    print(f"\n[3/5] Loading properties...")
    
    properties_df = None
    properties_csv = config.output_properties_csv
    if properties_csv.exists():
        properties_df = pd.read_csv(properties_csv)
        # Set smiles or SMILES as index (handle both cases)
        if 'smiles' in properties_df.columns:
            properties_df = properties_df.set_index('smiles')
        elif 'SMILES' in properties_df.columns:
            properties_df = properties_df.set_index('SMILES')
        print(f"  ✓ Loaded {len(properties_df)} property entries")
    else:
        print(f"  [WARN] Properties CSV not found: {properties_csv}")
    
    # [4/5] Process molecules to create HES data
    print(f"\n[4/5] Processing molecules to create HES data...")
    
    data_list = []
    failed_count = 0
    success_count = 0
    
    for idx, (smiles, graph_path) in enumerate(tqdm(zip(smiles_list, graph_files), total=len(smiles_list), desc="Processing")):
        data_dict = process_molecule(smiles, graph_path, matcher)
        
        if data_dict is not None:
            data_list.append(data_dict)
            success_count += 1
        else:
            failed_count += 1
            if failed_count <= 5:
                print(f"  [FAIL] Molecule {idx} ({smiles[:50]}...)")
    
    print(f"\n  Processing Summary:")
    print(f"    - Successful: {success_count}/{len(smiles_list)}")
    print(f"    - Failed: {failed_count}")
    
    if not data_list:
        print(f"  [ERROR] No molecules successfully processed!")
        return None
    
    # [5/5] Create HES dataset
    print(f"\n[5/5] Creating HES Dataset...")
    
    hes_dataset = create_hes_dataset(
        data_list,
        properties_csv=config.output_properties_csv,
        lazy_load=True
    )
    
    print(f"  ✓ Created HES dataset with {len(hes_dataset)} molecules")

    # Save per-sample preprocessed data for Stage 1 dataloader
    data_list_path = config.output_dir / "hes_data_list.pkl"
    with open(data_list_path, 'wb') as f:
        pickle.dump(data_list, f)
    print(f"  Per-sample HES data saved to: {data_list_path}")
    
    # Compute and display statistics
    print(f"\n{'='*80}")
    print(f"✅ HES Data Generation Complete!")
    print(f"{'='*80}")
    print(f"Dataset Statistics:")
    print(f"  Total molecules: {len(hes_dataset)}")
    print(f"  Motif vocabulary size: {vocab_stats['num_motifs']}")
    print(f"  Shape vocabulary size: {vocab_stats['num_shapes']}")
    
    # Sample statistics
    if data_list:
        motif_counts = [len(d['motif_indices']) for d in data_list]
        shape_counts = [len(d['shape_indices']) for d in data_list]
        print(f"  Motif statistics:")
        print(f"    - Mean motifs per molecule: {sum(motif_counts)/len(motif_counts):.2f}")
        print(f"    - Min: {min(motif_counts)}, Max: {max(motif_counts)}")
        print(f"  Shape statistics:")
        print(f"    - Mean shapes per molecule: {sum(shape_counts)/len(shape_counts):.2f}")
        print(f"    - Min: {min(shape_counts)}, Max: {max(shape_counts)}")
    
    # Save dataset metadata
    metadata = {
        'num_molecules': len(hes_dataset),
        'num_motifs': vocab_stats['num_motifs'],
        'num_shapes': vocab_stats['num_shapes'],
        'success_count': success_count,
        'failed_count': failed_count,
        'data_list_file': 'hes_data_list.pkl',
    }
    
    metadata_path = config.output_dir / "hes_dataset_metadata.pkl"
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"  Metadata saved to: {metadata_path}")
    
    # Return dataset for further use
    return hes_dataset


if __name__ == "__main__":
    dataset = main()
