"""
Step 4: Build PyTorch Geometric graphs for all molecules
Store as PyTorch Geometric .pt files
Independent implementation without MAGNet dependency
"""
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from utils.graph_builder import build_graph_simple, save_graph
from config import DEFAULT_CONFIG

def main():
    config = DEFAULT_CONFIG
    
    print("=" * 80)
    print("STEP 4: Build Molecular Graphs")
    print("=" * 80)
    
    # Load SMILES
    print(f"\n[1/3] Loading SMILES...")
    if not config.input_smiles.exists():
        print(f"  [ERROR] File not found: {config.input_smiles}")
        return None
    
    with open(config.input_smiles) as f:
        smiles_list = [line.strip() for line in f if line.strip()]
    print(f"  ✓ Loaded {len(smiles_list)} molecules")
    
    # Build graphs for all molecules using simple graph building
    print(f"\n[2/3] Building graphs (using simple graph features)...")
    failed = 0
    success = 0
    
    for idx, smiles in enumerate(tqdm(smiles_list, desc="Building graphs")):
        try:
            data = build_graph_simple(smiles)
            
            if data is not None:
                graph_path = config.graphs_dir / f"graph_{idx:06d}.pt"
                save_graph(data, graph_path)
                success += 1
            else:
                failed += 1
        except Exception as e:
            failed += 1
            if idx < 5:
                print(f"  [ERROR] Molecule {idx} ({smiles[:50]}...): {e}")
    
    print(f"\n[3/3] Verification...")
    num_saved = len(list(config.graphs_dir.glob("graph_*.pt")))
    print(f"  ✓ Verified {num_saved} graph files on disk")
    
    print(f"\n{'='*80}")
    print(f"✅ Step 4 Complete!")
    print(f"{'='*80}")
    print(f"Graph Building Results:")
    print(f"  Successful: {success}/{len(smiles_list)}")
    print(f"  Failed: {failed}")
    print(f"  Graph type: Simple (PyTorch Geometric)")
    print(f"  Output directory: {config.graphs_dir}")
    print(f"  Total graphs saved: {num_saved}")

if __name__ == "__main__":
    main()
