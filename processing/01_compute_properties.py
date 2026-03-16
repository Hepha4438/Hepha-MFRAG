"""
Step 1: Add docking scores to existing properties
Input: zinc250k.csv (with smiles, logP, qed, SAS already computed)
Output: properties.csv with added [docking_parp1, docking_fa7, docking_5ht1b, docking_braf, docking_jak2]
"""
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils.docking_estimator import compute_docking_scores_ecfp
from config import DEFAULT_CONFIG

def main():
    config = DEFAULT_CONFIG
    
    print("=" * 80)
    print("STEP 1: Add Docking Scores to Existing Properties")
    print("=" * 80)
    
    # Load zinc250k.csv (already has smiles, logP, qed, SAS)
    csv_path = Path(__file__).parent.parent / "data/smiles/zinc250k/zinc250k.csv"
    print(f"\n[1/2] Loading properties from {csv_path}...")
    if not csv_path.exists():
        print(f"  [ERROR] File not found: {csv_path}")
        return None
    
    df_props = pd.read_csv(csv_path)
    print(f"  ✓ Loaded {len(df_props)} molecules")
    print(f"  Columns: {list(df_props.columns)}")
    
    # Extract SMILES for docking calculation
    smiles_list = df_props['smiles'].tolist()
    
    # Compute only docking scores
    print(f"\n[2/2] Computing docking scores for {len(config.docking_proteins)} proteins...")
    df_docking = compute_docking_scores_ecfp(smiles_list, config.docking_proteins, config.num_processes)
    print(f"  ✓ Computed docking scores for {len(df_docking)} molecules")
    
    # Merge docking scores with existing properties
    print(f"\nMerging docking scores with existing properties...")
    df_merged = pd.merge(df_props, df_docking, on="smiles", how="outer")
    df_merged.to_csv(config.output_properties_csv, index=False)
    
    print(f"\n{'='*80}")
    print(f"✅ Step 1 Complete!")
    print(f"{'='*80}")
    print(f"Output: {config.output_properties_csv}")
    print(f"Shape: {df_merged.shape}")
    print(f"Columns: {list(df_merged.columns)}")
    print(f"\nFirst few rows:")
    print(df_merged.head())
    
    return df_merged

if __name__ == "__main__":
    df = main()
