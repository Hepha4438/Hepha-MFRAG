#!/usr/bin/env python
"""
Run full preprocessing pipeline
Execute all 4 steps in sequence
"""
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent))

from config import DEFAULT_CONFIG

def run_pipeline():
    """Run all 4 preprocessing steps"""
    
    print("\n" + "="*80)
    print(" "*20 + "MAGNET PREPROCESSING PIPELINE")
    print("="*80 + "\n")
    
    start_time = time.time()
    
    # Step 1: Compute properties
    print("\n[STEP 1/4] Computing molecular properties...")
    print("-" * 80)
    try:
        from importlib import import_module
        step1 = import_module("01_compute_properties")
        df_props = step1.main()
        if df_props is None:
            print("[ERROR] Step 1 failed")
            return False
        print(f"[✓] Step 1 completed in {time.time()-start_time:.1f}s\n")
    except Exception as e:
        print(f"[ERROR] Step 1 failed: {e}")
        return False
    
    # Step 2: Build motif vocabulary
    print("\n[STEP 2/4] Building motif vocabulary...")
    print("-" * 80)
    try:
        step2 = import_module("02_build_motif_vocab")
        motif_vocab = step2.main()
        if motif_vocab is None:
            print("[WARNING] Step 2 skipped (MAGNet not available)\n")
        else:
            print(f"[✓] Step 2 completed in {time.time()-start_time:.1f}s\n")
    except Exception as e:
        print(f"[ERROR] Step 2 failed: {e}")
        # Continue anyway
    
    # Step 3: Build shape vocabulary
    print("\n[STEP 3/4] Building shape vocabulary...")
    print("-" * 80)
    try:
        step3 = import_module("03_build_shape_vocab")
        shape_vocab = step3.main()
        if shape_vocab is None:
            print("[WARNING] Step 3 skipped (MAGNet not available)\n")
        else:
            print(f"[✓] Step 3 completed in {time.time()-start_time:.1f}s\n")
    except Exception as e:
        print(f"[ERROR] Step 3 failed: {e}")
        # Continue anyway
    
    # Step 4: Build graphs
    print("\n[STEP 4/4] Building molecular graphs...")
    print("-" * 80)
    try:
        step4 = import_module("04_build_graphs")
        step4.main()
        print(f"[✓] Step 4 completed in {time.time()-start_time:.1f}s\n")
    except Exception as e:
        print(f"[ERROR] Step 4 failed: {e}")
        return False
    
    # Summary
    total_time = time.time() - start_time
    print("\n" + "="*80)
    print(" "*25 + "PIPELINE COMPLETE!")
    print("="*80)
    print(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f}m)")
    print(f"\nOutput files:")
    print(f"  - Properties: {DEFAULT_CONFIG.output_properties_csv}")
    print(f"  - Motif vocab: {DEFAULT_CONFIG.motif_vocab_path}")
    print(f"  - Shape vocab: {DEFAULT_CONFIG.shape_vocab_path}")
    print(f"  - Graphs: {DEFAULT_CONFIG.graphs_dir}")
    print("="*80 + "\n")
    
    return True

if __name__ == "__main__":
    success = run_pipeline()
    sys.exit(0 if success else 1)
