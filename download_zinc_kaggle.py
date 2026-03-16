#!/usr/bin/env python
"""
Download ZINC250k from Kaggle
Using kagglehub library
"""

import sys
from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).parent.absolute()
DATA_DIR = PROJECT_ROOT / "data" / "smiles" / "zinc250k"
DATA_DIR.mkdir(parents=True, exist_ok=True)

ZINC250K_FILE = DATA_DIR / "zinc250k.smi"


def download_from_kaggle():
    """Download ZINC250k dataset from Kaggle"""
    
    print("=" * 70)
    print("Downloading ZINC250k from Kaggle...")
    print("=" * 70)
    print()
    
    try:
        import kagglehub
    except ImportError:
        print("❌ kagglehub not installed")
        print()
        print("Install with:")
        print("   pip install kagglehub")
        print()
        return False
    
    try:
        print("📥 Downloading dataset...")
        path = kagglehub.dataset_download("basu369victor/zinc250k")
        
        print(f"✓ Downloaded to: {path}")
        print()
        
        # Find data files
        download_path = Path(path)
        
        # Look for CSV or SMILES files
        csv_files = list(download_path.glob("*.csv"))
        smiles_files = list(download_path.glob("*.smi")) + list(download_path.glob("*.txt"))
        
        if csv_files:
            src_file = csv_files[0]
            print(f"📋 Found CSV file: {src_file.name}")
            print()
            
            # Convert CSV to SMILES
            print("🔄 Converting CSV to SMILES format...")
            import pandas as pd
            
            df = pd.read_csv(src_file)
            
            # Find SMILES column (usually named 'SMILES', 'smiles', or similar)
            smiles_col = None
            for col in df.columns:
                if 'smiles' in col.lower():
                    smiles_col = col
                    break
            
            if smiles_col is None:
                # If no SMILES column found, list available columns
                print(f"❌ No SMILES column found")
                print(f"   Available columns: {list(df.columns)}")
                return False
            
            # Extract SMILES
            print(f"   Using column: {smiles_col}")
            smiles_list = df[smiles_col].dropna().unique()
            
            print(f"   Found {len(smiles_list):,} unique SMILES")
            print()
            
            # Save to SMILES file
            print(f"💾 Saving to: {ZINC250K_FILE}")
            with open(ZINC250K_FILE, 'w') as f:
                for smi in smiles_list:
                    f.write(str(smi).strip() + '\n')
            
            # Verify
            count = sum(1 for _ in open(ZINC250K_FILE))
            size_mb = ZINC250K_FILE.stat().st_size / (1024 * 1024)
            
            print()
            print("=" * 70)
            print("✅ ZINC250k Downloaded & Converted Successfully!")
            print("=" * 70)
            print(f"Location: {ZINC250K_FILE}")
            print(f"Molecules: {count:,}")
            print(f"Size: {size_mb:.2f} MB")
            print("=" * 70)
            print()
            
            return True
        
        elif smiles_files:
            # Use SMILES file directly
            src_file = smiles_files[0]
            print(f"📋 Found SMILES file: {src_file.name}")
            print()
            
            print(f"💾 Copying to: {ZINC250K_FILE}")
            shutil.copy(src_file, ZINC250K_FILE)
            
            # Verify
            count = sum(1 for _ in open(ZINC250K_FILE))
            size_mb = ZINC250K_FILE.stat().st_size / (1024 * 1024)
            
            print()
            print("=" * 70)
            print("✅ ZINC250k Downloaded Successfully!")
            print("=" * 70)
            print(f"Location: {ZINC250K_FILE}")
            print(f"Molecules: {count:,}")
            print(f"Size: {size_mb:.2f} MB")
            print("=" * 70)
            print()
            
            return True
        
        else:
            print("❌ No SMILES or CSV files found in downloaded data")
            print(f"   Contents of {download_path}:")
            for item in download_path.iterdir():
                print(f"      - {item.name}")
            return False
        
    except Exception as e:
        print(f"❌ Download failed: {str(e)}")
        print()
        print("Troubleshooting:")
        print("1. Install kagglehub: pip install kagglehub")
        print("2. Set up Kaggle credentials:")
        print("   - Go to: https://www.kaggle.com/account/")
        print("   - Click 'Create New API Token'")
        print("   - Save to: ~/.kaggle/kaggle.json")
        print("3. Try again")
        print()
        return False


def check_existing():
    """Check if dataset already exists"""
    if ZINC250K_FILE.exists():
        count = sum(1 for _ in open(ZINC250K_FILE))
        size_mb = ZINC250K_FILE.stat().st_size / (1024 * 1024)
        
        print("✅ ZINC250k already exists")
        print(f"   Location: {ZINC250K_FILE}")
        print(f"   Molecules: {count:,}")
        print(f"   Size: {size_mb:.2f} MB")
        print()
        return True
    
    return False


def main():
    print()
    
    # Check if already exists
    if check_existing():
        return
    
    # Download
    if download_from_kaggle():
        print("Ready to use ZINC250k with MAGNet! 🧬")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
