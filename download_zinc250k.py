#!/usr/bin/env python
"""
Download ZINC250k from Kaggle and convert to SMILES
Keeps both CSV and SMILES formats
"""

import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.absolute()
DATA_DIR = PROJECT_ROOT / "data" / "smiles" / "zinc250k"
DATA_DIR.mkdir(parents=True, exist_ok=True)

ZINC250K_CSV = DATA_DIR / "zinc250k.csv"
ZINC250K_SMILES = DATA_DIR / "zinc250k.smi"


def download_from_kaggle():
    """Download ZINC250k from Kaggle using kagglehub"""
    
    print("📥 Downloading ZINC250k from Kaggle...")
    print()
    
    try:
        import kagglehub
        
        # Download latest version
        path = kagglehub.dataset_download("basu369victor/zinc250k")
        
        print(f"✓ Downloaded to: {path}")
        print()
        
        # Find CSV file in downloaded directory
        downloaded_dir = Path(path)
        csv_files = list(downloaded_dir.glob("*.csv"))
        
        if not csv_files:
            print("❌ No CSV file found in downloaded data")
            return None
        
        # Use first CSV found
        csv_file = csv_files[0]
        print(f"📄 Found CSV: {csv_file.name}")
        
        return csv_file
        
    except ImportError:
        print("❌ kagglehub not installed")
        print("   Install with: pip install kagglehub")
        return None
    except Exception as e:
        print(f"❌ Download failed: {str(e)}")
        return None


def copy_csv(source_csv: Path) -> bool:
    """Copy CSV to project data directory"""
    try:
        print(f"💾 Copying CSV...")
        
        # Read and write CSV
        df = pd.read_csv(source_csv)
        df.to_csv(ZINC250K_CSV, index=False)
        
        count = len(df)
        size = ZINC250K_CSV.stat().st_size / (1024 * 1024)
        
        print(f"✓ Saved: {ZINC250K_CSV.name}")
        print(f"   Rows: {count:,}")
        print(f"   Size: {size:.2f} MB")
        print()
        
        return df
        
    except Exception as e:
        print(f"❌ Copy failed: {str(e)}")
        return None


def convert_to_smiles(df) -> bool:
    """Convert CSV to SMILES format"""
    
    try:
        print("🧬 Converting to SMILES format...")
        
        # Try common column names for SMILES
        smiles_col = None
        for col in ["smiles", "SMILES", "smi", "SMI", "canonical_smiles"]:
            if col in df.columns:
                smiles_col = col
                break
        
        if smiles_col is None:
            print(f"❌ SMILES column not found. Available columns: {df.columns.tolist()}")
            return False
        
        # Write SMILES to file (one per line)
        smiles_list = df[smiles_col].dropna().unique()
        
        with open(ZINC250K_SMILES, 'w') as f:
            for smi in smiles_list:
                f.write(str(smi).strip() + '\n')
        
        size = ZINC250K_SMILES.stat().st_size / (1024 * 1024)
        
        print(f"✓ Saved: {ZINC250K_SMILES.name}")
        print(f"   Molecules: {len(smiles_list):,}")
        print(f"   Size: {size:.2f} MB")
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ Conversion failed: {str(e)}")
        return False


def main():
    print()
    print("=" * 70)
    print("ZINC250k Kaggle Downloader")
    print("=" * 70)
    print()
    
    # Check if already have both files
    if ZINC250K_CSV.exists() and ZINC250K_SMILES.exists():
        csv_size = ZINC250K_CSV.stat().st_size / (1024 * 1024)
        smi_size = ZINC250K_SMILES.stat().st_size / (1024 * 1024)
        csv_rows = len(pd.read_csv(ZINC250K_CSV))
        smi_rows = sum(1 for _ in open(ZINC250K_SMILES))
        
        print("✅ Both files already exist")
        print(f"   CSV: {csv_rows:,} molecules ({csv_size:.2f} MB)")
        print(f"   SMILES: {smi_rows:,} molecules ({smi_size:.2f} MB)")
        print()
        return
    
    # Download from Kaggle
    csv_file = download_from_kaggle()
    if csv_file is None:
        sys.exit(1)
    
    # Copy CSV
    df = copy_csv(csv_file)
    if df is None:
        sys.exit(1)
    
    # Convert to SMILES
    if not convert_to_smiles(df):
        sys.exit(1)
    
    print("=" * 70)
    print("✅ ZINC250k ready!")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
