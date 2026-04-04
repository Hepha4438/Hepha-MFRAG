import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import subprocess
import tempfile
import sys
from pathlib import Path
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

# ==============================================================================
# CONFIGURATIONS
# ==============================================================================
TARGETS = ['parp1', 'fa7', '5ht1b', 'braf', 'jak2']

# Box Parameters extracted from Saturn/GEAM
BOX_CONFIGS = {
    'fa7':   {'center': (10.131, 41.879, 32.097),   'size': (20.673, 20.198, 21.362)},
    'parp1': {'center': (26.413, 11.282, 27.238),   'size': (18.521, 17.479, 19.995)},
    '5ht1b': {'center': (-26.602, 5.277, 17.898),   'size': (22.5, 22.5, 22.5)},
    'jak2':  {'center': (114.758, 65.496, 11.345),  'size': (19.033, 17.929, 20.283)},
    'braf':  {'center': (84.194, 6.949, -7.081),    'size': (22.032, 19.211, 14.106)}
}

# The Target Thresholds for "Hit Ratio" specified by the user
THRESHOLDS = {
    'parp1': -9.7,
    'fa7': -7.5,
    '5ht1b': -8.8,
    'braf': -9.3,
    'jak2': -9.1
}

# Excutables installed via Conda
import shutil
VINA_EXECUTABLE = shutil.which("vina") or "vina"
OBABEL_EXECUTABLE = shutil.which("obabel") or "obabel"

# Directory to reference the .pdbqt grid receptors
RECEPTOR_DIR = Path("saturn/oracles/docking/docking_grids").absolute()


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def calculate_mpo_scores(df, target):
    """
    Step 1.4: Compute MPO Score.
    Favoring: High QED, Low SA (meaning easier synthesis), Low/Negative Docking.
    Using Z-score standardization.
    MPO = Z(QED) - Z(SA) - Z(Docking)
    """
    dock_col = f'Docking_{target}'
    z_qed = (df['QED'] - df['QED'].mean()) / df['QED'].std()
    z_sa = (df['SA'] - df['SA'].mean()) / df['SA'].std()
    z_dock = (df[dock_col] - df[dock_col].mean()) / df[dock_col].std()
    
    # Combined Multi-Parameter Optimization Score
    df[f'MPO_{target}'] = z_qed - z_sa - z_dock
    return df

def plot_distributions(df, target, output_dir):
    """
    Step 2.3: Plot the QED, SA, and logP distributions for Top 1000.
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    sns.histplot(df['QED'], kde=True, ax=axes[0], color='blue').set_title(f'{target.upper()} - Top 1000 QED')
    sns.histplot(df['SA'], kde=True, ax=axes[1], color='red').set_title(f'{target.upper()} - Top 1000 SA')
    sns.histplot(df['logP'], kde=True, ax=axes[2], color='green').set_title(f'{target.upper()} - Top 1000 logP')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'top1000_dist_{target}.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()


def evaluate_target(df, target, output_dir):
    """Evaluate one target and return summary dict."""
    os.makedirs(output_dir, exist_ok=True)

    # 2. Compute MPO Score
    df = calculate_mpo_scores(df.copy(), target)

    # 3. Extract Top 1000
    df_target_1000 = df.sort_values(f'MPO_{target}', ascending=False).head(1000).copy()

    # 4. Calculate LogP
    tqdm.pandas(desc="    Calculating LogP")
    df_target_1000['logP'] = df_target_1000['SMILES'].progress_apply(
        lambda x: Descriptors.MolLogP(Chem.MolFromSmiles(x)) if Chem.MolFromSmiles(x) else np.nan
    )
    df_target_1000 = df_target_1000.dropna(subset=['logP'])

    # 5. Plot Top 1000 Distributions
    plot_distributions(df_target_1000, target, output_dir=os.path.join(output_dir, 'evaluation_plots'))
    print(f"    [+] Saved Top 1000 distribution plots to {output_dir}/evaluation_plots")

    # 6. Extract Top 100 for Physical Docking
    df_top_100 = df_target_1000.head(100).copy()

    # 7. Run Physical Vina Docking
    print(f"    [*] Running Physical Vina Docking for {len(df_top_100)} molecules...")
    physical_scores = []
    for smi in tqdm(df_top_100['SMILES'].tolist(), desc="    Docking"):
        score = run_physical_vina_docking(smi, target)
        physical_scores.append(score)

    df_top_100[f'Physical_Vina_{target}'] = physical_scores

    # 8. Calculate Rigorous Hit Ratio
    thresh = THRESHOLDS[target]
    hits = df_top_100[
        (df_top_100['QED'] > 0.5) &
        (df_top_100['SA'] < 5.0) &
        (df_top_100[f'Physical_Vina_{target}'] <= thresh) &
        (df_top_100[f'Physical_Vina_{target}'] != 0.0)
    ]

    hit_ratio = (len(hits) / len(df_top_100)) * 100.0 if len(df_top_100) > 0 else 0.0
    print(f"    [+] {target.upper()} Hit Ratio: {hit_ratio:.2f}% (Threshold <= {thresh})")

    # Save exact result per target for tracking
    top100_path = os.path.join(output_dir, f"top100_{target}_real_docking.csv")
    df_top_100.to_csv(top100_path, index=False)

    return {
        'Target': target.upper(),
        'Target Threshold': thresh,
        'Total Top Evaluated': len(df_top_100),
        'Strict Hit Count': len(hits),
        'Hit Ratio (%)': round(hit_ratio, 2)
    }

def run_physical_vina_docking(smiles, target):
    """
    Step 3.4: Convert SMILES -> 3D RDKit -> obabel .pdbqt -> Physical Docking
    """
    center = BOX_CONFIGS[target]['center']
    size = BOX_CONFIGS[target]['size']
    receptor_file = RECEPTOR_DIR / f"{target}.pdbqt"
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # File paths
        mol_path = os.path.join(tmpdir, "ligand.mol")
        pdbqt_path = os.path.join(tmpdir, "ligand.pdbqt")
        out_pdbqt_path = os.path.join(tmpdir, "out.pdbqt")
        
        # 1. RDKit to 3D Mol
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return 0.0 # Error parsing SMILES
        try:
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
            Chem.MolToMolFile(mol, mol_path)
        except Exception:
            return 0.0 # RDKit fail to embed 3D
        
        # 2. OpenBabel: Mol -> PDBQT
        try:
            subprocess.run(
                [OBABEL_EXECUTABLE, "-imol", mol_path, "-opdbqt", "-O", pdbqt_path],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except Exception:
            return 0.0
            
        # 3. Run AutoDock Vina
        cmd = [
            VINA_EXECUTABLE,
            "--receptor", str(receptor_file),
            "--ligand", pdbqt_path,
            "--out", out_pdbqt_path,
            "--center_x", str(center[0]), "--center_y", str(center[1]), "--center_z", str(center[2]),
            "--size_x", str(size[0]), "--size_y", str(size[1]), "--size_z", str(size[2]),
            "--cpu", "1",
            "--exhaustiveness", "1"  # Fast mode
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            log_output = result.stdout
            
            # The FIRST binding mode is the best one -> parse Vina Score (kcal/mol)
            for line in log_output.split('\n'):
                parts = line.split()
                if len(parts) >= 2 and parts[0] == '1':
                    return float(parts[1])
            return 0.0
        except Exception:
            return 0.0

# ==============================================================================
# PIPELINE EXECUTION
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Final physical-docking evaluation and hit-ratio computation")
    parser.add_argument("--target", type=str, default="all", choices=TARGETS + ["all"], help="Target protein or 'all'")
    parser.add_argument("--input-csv", type=str, default="evaluation_results.csv", help="Input CSV for single-target mode (or fallback)")
    parser.add_argument("--input-template", type=str, default="stage2_rl/evaluation_results_{target}.csv", help="Input template for all-target mode")
    parser.add_argument("--output-dir", type=str, default="stage2_rl/final_eval", help="Directory to save final evaluation outputs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    summary_results = []

    selected_targets = TARGETS if args.target == "all" else [args.target]

    for target in selected_targets:
        if args.target == "all":
            candidate_path = args.input_template.format(target=target)
            csv_file = candidate_path if os.path.exists(candidate_path) else args.input_csv
        else:
            csv_file = args.input_csv

        if not os.path.exists(csv_file):
            print(f"\n[WARN] Input CSV not found for {target}: {csv_file}. Skipping.")
            continue

        print(f"\n{'='*50}\n[*] Evaluating Target: {target.upper()}\n{'='*50}")
        print(f"[*] Step 1: Loading raw data from {csv_file}")

        df = pd.read_csv(csv_file)
        if 'SMILES' not in df.columns:
            print(f"[WARN] Missing SMILES column in {csv_file}. Skipping {target}.")
            continue

        df = df.drop_duplicates(subset=['SMILES']).reset_index(drop=True)
        print(f"[*] Total unique molecules: {len(df)}")

        target_output_dir = os.path.join(args.output_dir, target)
        summary = evaluate_target(df, target, target_output_dir)
        summary_results.append(summary)

    if not summary_results:
        print("\n[WARN] No target evaluation completed successfully.")
        return

    summary_df = pd.DataFrame(summary_results)
    print("\n\n" + "="*50)
    print("FINAL HIT RATIO SUMMARY TABLE")
    print("="*50)
    print(summary_df.to_string(index=False))

    summary_path = os.path.join(args.output_dir, "FINAL_HIT_RATIO_SUMMARY.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary to: {summary_path}")
    print("="*50)

if __name__ == "__main__":
    main()
