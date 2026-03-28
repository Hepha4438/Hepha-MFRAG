import os

# Suppress RDKit deprecation warnings
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Monkey-patch os.makedirs to ignore FileExistsError globally
original_makedirs = os.makedirs
def safe_makedirs(name, mode=0o777, exist_ok=False):
    # Always set exist_ok=True to prevent race conditions during Joblib multiprocessing
    original_makedirs(name, mode=mode, exist_ok=True)
os.makedirs = safe_makedirs

import sys
import pandas as pd
import numpy as np
import subprocess
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
import joblib
from joblib import Parallel, delayed
import uuid
import shutil

# -------------------------------------------------------------------------
# MONKEY-PATCH SUBPROCESS TO BYPASS HARDCODED LINUX EXECUTABLES IN SATURN
# -------------------------------------------------------------------------

# Discover vina binary location robustly
_VINA_BIN = None
if _VINA_BIN is None:
    # Method 1: Look in current Python environment
    candidate = os.path.join(sys.prefix, 'bin', 'vina')
    if os.path.exists(candidate):
        _VINA_BIN = candidate
    else:
        # Method 2: Use shutil.which which respects env PATH
        _VINA_BIN = shutil.which('vina')
        
if _VINA_BIN:
    print(f"[INIT] Detected vina binary: {_VINA_BIN}", flush=True)
else:
    print(f"[WARN] vina binary not found - docking may fail", flush=True)

original_check_output = subprocess.check_output

def patched_check_output(args, **kwargs):
    if isinstance(args, list):
        args_copy = list(args)
        
        # Catch and replace broken Linux binary path with our global 'vina'
        if len(args_copy) > 0 and ('qvina02' in args_copy[0] or 'qvina' in args_copy[0]):
            args_copy[0] = _VINA_BIN if _VINA_BIN else 'vina'
            print(f"[SUBPROCESS PATCH] Replaced with: {args_copy[0]}", flush=True)
            
        try:
            return original_check_output(args_copy, **kwargs)
        except TypeError as e:
            # Handle possible python version kwarg typoes in Saturn e.g. universal_newline
            if "unexpected keyword argument 'universal_newline'" in str(e) and 'universal_newline' in kwargs:
                kwargs['universal_newlines'] = kwargs.pop('universal_newline')
                return original_check_output(args_copy, **kwargs)
            raise
            
    return original_check_output(args, **kwargs)

# Apply subprocess patch globally before importing Saturn
subprocess.check_output = patched_check_output

# Add Saturn to path *before* initiating processes
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(PROJECT_ROOT, 'saturn'))

from oracles.docking.quickvina2 import DockingVina

# -------------------------------------------------------------------------
# PATCH SATURN'S VINA OUTPUT PARSING BUG (list index out of range)
# -------------------------------------------------------------------------
_orig_docking_method = DockingVina.docking

def _patched_docking(self, receptor_file, ligand_mol_file, ligand_pdbqt_file, docking_pdbqt_file):
    """Patched version of docking() with safety checks for vina output parsing."""
    from openbabel import pybel
    
    ms = list(pybel.readfile("mol", ligand_mol_file))
    m = ms[0]
    m.write("pdbqt", ligand_pdbqt_file, overwrite=True)
    run_line = '%s --receptor %s --ligand %s --out %s' % (self.vina_program,
                                                          receptor_file, ligand_pdbqt_file, docking_pdbqt_file)
    run_line += ' --center_x %s --center_y %s --center_z %s' %(self.box_center)
    run_line += ' --size_x %s --size_y %s --size_z %s' %(self.box_size)
    run_line += ' --cpu %d' % (self.num_cpu_dock)
    run_line += ' --num_modes %d' % (self.num_modes)
    run_line += ' --exhaustiveness %d ' % (self.exhaustiveness)
    result = subprocess.check_output(run_line.split(),
                                     stderr=subprocess.STDOUT,
                                     timeout=self.timeout_dock, universal_newlines=True)
    result_lines = result.split('\n')

    check_result = False
    affinity_list = list()
    for result_line in result_lines:
        if result_line.startswith('-----+'):
            check_result = True
            continue
        if not check_result:
            continue
        if result_line.startswith('Writing output'):
            break
        if result_line.startswith('Refine time'):
            break
        lis = result_line.strip().split()
        if len(lis) < 2:  # FIX: Safety check for malformed lines
            continue
        if not lis[0].isdigit():
            break
        affinity = float(lis[1])
        affinity_list += [affinity]
    return affinity_list

DockingVina.docking = _patched_docking

# Detect grid location (use local data/docking_grids, NOT Saturn's grids)
SATURN_GRIDS = os.path.join(PROJECT_ROOT, 'data', 'docking_grids')
if not os.path.exists(SATURN_GRIDS):
    # Fallback to Saturn's grids if local not found
    SATURN_GRIDS = os.path.join(PROJECT_ROOT, 'saturn', 'oracles', 'docking', 'docking_grids')
if not os.path.exists(SATURN_GRIDS):
    print(f"[WARN] Grid directory not found at {SATURN_GRIDS}", flush=True)
print(f"[INIT] Docking grids located at: {SATURN_GRIDS}", flush=True)

sys.path.append(os.path.join(PROJECT_ROOT, 'processing'))
from utils.fingerprint_utils import compute_ecfp

# -------------------------------------------------------------------------
# MONKEY-PATCH DOCKING VINA TO WORK SAFELY WITH JOBLIB PARALLEL
# Note: To survive Joblib Loky backend serialization, we must re-apply 
# the patches directly inside the worker.
# -------------------------------------------------------------------------
def apply_worker_patches():
    # Patch subprocess
    import subprocess
    if getattr(subprocess, '_patched_for_saturn', False) is False:
        _orig_check = subprocess.check_output
        def _patched_check(args, **kwargs):
            if isinstance(args, list) and len(args)>0 and ('qvina02' in args[0] or 'qvina' in args[0]):
                args[0] = _VINA_BIN if _VINA_BIN else 'vina'
            try:
                return _orig_check(args, **kwargs)
            except TypeError as e:
                if "unexpected keyword argument 'universal_newline'" in str(e) and 'universal_newline' in kwargs:
                    kwargs['universal_newlines'] = kwargs.pop('universal_newline')
                    return _orig_check(args, **kwargs)
                raise
        subprocess.check_output = _patched_check
        subprocess._patched_for_saturn = True

    # Patch os.makedirs to be race-condition safe
    import os
    if getattr(os, '_patched_for_saturn', False) is False:
        _orig_makedirs = os.makedirs
        def _safe_makedirs(name, mode=0o777, exist_ok=False):
            _orig_makedirs(name, mode=mode, exist_ok=True)
        os.makedirs = _safe_makedirs
        os._patched_for_saturn = True

    # Patch DockingVina: Replace Saturn's sequential temp dir logic with UUIDs
    if getattr(DockingVina, '_patched_for_saturn', False) is False:
        _orig_init = DockingVina.__init__
        
        def _patched_init(self, target):
            # Manually set box center/size WITHOUT calling original __init__ yet
            # (This avoids Saturn's sequential tmp0, tmp1, tmp2 logic entirely)
            if target == 'fa7':
                self.box_center = (10.131, 41.879, 32.097)
                self.box_size = (20.673, 20.198, 21.362)
            elif target == 'parp1':
                self.box_center = (26.413, 11.282, 27.238)
                self.box_size = (18.521, 17.479, 19.995)
            elif target == '5ht1b':
                self.box_center = (-26.602, 5.277, 17.898)
                self.box_size = (22.5, 22.5, 22.5)
            elif target == 'jak2':
                self.box_center = (114.758, 65.496, 11.345)
                self.box_size = (19.033, 17.929, 20.283)
            elif target == 'braf':
                self.box_center = (84.194, 6.949, -7.081)
                self.box_size = (22.032, 19.211, 14.106)
            
            # Set paths to local Saturn grids
            self.vina_program = os.path.join(SATURN_GRIDS, 'qvina02')
            self.receptor_file = os.path.join(SATURN_GRIDS, f'{target}.pdbqt')
            
            # Config parameters
            self.exhaustiveness = 1
            self.num_sub_proc = 1  # Disable recursive multiprocessing
            self.num_cpu_dock = 5
            self.num_modes = 10
            self.timeout_gen3d = 30
            self.timeout_dock = 100
            
            # Create UNIQUE temp directory using UUID instead of sequential nums
            self.temp_dir = f'tmp/tmp_{uuid.uuid4().hex[:8]}'
            os.makedirs(self.temp_dir, exist_ok=True)
        
        DockingVina.__init__ = _patched_init
        DockingVina._patched_for_saturn = True

# Apply them aggressively in the main process too
apply_worker_patches()


def load_dataset(sample_size=1000):
    dataset_path = os.path.join(PROJECT_ROOT, 'data/smiles/zinc250k/zinc250k.csv')
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    df = pd.read_csv(dataset_path)
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    return df['smiles'].tolist()


# Top-level docking function - NOW WITH MOCK MODE FOR QUICK TESTING
def compute_docking_sequential(smiles_list, use_mock=False):
    """
    Compute docking scores sequentially in the MAIN PROCESS ONLY.
    
    If use_mock=True, generates random scores for rapid prototyping.
    This allows us to test the full RF training pipeline while we debug Saturn's vina parsing.
    """
    target_proteins = ['parp1', 'fa7', '5ht1b', 'braf', 'jak2']
    checkpoint_file = os.path.join(PROJECT_ROOT, "surrogate_training_checkpoint.csv")
    
    if use_mock:
        print("[MOCK MODE] Generating synthetic docking scores instead of real Vina calls.")
        np.random.seed(42)
        results = []
        for smiles in tqdm(smiles_list, desc="Mock Docking"):
            row = {"smiles": smiles}
            for target in target_proteins:
                # Generate realistic-looking affinity scores (typically -12 to 0)
                row[target] = np.random.uniform(-12, -3)
            results.append(row)
        
        df = pd.DataFrame(results)
        df.to_csv(checkpoint_file, index=False)
        return df
    
    # REAL DOCKING (original code)
    processed_smiles = set()
    if os.path.exists(checkpoint_file):
        df_checkpoint = pd.read_csv(checkpoint_file)
        processed_smiles = set(df_checkpoint['smiles'].values)
        print(f"[CHECKPOINT] Loaded {len(processed_smiles)} previously processed molecules")
    
    to_process = [s for s in smiles_list if s not in processed_smiles]
    if not to_process:
        print("All molecules already processed! Reading from checkpoint.")
        return pd.read_csv(checkpoint_file)
    
    print(f"[DOCKING] Scoring {len(to_process)} molecules sequentially...")
    print(f"[DOCKING] NOTE: Running in main process (Jobs stay at 1 to avoid nested multiprocessing)")
    
    results = []
    for i, smiles in tqdm(enumerate(to_process), total=len(to_process), desc="Vina Docking"):
        row = {"smiles": smiles}
        
        for target in target_proteins:
            try:
                oracle = DockingVina(target=target)
                res = oracle.predict([smiles])
                row[target] = res[0] if res and len(res) > 0 else np.nan
                
            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)[:150]}"
                print(f"  [ERR] {smiles[:40]} + {target} -> {error_msg}", flush=True)
                row[target] = np.nan
            finally:
                if 'oracle' in locals() and hasattr(oracle, "temp_dir"):
                    temp = oracle.temp_dir
                    if os.path.exists(temp):
                        shutil.rmtree(temp, ignore_errors=True)
        
        results.append(row)
        
        # Checkpoint every 50 molecules for crash recovery
        if (i + 1) % 50 == 0:
            df_batch = pd.DataFrame(results)
            results = []
            
            if not os.path.exists(checkpoint_file):
                df_batch.to_csv(checkpoint_file, index=False)
            else:
                df_batch.to_csv(checkpoint_file, mode='a', header=False, index=False)
            
            print(f"  [CHECKPOINT] Saved {i+1} molecules", flush=True)
    
    # Final batch
    if results:
        df_final = pd.DataFrame(results)
        if not os.path.exists(checkpoint_file):
            df_final.to_csv(checkpoint_file, index=False)
        else:
            df_final.to_csv(checkpoint_file, mode='a', header=False, index=False)
    
    return pd.read_csv(checkpoint_file)


def generate_docking_ground_truths_parallel(smiles_list, target_proteins=None, chunk_size=100, use_mock=True):
    """
    WRAPPER: Calls sequential docking function (no parallelization due to Saturn Manager incompatibility).
    
    use_mock=True: Returns synthetic random docking scores for pipeline testing
    use_mock=False: Attempts real Vina docking (slow ~77s/molecule, currently has parsing issues)
    """
    return compute_docking_sequential(smiles_list, use_mock=use_mock)

def train_main():
    output_dir = os.path.join(PROJECT_ROOT, "processing/output/models")
    os.makedirs(output_dir, exist_ok=True)
    
    target_proteins = ['parp1', 'fa7', '5ht1b', 'braf', 'jak2']
    
    print("1. Loading dataset (1,000 molecules)...")
    smiles_list = load_dataset(sample_size=1000)
    
    print("2. Generating ground truths via Vina (with chunking + parallelism)...")
    df_scores = generate_docking_ground_truths_parallel(smiles_list, chunk_size=100, use_mock=False)
    
    # Subset in case checkpoint accrued more somehow
    df_scores = df_scores[df_scores['smiles'].isin(smiles_list)]
    
    print("\n3. Computing ECFP features (2048-bit radius=2)...")
    features = []
    valid_idx = []
    
    for idx, row in tqdm(df_scores.iterrows(), total=len(df_scores), desc="ECFP Gen"):
        fp = compute_ecfp(row['smiles'])
        if fp is not None:
            features.append(fp)
            valid_idx.append(idx)
            
    X = np.array(features)
    df_valid = df_scores.iloc[valid_idx].copy()
    
    print(f"\n4. Training Surrogate Random Forest Models on dataset shape: X={X.shape}")
    for target in target_proteins:
        target_mask = ~df_valid[target].isna()
        X_target = X[target_mask]
        y_target = df_valid[target][target_mask].values
        
        if len(y_target) == 0:
            print(f"Skipping {target} because no valid docking scores were found.")
            continue
            
        print(f"  -> Training RF for {target} on {len(y_target)} valid samples...")
        rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, max_depth=15, random_state=42)
        rf.fit(X_target, y_target)
        
        save_path = os.path.join(output_dir, f"rf_{target}.joblib")
        joblib.dump(rf, save_path)
        print(f"  -> Saved surrogate {target} model: {save_path}")
        
    print("\nSurrogate pipeline complete.")

if __name__ == "__main__":
    train_main()
