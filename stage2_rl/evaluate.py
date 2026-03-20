import argparse
import sys
import os
import torch
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import QED, Descriptors
from rdkit import RDLogger
from pathlib import Path
from tqdm import tqdm

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "stage1_hes"))
sys.path.insert(0, str(PROJECT_ROOT / "processing"))
sys.path.insert(0, str(PROJECT_ROOT))

from stage2_rl.environment.molecule_env import MoleculeEnv
from stage2_rl.models.sac_agent import SACAgent
from stage2_rl.train import load_stage1_components, initialize_training
from processing.utils.docking_estimator import estimate_docking_score_ecfp

def compute_sa_reward(mol):
    if mol is None:
        return 0.0
    try:
        # Use TPSA as proxy for SA (0-140 normally)
        sa_proxy = Descriptors.TPSA(mol) / 140.0
        # Normalize: higher SA proxy → lower reward
        r_sa = 1.0 - sa_proxy
        return float(np.clip(r_sa, 0.0, 1.0))
    except:
        return 0.0

def load_training_dataset(path):
    print(f"Loading reference dataset from {path}...")
    dataset = set()
    try:
        with open(path, "r") as f:
            for line in f:
                smiles = line.strip().split()[0]
                if smiles != "smiles": # header
                    dataset.add(smiles)
    except Exception as e:
        print(f"Warning: Could not load reference dataset: {e}")
    return dataset

def evaluate():
    parser = argparse.ArgumentParser(description="Evaluate Stage 2 SAC Agent")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of molecules to generate")
    parser.add_argument("--checkpoint", type=str, default=str(PROJECT_ROOT / "stage2_rl/checkpoints/agent_best.pt"), help="Agent checkpoint")
    parser.add_argument("--output", type=str, default="evaluation_results.csv", help="Output CSV file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("=" * 80)
    print("EVALUATING M-FRAG SAC AGENT")
    print("=" * 80)
    
    # 1. Setup Environment and Agent
    hes_model, motif_vocab, shape_vocab, property_scaler = load_stage1_components()
    env, agent, replay_buffer, reward_computer = initialize_training(
        hes_model, motif_vocab, shape_vocab, property_scaler
    )
    
    if os.path.exists(args.checkpoint):
        agent.load(args.checkpoint)
        print(f"[✓] Loaded agent from {args.checkpoint}")
    else:
        print(f"[x] Checkpoint not found at {args.checkpoint}!")
        return
        
    reference_data_path = PROJECT_ROOT / "data/smiles/zinc250k/zinc250k.smi"
    reference_dataset = load_training_dataset(reference_data_path)
    print(f"Loaded {len(reference_dataset)} reference SMILES.")
    
    docking_proteins = ["parp1", "fa7", "5ht1b", "braf", "jak2"]
    
    generated_smiles = []
    
    # Generation loop
    print(f"\nGenerating {args.num_samples} molecules...")
    for i in tqdm(range(args.num_samples)):
        state = env.reset() # This now randomly picks from STARTING_SCAFFOLDS in the env
        done = False
        step_count = 0
        
        while not done and step_count < 20: # MAX_STEPS_PER_EPISODE
            action_mask = env.get_action_mask()
            # USE STOCHASTIC POLICY (training=True) TO SAMPLE DIVERSE MOLECULES
            # If training=False, it uses the greedy deterministic action and always generates the exact same molecule!
            action = agent.select_action(state, training=True, action_mask=action_mask)
            next_state, reward, done, info = env.step(action)
            state = next_state
            step_count += 1
            
        final_smiles = info.get("final_smiles", None)
        if final_smiles is None and hasattr(env, "molecule_smiles"):
            final_smiles = env.molecule_smiles
            
        if final_smiles:
            generated_smiles.append(final_smiles)
            
    # Compute Metrics
    print("\nComputing metrics...")
    valid_count = 0
    valid_mols = []
    valid_smiles_list = []
    
    for s in generated_smiles:
        try:
            mol = Chem.MolFromSmiles(s)
            if mol is not None:
                Chem.SanitizeMol(mol)
                valid_count += 1
                valid_mols.append(mol)
                canon_s = Chem.MolToSmiles(mol)
                valid_smiles_list.append(canon_s)
        except:
            pass
            
    validity = valid_count / args.num_samples if args.num_samples > 0 else 0
    unique_smiles = set(valid_smiles_list)
    uniqueness = len(unique_smiles) / valid_count if valid_count > 0 else 0
    
    novel_smiles = unique_smiles - reference_dataset
    novelty = len(novel_smiles) / len(unique_smiles) if len(unique_smiles) > 0 else 0
    
    results = []
    
    print("Calculating properties for valid molecules...")
    for s, mol in tqdm(zip(valid_smiles_list, valid_mols), total=len(valid_mols)):
        qed_val = QED.qed(mol)
        sa_val = compute_sa_reward(mol)
        
        # Calculate docking
        docking_scores = []
        for prot in docking_proteins:
            score = estimate_docking_score_ecfp(s, prot)
            docking_scores.append(score)
            
        row = {
            "SMILES": s,
            "QED": qed_val,
            "SA": sa_val
        }
        for i, prot in enumerate(docking_proteins):
            row[f"Docking_{prot}"] = docking_scores[i]
            
        results.append(row)
        
    df_results = pd.DataFrame(results)
    if not df_results.empty:
        avg_qed = df_results["QED"].mean()
        avg_sa = df_results["SA"].mean()
        avg_docking = {prot: df_results[f"Docking_{prot}"].mean() for prot in docking_proteins}
    else:
        avg_qed = 0.0
        avg_sa = 0.0
        avg_docking = {prot: 0.0 for prot in docking_proteins}
        
    # Output
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)
    print(f"Total requested     : {args.num_samples}")
    print(f"Validity            : {validity*100:.2f}%")
    print(f"Uniqueness          : {uniqueness*100:.2f}%")
    print(f"Novelty             : {novelty*100:.2f}%")
    print("-" * 50)
    print(f"Average QED         : {avg_qed:.4f}")
    print(f"Average SA (proxy)  : {avg_sa:.4f}")
    print("-" * 50)
    for prot in docking_proteins:
        print(f"Average Docking {prot.upper():<5}: {avg_docking[prot]:.4f}")
    print("="*50)
    
    df_results.to_csv(args.output, index=False)
    print(f"\nAll valid generated SMILES and properties saved to {args.output}")

if __name__ == "__main__":
    evaluate()