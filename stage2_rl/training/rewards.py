"""
Reward Computer: Calculate molecular generation rewards
Implements property-based reward shaping as per Stage 2 specification

Key workflow:
1. HES model outputs Y_raw (raw property predictions)
2. StandardScaler (from Stage 1) normalizes: Y_norm = (Y_raw - mean) / scale
3. Reward computed using normalized Y: R_prop,i = exp(-(Y_norm_i - Y*_i)² / (2σ_i²))
"""
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import QED, Crippen, Descriptors, AllChem
import pickle
from pathlib import Path
from typing import Tuple, Optional
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "stage1_hes"))
sys.path.insert(0, str(PROJECT_ROOT / "processing"))

from stage2_rl.training.config import *


class RewardComputer:
    """
    Compute rewards for molecular generation based on property improvements.
    
    Workflow:
    1. Get Y from HES model (raw property predictions)
    2. Normalize Y using StandardScaler: Y_norm = (Y - mean) / scale
    3. Compute Gaussian reward: R_prop = exp(-(Y_norm - Y*)²/(2σ²))
    
    Reward structure:
    - r_A1: after scaffold extension (Phase A1)
    - r_A2: after atom/bond labeling (Phase A2)
    - r_terminal: at episode end based on overall property change
    """
    
    def __init__(self, property_scaler_path: Path = STAGE1_SCALER,
                 hes_model = None,
                 target_properties: np.ndarray = None,
                 property_sigma: np.ndarray = None,
                 hes_output_is_normalized: bool = HES_OUTPUT_IS_NORMALIZED,
                 target_properties_are_normalized: bool = TARGET_PROPERTIES_ARE_NORMALIZED):
        """
        Initialize reward computer with HES model and StandardScaler.
        
        Args:
            property_scaler_path: Path to pickled StandardScaler from Stage 1 (for Y normalization)
            hes_model: Trained HES model for Y prediction
            target_properties: Y_i* HYPERPARAMETER (target values for each property, in normalized space) [8,]
            property_sigma: σ_i HYPERPARAMETER (tolerance for each property, in normalized space) [8,]
            hes_output_is_normalized: If True, skip StandardScaler.transform on HES outputs
            target_properties_are_normalized: If False, apply SS.transform to Y* once at init
        
        CRITICAL: 
        - Y* and σ_i are USER-SPECIFIED HYPERPARAMETERS, not derived from dataset
        - StandardScaler is for NORMALIZATION ONLY: Y_norm = (Y_raw - mean) / scale
        - HES outputs Y_raw, which must be normalized before reward computation
        """
        with open(property_scaler_path, 'rb') as f:
            self.property_scaler = pickle.load(f)
        
        self.hes_model = hes_model
        self.hes_output_is_normalized = hes_output_is_normalized
        self.target_properties_are_normalized = target_properties_are_normalized
        
        # Target properties Y_i* - HYPERPARAMETER (must be provided by user)
        if target_properties is None:
            # Default: use [0, 0, 0, ...] (normalized space)
            # User should override with actual target values
            self.target_properties = np.zeros(NUM_PROPERTIES, dtype=np.float32)
            print("[WARNING] Y* not provided, using zeros (normalized space)")
        else:
            self.target_properties = np.array(target_properties, dtype=np.float32)

        if not self.target_properties_are_normalized:
            self.target_properties = self.property_scaler.transform(
                self.target_properties.reshape(1, -1)
            )[0].astype(np.float32)
        
        # Tolerance σ_i - HYPERPARAMETER (must be provided by user)
        if property_sigma is None:
            # Default: use 1.0 for all (reasonable default in normalized space)
            # User should override if needed
            self.property_sigma = np.ones(NUM_PROPERTIES, dtype=np.float32)
            print("[WARNING] σ_i not provided, using 1.0 for all properties")
        else:
            self.property_sigma = np.array(property_sigma, dtype=np.float32)
        
        print(f"[✓] Loaded property scaler from {property_scaler_path}")
        print(f"    Scaler mean (for normalization): {self.property_scaler.mean_}")
        print(f"    Scaler scale (for normalization): {self.property_scaler.scale_}")
        print(f"    Target properties Y_i* (HYPERPARAMETER): {self.target_properties}")
        print(f"    Tolerances σ_i (HYPERPARAMETER): {self.property_sigma}")
        print(f"    HES output normalized: {self.hes_output_is_normalized}")
        print(f"    Y* normalized: {self.target_properties_are_normalized}")
    
    def compute_properties(self, mol: Optional[Chem.Mol]) -> np.ndarray:
        """
        Compute Y properties for a molecule using HES model.
        
        Workflow:
        1. Convert Chem.Mol → PyTorch Geometric Data (atomic graph G + scaffold graph Sc)
        2. Pass through HES model
        3. Return Y_raw (raw output from property_head) ∈ ℝ^8
        
        Note: HES output Y_raw will be normalized by compute_property_reward() 
        using StandardScaler: Y_norm = (Y_raw - mean) / scale
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            Y_raw: (8,) numpy array of raw property predictions from HES
        """
        if mol is None:
            return np.zeros(NUM_PROPERTIES, dtype=np.float32)
        
        # If HES model not available, return zeros (placeholder)
        if self.hes_model is None:
            return np.zeros(NUM_PROPERTIES, dtype=np.float32)
        
        try:
            # Import graph builder to convert Chem.Mol → PyTorch Geometric Data
            from processing.utils.graph_builder import build_graph_simple
            
            # Get SMILES from molecule
            smiles = Chem.MolToSmiles(mol)
            if smiles is None:
                return np.zeros(NUM_PROPERTIES, dtype=np.float32)
            
            # Build PyTorch Geometric Data object (atomic graph G)
            data_g = build_graph_simple(smiles)
            if data_g is None:
                return np.zeros(NUM_PROPERTIES, dtype=np.float32)
            
            # For Stage 2, treat molecule as both G and Sc (temporary)
            # In full implementation, would extract actual scaffold
            data_sc = build_graph_simple(smiles)  # Same as G for now
            
            # Get device from HES model
            device = next(self.hes_model.parameters()).device
            
            # Move data to device
            data_g = data_g.to(device)
            data_sc = data_sc.to(device)
            data_g.x = data_g.x.float()
            data_sc.x = data_sc.x.float()

            # Match Stage 1 HES input dimensions: G=15, Sc=16
            if data_g.x.dim() == 1:
                data_g.x = data_g.x.unsqueeze(-1)
            if data_g.x.size(1) < 15:
                pad = torch.zeros(data_g.x.size(0), 15 - data_g.x.size(1), device=device)
                data_g.x = torch.cat([data_g.x, pad], dim=1)
            elif data_g.x.size(1) > 15:
                data_g.x = data_g.x[:, :15]

            if data_sc.x.dim() == 1:
                data_sc.x = data_sc.x.unsqueeze(-1)
            if data_sc.x.size(1) < 16:
                pad = torch.zeros(data_sc.x.size(0), 16 - data_sc.x.size(1), device=device)
                data_sc.x = torch.cat([data_sc.x, pad], dim=1)
            elif data_sc.x.size(1) > 16:
                data_sc.x = data_sc.x[:, :16]
            
            # Create batch tensors (single molecule = batch size 1, all nodes belong to graph 0)
            batch_g = torch.zeros(data_g.num_nodes, dtype=torch.long, device=device)
            batch_sc = torch.zeros(data_sc.num_nodes, dtype=torch.long, device=device)
            
            # Forward pass through HES model
            with torch.no_grad():
                outputs = self.hes_model(
                    x_g=data_g.x,
                    edge_index_g=data_g.edge_index,
                    edge_attr_g=data_g.edge_attr if hasattr(data_g, 'edge_attr') else None,
                    x_sc=data_sc.x,
                    edge_index_sc=data_sc.edge_index,
                    edge_attr_sc=data_sc.edge_attr if hasattr(data_sc, 'edge_attr') else None,
                    batch_g=batch_g,
                    batch_sc=batch_sc,
                )
            
            # Extract property predictions (Y_raw)
            prop_pred = outputs["prop_pred"]  # Shape: (1, 8)
            Y_raw = prop_pred[0].cpu().numpy()  # Shape: (8,)
            
            return Y_raw.astype(np.float32)
            
        except Exception as e:
            print(f"[WARNING] HES property prediction failed: {e}")
            return np.zeros(NUM_PROPERTIES, dtype=np.float32)
    
    def compute_property_reward(self, properties: np.ndarray, is_terminal: bool = False) -> float:
        """
        Compute R_prop from properties using Gaussian reward function.
        
        CRITICAL workflow:
        1. Input properties (raw Y from HES)
        2. Normalize: Y_norm = StandardScaler.transform(Y)
        3. Compute reward: R_prop,i = exp(-(Y_norm_i - Y_i*)^2 / (2*σ_i^2))
        
        Args:
            properties: (8,) array of raw molecular properties from HES
            is_terminal: whether to exclude QED and SA for terminal reward
        
        Returns:
            R_prop: float, average Gaussian reward across all properties
        """
        if properties.ndim == 1:
            properties = properties.reshape(1, -1)
        
        if self.hes_output_is_normalized:
            properties_normalized = properties
        else:
            # Normalize properties using StandardScaler from Stage 1
            # This applies: Y_norm = (Y_raw - mean) / scale
            properties_normalized = self.property_scaler.transform(properties)  # Shape: (batch, 8)
        
        # Compute Gaussian reward for each property using NORMALIZED values
        # R_prop,i = exp(-(Y_norm_i - Y_i*)^2 / (2*σ_i^2))
        gaussian_rewards = np.exp(
            -((properties_normalized - self.target_properties) ** 2) / (2 * self.property_sigma ** 2)
        )  # Shape: (batch, 8)
        
        # Use all 8 properties, but apply weights
        from stage2_rl.training import config
        
        weights = np.zeros(8)
        # LogP
        weights[0] = 0.0  # By default, config doesn't have W_LOGP, taking up remainder if any, but let's leave 0.0
        # QED and SA
        weights[1] = config.W_QED
        weights[2] = config.W_SA
        # 5 docking scores share W_DOCKING
        weights[3:8] = config.W_DOCKING / 5.0
        
        if is_terminal:
            # Exclude QED and SA because they are computed natively by RDKit
            weights[1] = 0.0
            weights[2] = 0.0
            
        weight_sum = np.sum(weights)
        if weight_sum > 0:
            r_prop = np.sum(gaussian_rewards * weights, axis=1) / weight_sum
        else:
            r_prop = np.mean(gaussian_rewards, axis=1)
            
        return r_prop[0] if len(r_prop) == 1 else r_prop
    
    def compute_qed_reward(self, mol: Optional[Chem.Mol]) -> float:
        """
        Compute R_QED reward.
        
        Formula: R_QED = QED(molecule) ∈ [0, 1]
        
        Args:
            mol: RDKit molecule
        
        Returns:
            R_QED: float in [0, 1]
        """
        if mol is None:
            return 0.0
        
        try:
            return float(QED.qed(mol))
        except:
            return 0.0
    
    def compute_sa_reward(self, mol: Optional[Chem.Mol]) -> float:
        """
        Compute R_SA reward.
        
        Formula: R_SA = (10 - SA_Score) / 9 ∈ [0, 1]
        
        Args:
            mol: RDKit molecule
        
        Returns:
            R_SA: float in [0, 1]
        """
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
    

    def compute_reward_A1(self, mol_before: Chem.Mol, mol_after: Chem.Mol) -> float:
        """
        Compute reward after Phase A1 (scaffold extension).
        
        Formula: r_A1 = ΔR_prop = R_prop(after) - R_prop(before)
        where R_prop = mean over i of exp(-(Y_i - Y_i*)^2 / (2σ_i^2))
        
        Phase A1 uses encoder_Sc (scaffold encoder from HES model)
        
        Args:
            mol_before: molecule before Phase A1
            mol_after: molecule after Phase A1
        
        Returns:
            reward: float value (r_A1)
        """
        if mol_before is None or mol_after is None:
            return 0.0
        
        # Compute properties
        props_before = self.compute_properties(mol_before)
        props_after = self.compute_properties(mol_after)
        
        # Compute R_prop using Gaussian reward function
        r_prop_before = self.compute_property_reward(props_before)
        r_prop_after = self.compute_property_reward(props_after)
        
        # r_A1 = ΔR_prop
        reward_A1 = r_prop_after - r_prop_before
        
        # Check validity
        try:
            Chem.SanitizeMol(mol_after)
        except:
            reward_A1 -= 0.5  # Penalty for invalid molecule
        
        return float(reward_A1)
    
    def compute_reward_A2(self, mol_before: Chem.Mol, mol_after: Chem.Mol,
                         w_qed: float = 0.3, w_sa: float = 0.3) -> float:
        """
        Compute reward after Phase A2 (atom/bond labeling).
        
        Formula: r_A2 = Δ(R_prop + w₁R_QED + w₂R_SA)
        where:
        - R_prop = mean Gaussian reward for all properties
        - R_QED = QED(molecule) ∈ [0, 1]
        - R_SA = (10 - SA_Score) / 9 ∈ [0, 1]
        
        Phase A2 uses encoder_G (atomic graph encoder from HES model)
        
        Args:
            mol_before: molecule before Phase A2
            mol_after: molecule after Phase A2
            w_qed: weight for QED component
            w_sa: weight for SA component
        
        Returns:
            reward: float value (r_A2)
        """
        if mol_before is None or mol_after is None:
            return 0.0
        
        # Compute properties
        props_before = self.compute_properties(mol_before)
        props_after = self.compute_properties(mol_after)
        
        # Compute R_prop
        r_prop_before = self.compute_property_reward(props_before)
        r_prop_after = self.compute_property_reward(props_after)
        
        # Compute R_QED
        r_qed_before = self.compute_qed_reward(mol_before)
        r_qed_after = self.compute_qed_reward(mol_after)
        
        # Compute R_SA
        r_sa_before = self.compute_sa_reward(mol_before)
        r_sa_after = self.compute_sa_reward(mol_after)
        
        # Total reward before and after
        total_before = r_prop_before + w_qed * r_qed_before + w_sa * r_sa_before
        total_after = r_prop_after + w_qed * r_qed_after + w_sa * r_sa_after
        
        # r_A2 = Δ(R_prop + w₁R_QED + w₂R_SA)
        reward_A2 = total_after - total_before
        
        # Check validity
        try:
            Chem.SanitizeMol(mol_after)
        except:
            reward_A2 -= 0.5  # Penalty for invalid molecule
        
        return float(reward_A2)
    
    def compute_terminal_reward(self, mol_initial: Chem.Mol, mol_final: Chem.Mol,
                               w_qed: float = 0.3, w_sa: float = 0.3) -> float:
        """
        Compute terminal reward at episode end.
        
        Formula: r_terminal = Δ(R_prop + w₁R_QED + w₂R_SA) + validity_bonus
        
        Calculates overall improvement from episode start to end.
        
        Args:
            mol_initial: initial molecule (starting scaffold)
            mol_final: final molecule (after full generation)
            w_qed: weight for QED component
            w_sa: weight for SA component
        
        Returns:
            reward: float value (r_terminal)
        """
        if mol_final is None:
            return TERMINAL_INVALID_MOLECULE_PENALTY
        
        # Compute properties
        props_initial = self.compute_properties(mol_initial)
        props_final = self.compute_properties(mol_final)
        
        # Compute R_prop
        r_prop_initial = self.compute_property_reward(props_initial, is_terminal=True)
        r_prop_final = self.compute_property_reward(props_final, is_terminal=True)
        
        # Compute R_QED
        r_qed_initial = self.compute_qed_reward(mol_initial)
        r_qed_final = self.compute_qed_reward(mol_final)
        
        # Compute R_SA
        r_sa_initial = self.compute_sa_reward(mol_initial)
        r_sa_final = self.compute_sa_reward(mol_final)
        
        # Total reward before and after
        total_initial = r_prop_initial + w_qed * r_qed_initial + w_sa * r_sa_initial
        total_final = r_prop_final + w_qed * r_qed_final + w_sa * r_sa_final
        
        # Terminal reward = Δ(R_prop + w₁R_QED + w₂R_SA)
        reward_terminal = total_final - total_initial
        
        # Bonus/penalty for valid/invalid molecule
        try:
            Chem.SanitizeMol(mol_final)
            reward_terminal += TERMINAL_VALID_MOLECULE_BONUS
        except:
            reward_terminal += TERMINAL_INVALID_MOLECULE_PENALTY
        
        # Penalty if molecule is too large
        if mol_final.GetNumAtoms() > MAX_ATOMS_PER_MOLECULE:
            reward_terminal -= 0.5
        
        return float(reward_terminal)



class ReplayBuffer:
    """
    Replay buffer for storing and sampling experience trajectories.
    """
    
    def __init__(self, capacity: int = REPLAY_BUFFER_SIZE):
        """
        Initialize replay buffer.
        
        Args:
            capacity: maximum number of experiences to store
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        experience = (state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple:
        """
        Sample a batch of experiences.
        
        Args:
            batch_size: number of experiences to sample
        
        Returns:
            batch: (states, actions, rewards, next_states, dones)
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])
        
        return (
            np.array(states),
            actions,  # Keep as list of dicts for now
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )
    
    def __len__(self):
        return len(self.buffer)
