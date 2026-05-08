"""
MoleculeEnv: Gym environment for hierarchical molecular generation
Implements the action flow from Stage 2 specification:
  a1 (attachment) → a2 (shape) → a3 (orientation) → atom+bond labeling → reward
"""
import gym
from gym import spaces
import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import AllChem, QED, Descriptors
from pathlib import Path
import sys
from typing import Tuple, Dict, Any

# Add stage1_hes to path for model access
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "stage1_hes"))
sys.path.insert(0, str(PROJECT_ROOT / "processing"))

from stage2_rl.training.config import *

# ===== Molecular Constants =====
VALENCE_DICT = {6: 4, 7: 3, 8: 2, 16: 2, 15: 3, 9: 1, 17: 1, 35: 1, 53: 1}
def load_starting_scaffolds(filepath="data/smiles/starting_scaffolds.smi"):
    scaffolds = []
    try:
        # Đường dẫn tuyệt đối từ PROJECT_ROOT
        import os
        from pathlib import Path
        root = Path(__file__).parent.parent.parent
        full_path = root / filepath
        
        with open(full_path, "r") as f:
            for line in f:
                s = line.strip().split()[0]
                if s and s != "smiles":
                    scaffolds.append(s)
    except Exception as e:
        print(f"Warning: Could not load {filepath}. Using fallback.")
        scaffolds = ["c1ccccc1", "C1CCCC1", "C1=CN=CN1", "c1ccncc1", "c1cnccc1"]
    return scaffolds
# Simple ring scaffolds for training initialization
STARTING_SCAFFOLDS = load_starting_scaffolds()


class MoleculeEnv(gym.Env):
    """
    Hierarchical molecular generation environment.
    
    Action space (hierarchical):
    - a1: attachment point on scaffold (0 to N_sc-1)
    - a2: shape/motif to add (0 to NUM_SHAPES-1)
    - a3: orientation/rotation (0 to max_angles-1)
    - atom_types: for each vacant position (0 to NUM_ATOM_TYPES)
    - bond_types: for each new/modified bond (0 to NUM_BOND_TYPES)
    
    State:
    - HES encoding: concatenated encoder_g(x_g) + encoder_sc(x_sc) = 256-dim vector
    - Episode step count
    
    Reward calculation:
    - r_A1: after Phase A1 (scaffold + motif addition)
    - r_A2: after Phase A2 (atom/bond labeling)
    - Terminal reward: property changes from start to end
    """
    
    def __init__(self, hes_model, motif_vocab, shape_vocab, shape_templates, shape_to_motifs, property_scaler, reward_computer=None, target_protein: str = "parp1", warmup_episodes: int = WARMUP_EPISODES):
        """
        Initialize environment.
        
        Args:
            hes_model: Loaded Stage 1 HES model
            motif_vocab: dict{smiles → ecfp_array}
            shape_vocab: dict{shape_hash → {motifs, count, avg_ecfp}}
            shape_templates: dict{shape_hash → list of config dicts}
            warmup_episodes: Number of episodes to use template-guided warmup
            property_scaler: Fitted StandardScaler for property normalization
            reward_computer: RewardComputer instance (will be created if None)
            target_protein: Target protein for single-target reward optimization
        """
        super().__init__()
        
        self.hes_model = hes_model
        self.motif_vocab = motif_vocab
        self.shape_vocab = shape_vocab
        self.shape_templates = shape_templates
        self.shape_to_motifs = shape_to_motifs
        self.property_scaler = property_scaler
        self.warmup_episodes = warmup_episodes
        
        self.current_episode_num = 0
        self.is_training = False
        
        # Initialize reward computer if not provided
        if reward_computer is None:
            from stage2_rl.training.rewards import RewardComputer
            self.reward_computer = RewardComputer(
                property_scaler_path=STAGE1_SCALER,
                hes_model=self.hes_model,
                target_protein=target_protein,
                hes_output_is_normalized=HES_OUTPUT_IS_NORMALIZED,
                target_properties_are_normalized=TARGET_PROPERTIES_ARE_NORMALIZED,
            )
        else:
            self.reward_computer = reward_computer
        
        # Determine number of shapes and motifs
        self.num_shapes = len(shape_vocab)
        self.num_motifs = len(motif_vocab)
        
        # ===== STATE ENCODER MLP =====
        # h_s = MLP(Encoder_G(G_t) || Encoder_Sc(Sc_t))
        # Input: 512-dim (256 from encoder_g + 256 from encoder_sc)
        # Output: HES_ENCODING_DIM (512)
        self.state_encoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, HES_ENCODING_DIM),
        )
        device = next(hes_model.parameters()).device
        self.state_encoder = self.state_encoder.to(device)
        
        # State space: HES encoding (256-dim)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(HES_ENCODING_DIM,),
            dtype=np.float32
        )
        
        # Action space (hierarchical, but we'll treat as discrete for SAC)
        # Total action space size:
        # - a1: up to MAX_ATOMS_PER_MOLECULE (attachment points on scaffold)
        # - a2: NUM_SHAPES choices
        # - a3: assume 4 possible orientations (0, 90, 180, 270 degrees)
        # - atom_labeling: up to MAX_ATOMS_PER_MOLECULE atoms × NUM_ATOM_TYPES
        # - bond_labeling: up to MAX_ATOMS_PER_MOLECULE bonds × NUM_BOND_TYPES
        
        # STOP is represented as the last value of a1 (index MAX_ATOMS_PER_MOLECULE)
        self.action_space = spaces.Dict({
            'a1': spaces.Discrete(MAX_ATOMS_PER_MOLECULE + 1),       # attachment point + STOP
            'a2': spaces.Discrete(self.num_shapes),                   # shape
            'a3': spaces.Discrete(4),                                 # orientation (4 angles)
            'a2_motif_idx': spaces.Discrete(MAX_MOTIFS_PER_SHAPE),
            'a2_motif_attach': spaces.Discrete(MAX_ATOMS_PER_MOTIF),
        })
        
        # Current molecular structures
        self.molecule_smiles = None          # Current SMILES
        self.current_molecule = None         # RDKit molecule
        self.scaffold_molecule = None        # Scaffold part
        self.initial_molecule = None         # Initial molecule (for reward)
        self.generated_motifs = []           # List of added motifs
        self.episode_step = 0
        self.junction_atom_idx = None        # Index of junction atom for current motif (not predicted in A2)
        
    def set_episode(self, current_episode_num: int, training: bool = True):
        """Set current episode number and mode, primarily used for warmup strategy."""
        self.current_episode_num = current_episode_num
        self.is_training = training
        
    def reset(self, initial_smiles=None):
        """
        Reset environment.
        
        Args:
            initial_smiles: Optional initial molecule SMILES.
                           - If provided: use it directly (evaluation mode)
                           - If None: sample from simple STARTING_SCAFFOLDS (training mode)
        
        Returns:
            obs: Initial state (HES encoding)
        """
        if initial_smiles is None:
            import random
            initial_smiles = random.choice(STARTING_SCAFFOLDS)
        
        self.current_molecule = Chem.MolFromSmiles(initial_smiles) if initial_smiles else None
        
        # Fallback to Benzene if RDKit fails to parse the SMILES or initial_smiles is None
        if self.current_molecule is None:
            self.current_molecule = Chem.MolFromSmiles('c1ccccc1')
            self.molecule_smiles = 'c1ccccc1'
        else:
            self.molecule_smiles = initial_smiles
            
        self.initial_molecule = self.current_molecule  # Store initial for reward
        self.scaffold_molecule = self.current_molecule  # Initially, molecule = scaffold
        self.generated_motifs = []
        self.episode_step = 0
        self.junction_atom_idx = None
        
        # Compute and return HES encoding for initial state
        obs = self._get_hes_encoding()
        
        return obs
    def step(self, action: Dict[str, int]) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one step of environment.
        """
        self.episode_step += 1
        
        # Extract actions
        a1 = action.get('a1', 0)  # attachment point or STOP
        a2 = action.get('a2', 0)  # shape
        a3 = action.get('a3', 0)  # orientation
        stop_index = MAX_ATOMS_PER_MOLECULE
        
        info = {'step': self.episode_step, 'motif_count': len(self.generated_motifs)}
        reward = 0.0
        done = False
        
        # Check STOP action (as a1 special value) or max motifs
        if a1 == stop_index or len(self.generated_motifs) >= MAX_MOTIFS_PER_EPISODE:
            # Episode termination: calculate terminal reward
            done = True
            final_molecule = self.current_molecule
            reward = self._compute_terminal_reward(self.initial_molecule, final_molecule)
            info['terminated_by'] = 'STOP' if a1 == stop_index else 'max_motifs'
            return self._get_hes_encoding(), reward, done, info
        
        # Capture scaffold size BEFORE adding the new shape
        n_scaffold_atoms = self.scaffold_molecule.GetNumAtoms()
        
        # Phase A1: Scaffold extension
        result_A1 = self._phase_A1(a1, a2, a3)
        if result_A1 is None or len(result_A1) < 3:
            reward = -1.0
            done = True  # Terminate episode on invalid action
            info['error'] = 'invalid_phase_A1'
            return self._get_hes_encoding(), reward, done, info
        
        reward_A1, new_scaffold, junction_atom_idx = result_A1
        self.scaffold_molecule = new_scaffold
        self.junction_atom_idx = junction_atom_idx  # Store for Phase A2
        reward += reward_A1
        
        # --- WARMUP STRATEGY ---
        if self.is_training and self.current_episode_num < self.warmup_episodes:
            self._extract_warmup_actions(a2, n_scaffold_atoms, action)
        
        # Phase A2: Atom and bond labeling
        a2_motif_idx = action.get('a2_motif_idx')
        a2_motif_attach = action.get('a2_motif_attach')
        reward_A2, labeled_molecule = self._phase_A2(self.scaffold_molecule, a1, a2, a2_motif_idx, a2_motif_attach)
        
        if labeled_molecule is None:
            # Invalid labeling - assign penalty
            reward += -1.0
            done = True  # Terminate episode on invalid labeling
            info['error'] = 'invalid_phase_A2'
            return self._get_hes_encoding(), reward, done, info
        
        # CRITICAL FIX: Update all references to allow sequential growth
        self.current_molecule = labeled_molecule
        self.molecule_smiles = Chem.MolToSmiles(labeled_molecule)
        self.scaffold_molecule = labeled_molecule  # <-- Dòng cực kỳ quan trọng
        reward += reward_A2
        
        # Track successful motif addition
        self.generated_motifs.append(self.molecule_smiles)
        
        # Check if molecule is valid
        try:
            Chem.SanitizeMol(self.current_molecule)
        except:
            reward = min(reward, -0.5)  # Penalize invalid molecules
            info['valid_molecule'] = False
        else:
            info['valid_molecule'] = True
        
        # Check termination by size
        if self.current_molecule.GetNumAtoms() > MAX_ATOMS_PER_MOLECULE:
            done = True
            info['terminated_by'] = 'max_atoms'
        
        # Get next state
        obs = self._get_hes_encoding()
        
        return obs, reward, done, info   
    def _extract_warmup_actions(self, a2: int, n_scaffold_atoms: int, action: Dict[str, Any]):
        """
        Extract ground-truth motif idx and attach node for the chosen shape to provide perfect labels.
        Forces the 'action' dict to reflect a valid motif matching the joint atom.
        """
        if not self.is_training or self.current_episode_num >= self.warmup_episodes:
            return
            
        a1 = action.get('a1', 0)
        if a1 >= self.scaffold_molecule.GetNumAtoms():
            return
            
        joint_atom = self.scaffold_molecule.GetAtomWithIdx(a1)
        joint_element = joint_atom.GetSymbol()
        
        if a2 not in self.shape_to_motifs:
            return
            
        candidate_motifs = self.shape_to_motifs[a2]
        
        # Find first motif that matches joint_element
        valid_idx = -1
        valid_attach = -1
        for idx, motif in enumerate(candidate_motifs):
            for node_idx, symbol in enumerate(motif['atoms']):
                if symbol == joint_element:
                    valid_idx = idx
                    valid_attach = node_idx
                    break
            if valid_idx != -1:
                break
                
        if valid_idx != -1:
            action['a2_motif_idx'] = valid_idx
            action['a2_motif_attach'] = valid_attach

    def _phase_A1(self, a1: int, a2: int, a3: int):
        """
        Phase A1: Dummy pass-through for fragment-based approach.
        Graph merging is fully handled in Phase A2 via overlap.
        """
        if a1 >= self.scaffold_molecule.GetNumAtoms():
            return None
        if a2 >= self.num_shapes:
            return None
            
        return 0.0, self.scaffold_molecule, a1

    def _phase_A2(self, scaffold_mol, a1: int, a2: int, a2_motif_idx: int, a2_motif_attach: int) -> Tuple[float, Any]:
        """
        Phase A2: Pure Fragment-Based approach.
        1. Identify joint atom a1.
        2. Filter valid motifs from shape a2.
        3. Match motif node a2_motif_attach to a1.
        4. Validate valency.
        5. Merge graph.
        """
        try:
            # 1. Identify Joint Atom
            if a1 >= scaffold_mol.GetNumAtoms():
                return -1.0, None
            joint_atom = scaffold_mol.GetAtomWithIdx(a1)
            joint_element = joint_atom.GetSymbol()
            
            # 2. Filter Candidates for shape_idx = a2
            if a2 not in self.shape_to_motifs:
                return -1.0, None
                
            candidate_motifs = self.shape_to_motifs[a2]
            valid_motifs = [m for m in candidate_motifs if joint_element in m['atoms']]
            
            # BUG 1 FIX: Safe Indexing + Empty List Return
            if not valid_motifs:
                return -1.0, None
            chosen_motif_idx = a2_motif_idx % len(valid_motifs)
            motif = valid_motifs[chosen_motif_idx]
            
            valid_attach_nodes = [i for i, symbol in enumerate(motif['atoms']) if symbol == joint_element]
            if not valid_attach_nodes:
                return -1.0, None
            chosen_attach_node = valid_attach_nodes[a2_motif_attach % len(valid_attach_nodes)]
            
            # Convert motif string to graph
            motif_mol = Chem.MolFromSmiles(motif['smiles'])
            if motif_mol is None:
                return -1.0, None
            
            # BUG 2 FIX: RDKit Explicit Valency Calculation
            pt = Chem.GetPeriodicTable()
            max_valence = pt.GetDefaultValence(joint_atom.GetAtomicNum())
            
            current_bonds = joint_atom.GetExplicitValence()
            motif_attach_atom = motif_mol.GetAtomWithIdx(chosen_attach_node)
            internal_bonds = motif_attach_atom.GetExplicitValence()
            
            if (current_bonds + internal_bonds) > max_valence:
                return -1.0, None
                
            # BUG 3 FIX: Safe RDKit Graph Merge
            combined = Chem.CombineMols(scaffold_mol, motif_mol)
            rw_mol = Chem.RWMol(combined)
            
            offset = scaffold_mol.GetNumAtoms()
            motif_match_idx = offset + chosen_attach_node
            
            # Add bonds from a1 to the neighbors of the motif's attachment node
            for bond in motif_mol.GetBonds():
                if bond.GetBeginAtomIdx() == chosen_attach_node:
                    neighbor_idx = offset + bond.GetEndAtomIdx()
                    rw_mol.AddBond(a1, neighbor_idx, bond.GetBondType())
                elif bond.GetEndAtomIdx() == chosen_attach_node:
                    neighbor_idx = offset + bond.GetBeginAtomIdx()
                    rw_mol.AddBond(a1, neighbor_idx, bond.GetBondType())
                    
            # Safely remove the redundant motif attachment node
            rw_mol.RemoveAtom(motif_match_idx)
            
            new_mol = rw_mol.GetMol()
            try:
                Chem.SanitizeMol(new_mol)
            except:
                return -1.0, None
                
            reward = self.reward_computer.compute_reward_A2(scaffold_mol, new_mol)
            reward_A1 = self.reward_computer.compute_reward_A1(scaffold_mol, new_mol)
            return reward + reward_A1, new_mol
            
        except Exception as e:
            return -1.0, None

    def _get_allowed_atoms_by_valence(self, atom: Chem.Atom) -> set:
        """
        Get allowed atom types based on valency constraint.
        
        Spec: Valency masking mechanism to ensure chemical validity
        
        Args:
            atom: RDKit Atom object
        
        Returns:
            Set of allowed atomic numbers (or empty if no valid type)
        """
        allowed = set()
        remaining_valence = atom.GetImplicitValence()
        
        if remaining_valence == 0:
            return allowed
        
        # Check which supported atoms can satisfy remaining valence
        for atom_symbol, atom_num in ATOM_TYPES.items():
            max_valence = VALENCE_DICT.get(atom_num, 0)
            # Atom can be assigned if its valence >= remaining valence needed
            if max_valence >= remaining_valence:
                allowed.add(atom_num)
        
        return allowed
    
    def _get_allowed_bonds(self, mol: Chem.RWMol, bond: Chem.Bond) -> list:
        """
        Get allowed bond types based on valency constraint.
        
        Spec: Valency masking for bond types (Single, Double, Triple)
        
        Args:
            mol: RWMol molecule
            bond: Bond object
        
        Returns:
            List of allowed bond types
        """
        allowed = []
        
        try:
            begin_atom = mol.GetAtomWithIdx(bond.GetBeginAtomIdx())
            end_atom = mol.GetAtomWithIdx(bond.GetEndAtomIdx())
            
            old_bond_val = bond.GetBondTypeAsDouble()
            begin_valence = begin_atom.GetImplicitValence() + old_bond_val
            end_valence = end_atom.GetImplicitValence() + old_bond_val
            
            # Single bond: uses 1 valence from each atom
            if begin_valence >= 1 and end_valence >= 1:
                allowed.append(Chem.BondType.SINGLE)
            
            # Double bond: uses 2 valence from each atom
            if begin_valence >= 2 and end_valence >= 2:
                allowed.append(Chem.BondType.DOUBLE)
            
            # Triple bond: uses 3 valence from each atom
            if begin_valence >= 3 and end_valence >= 3:
                allowed.append(Chem.BondType.TRIPLE)
            
        except:
            pass
        
        return allowed
    
    def _get_hes_encoding(self) -> np.ndarray:
        """
        Compute HES encoding for current molecule state.
        
        Returns HES encoding from Stage 1 model:
        h_s = MLP(Encoder_G(G_t) || Encoder_Sc(Sc_t))
        
        where Sc_t is extracted via MAGNet decomposition (ring + junction atoms only)
        """
        if self.current_molecule is None:
            return np.zeros(HES_ENCODING_DIM, dtype=np.float32)
        
        try:
            from processing.utils.graph_builder import build_graph_simple, mol_to_pyg_data
            from processing.utils.scaffold_extractor import get_motif_decomposition, get_scaffold_from_decomposition
            
            smiles = Chem.MolToSmiles(self.current_molecule)
            if smiles is None:
                return np.zeros(HES_ENCODING_DIM, dtype=np.float32)
            
            # Build atomic graph G
            data_g = build_graph_simple(smiles)
            if data_g is None:
                return np.zeros(HES_ENCODING_DIM, dtype=np.float32)
            
            # Extract scaffold using MAGNet decomposition (rings + junctions only)
            try:
                mol_decomp = get_motif_decomposition(smiles)
                scaffold_mol, _, _ = get_scaffold_from_decomposition(mol_decomp)
            except:
                scaffold_mol = None
            
            # Build scaffold graph from extracted scaffold
            if scaffold_mol is not None:
                data_sc = mol_to_pyg_data(scaffold_mol, add_smiles=False)
            else:
                from torch_geometric.data import Data
                # Create a null scaffold molecule node as in Stage 1
                null_sc = Chem.MolFromSmiles('C') # Dummy atom
                x_sc = torch.zeros((1, 16), dtype=torch.float)
                x_sc[0, 0] = 0  # atomic_number = 0
                x_sc[0, -1] = 1 # null_flag = 1
                edge_index_sc = torch.empty((2, 0), dtype=torch.long)
                data_sc = Data(x=x_sc, edge_index=edge_index_sc)
            
            device = next(self.hes_model.parameters()).device
            data_g = data_g.to(device)
            data_g.x = data_g.x.float()
            
            if data_sc is not None:
                data_sc = data_sc.to(device)
                data_sc.x = data_sc.x.float()
            
            # Match Stage 1 HES input dimensions: G=15, Sc=16
            if data_g.x.dim() == 1:
                data_g.x = data_g.x.unsqueeze(-1)
            if data_g.x.size(1) < 15:
                pad = torch.zeros(data_g.x.size(0), 15 - data_g.x.size(1), device=device)
                data_g.x = torch.cat([data_g.x, pad], dim=1)
            elif data_g.x.size(1) > 15:
                data_g.x = data_g.x[:, :15]
            
            if data_sc is not None:
                if data_sc.x.dim() == 1:
                    data_sc.x = data_sc.x.unsqueeze(-1)
                if data_sc.x.size(1) < 16:
                    pad = torch.zeros(data_sc.x.size(0), 16 - data_sc.x.size(1), device=device)
                    data_sc.x = torch.cat([data_sc.x, pad], dim=1)
                elif data_sc.x.size(1) > 16:
                    data_sc.x = data_sc.x[:, :16]
            
            # Batch tensors
            batch_g = torch.zeros(data_g.num_nodes, dtype=torch.long, device=device)
            if data_sc is not None:
                batch_sc = torch.zeros(data_sc.num_nodes, dtype=torch.long, device=device)
            else:
                batch_sc = torch.zeros(0, dtype=torch.long, device=device)
            
            # Forward through HES encoders
            with torch.no_grad():
                outputs = self.hes_model(
                    x_g=data_g.x,
                    edge_index_g=data_g.edge_index,
                    edge_attr_g=data_g.edge_attr if hasattr(data_g, 'edge_attr') else None,
                    x_sc=data_sc.x if data_sc is not None else torch.zeros((0, 16), device=device),
                    edge_index_sc=data_sc.edge_index if data_sc is not None else torch.zeros((2, 0), dtype=torch.long, device=device),
                    edge_attr_sc=data_sc.edge_attr if data_sc is not None and hasattr(data_sc, 'edge_attr') else None,
                    batch_g=batch_g,
                    batch_sc=batch_sc,
                )
            
                # Extract encodings from HES model
                emb_g = outputs.get("emb_g_mole", torch.zeros(1, HES_FEATURE_DIM, device=device))
                emb_sc = outputs.get("emb_sc_mole", torch.zeros(1, HES_FEATURE_DIM, device=device))
                
                # Ensure shapes are correct
                if emb_g.dim() == 1:
                    emb_g = emb_g.unsqueeze(0)
                if emb_sc.dim() == 1:
                    emb_sc = emb_sc.unsqueeze(0)
                
                # Concatenate: encoder_g || encoder_sc = (256,)
                concat_embedding = torch.cat([emb_g[0], emb_sc[0]], dim=0)  # (256,)
                
                # Apply state encoder MLP: h_s = MLP(concat_embedding)
                h_s = self.state_encoder(concat_embedding)  # (HES_ENCODING_DIM,)
                
            return h_s.cpu().numpy().astype(np.float32)
            
        except Exception as e:
            print(f"[WARNING] HES encoding computation failed: {e}")
            return np.zeros(HES_ENCODING_DIM, dtype=np.float32)
    
    def _compute_terminal_reward(self, mol_initial: Chem.Mol, mol_final: Chem.Mol) -> float:
        """
        Compute terminal reward using RewardComputer.
        
        Formula: r_terminal = Δ(R_prop + w₁R_QED + w₂R_SA) + validity_bonus
        """
        return self.reward_computer.compute_terminal_reward(mol_initial, mol_final)

    def get_motif_idx_mask(self, a1: int, a2: int) -> np.ndarray:
        """Get valid motifs mask for chosen attachment point and shape."""
        # Start with all zeros
        from stage2_rl.training.config import MAX_MOTIFS_PER_SHAPE, MAX_ATOMS_PER_MOTIF
        mask = np.zeros(MAX_MOTIFS_PER_SHAPE, dtype=np.float32)
        
        if a1 >= self.scaffold_molecule.GetNumAtoms() or a2 not in self.shape_to_motifs:
            return mask
            
        joint_atom = self.scaffold_molecule.GetAtomWithIdx(a1)
        joint_element = joint_atom.GetSymbol()
        
        candidate_motifs = self.shape_to_motifs[a2]
        valid_motifs = [m for m in candidate_motifs if joint_element in m['atoms']]
        
        if valid_motifs:
            # All motifs within the valid_motifs array bounds are pickable
            mask[:len(valid_motifs)] = 1.0
            
        return mask

    def get_motif_attach_mask(self, a1: int, a2: int, a2_motif_idx: int) -> np.ndarray:
        """Get valid attachment nodes on the chosen motif matching the scaffold element."""
        from stage2_rl.training.config import MAX_ATOMS_PER_MOTIF
        mask = np.zeros(MAX_ATOMS_PER_MOTIF, dtype=np.float32)
        
        if a1 >= self.scaffold_molecule.GetNumAtoms() or a2 not in self.shape_to_motifs:
            return mask
            
        joint_atom = self.scaffold_molecule.GetAtomWithIdx(a1)
        joint_element = joint_atom.GetSymbol()
        
        candidate_motifs = self.shape_to_motifs[a2]
        valid_motifs = [m for m in candidate_motifs if joint_element in m['atoms']]
        
        if not valid_motifs:
            return mask
            
        chosen_motif_idx = a2_motif_idx % len(valid_motifs)
        motif = valid_motifs[chosen_motif_idx]
        
        valid_attach_nodes = [i for i, symbol in enumerate(motif['atoms']) if symbol == joint_element]
        
        for idx in valid_attach_nodes:
            if idx < MAX_ATOMS_PER_MOTIF:
                mask[idx] = 1.0
                
        return mask

    def get_action_mask(self) -> dict:
        """
        Get valid action mask for current state.
        Returns a dict with 'a1', 'a2', 'a3' keys, where 1 means valid and 0 means invalid.
        """
        device = next(self.hes_model.parameters()).device
        
        # a1 mask: valid attachment points + stop action
        a1_mask_arr = np.zeros(MAX_ATOMS_PER_MOLECULE + 1, dtype=np.float32)
        valid_points = self._get_valid_attachment_points()
        for pt in valid_points:
            if pt < MAX_ATOMS_PER_MOLECULE:
                a1_mask_arr[pt] = 1.0
        # Stop action is index MAX_ATOMS_PER_MOLECULE
        if len(self.generated_motifs) >= 1 or len(valid_points) == 0:
            a1_mask_arr[-1] = 1.0
        else:
            a1_mask_arr[-1] = 0.0
        
        a1_mask = torch.tensor(a1_mask_arr, device=device).unsqueeze(0)  # Add batch dim
        
        # a2 mask: all shapes valid
        a2_mask = torch.ones((1, NUM_SHAPES), device=device)
        
        # a3 mask: all orientations valid
        a3_mask = torch.ones((1, 4), device=device)
        
        return {
            'a1': a1_mask,
            'a2': a2_mask,
            'a3': a3_mask
        }

    def _get_valid_attachment_points(self) -> list:
        """Get valid attachment points for the current scaffold."""
        # Simple heuristic: any atom with open implicit valence
        if self.current_molecule is None:
            return []
            
        valid_pts = []
        for atom in self.current_molecule.GetAtoms():
            if atom.GetImplicitValence() > 0:
                valid_pts.append(atom.GetIdx())
        return valid_pts

    def render(self, mode='human'):
        """Render environment (print current SMILES)."""
        if mode == 'human':
            print(f"Step {self.episode_step}: {self.molecule_smiles}")
