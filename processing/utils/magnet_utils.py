"""
Standalone utilities copied from MAGNet for independent processing.
This module contains code extracted from MAGNet to make the pipeline self-contained.
"""
import numpy as np
import torch
from rdkit import Chem
from pathlib import Path


# ============================================================================
# CONSTANTS (from MAGNet/models/magnet/src/chemutils/constants.py)
# ============================================================================
ATOM_LIST = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "Si", "B", "Se"]
BOND_LIST = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
]


# ============================================================================
# Standalone Atom Featurizer (adapted from MAGNet)
# ============================================================================
class SimpleAtomFeaturizer(torch.nn.Module):
    """
    Simplified atom featurizer for molecular graphs.
    Produces fixed-dimension node embeddings without requiring training.
    """
    def __init__(self, output_dim: int = 320):
        super().__init__()
        self.output_dim = output_dim
        self.num_atoms = len(ATOM_LIST)
        
        # Simple embeddings
        self.atom_embedding = torch.nn.Embedding(self.num_atoms, 64)
        self.charge_embedding = torch.nn.Embedding(5, 32)  # -2, -1, 0, 1, 2 charges
        self.bond_count_embedding = torch.nn.Embedding(5, 32)  # 0-4 bonds
        self.ring_embedding = torch.nn.Embedding(2, 32)  # in ring or not
        
        # Linear projection to output dimension
        total_dim = 64 + 32 + 32 + 32 + 100  # last 100 for padding
        self.projection = torch.nn.Linear(total_dim, output_dim)
    
    def forward(self, mol, device='cpu'):
        """
        Generate atom features for molecule.
        
        Args:
            mol: RDKit molecule object
            device: torch device
            
        Returns:
            torch.Tensor of shape (num_atoms, output_dim)
        """
        num_atoms = mol.GetNumAtoms()
        atom_indices = []
        charge_indices = []
        bond_count_indices = []
        ring_indices = []
        
        for atom in mol.GetAtoms():
            idx = ATOM_LIST.index(atom.GetSymbol()) if atom.GetSymbol() in ATOM_LIST else 0
            atom_indices.append(idx)
            
            charge = atom.GetFormalCharge()
            charge_idx = min(max(charge + 2, 0), 4)  # clamp to [0,4] for [-2,2]
            charge_indices.append(charge_idx)
            
            bond_count = atom.GetTotalDegree()
            bond_count_idx = min(bond_count, 4)
            bond_count_indices.append(bond_count_idx)
            
            in_ring = 1 if atom.IsInRing() else 0
            ring_indices.append(in_ring)
        
        atom_indices = torch.tensor(atom_indices, dtype=torch.long, device=device)
        charge_indices = torch.tensor(charge_indices, dtype=torch.long, device=device)
        bond_count_indices = torch.tensor(bond_count_indices, dtype=torch.long, device=device)
        ring_indices = torch.tensor(ring_indices, dtype=torch.long, device=device)
        
        # Generate embeddings
        atom_embs = self.atom_embedding(atom_indices)  # (N, 64)
        charge_embs = self.charge_embedding(charge_indices)  # (N, 32)
        bond_embs = self.bond_count_embedding(bond_count_indices)  # (N, 32)
        ring_embs = self.ring_embedding(ring_indices)  # (N, 32)
        
        # Concatenate
        combined = torch.cat([atom_embs, charge_embs, bond_embs, ring_embs], dim=1)  # (N, 160)
        
        # Pad to match total_dim
        padding_size = 100
        padding = torch.zeros(num_atoms, padding_size, device=device)
        combined = torch.cat([combined, padding], dim=1)  # (N, 260)
        
        # Project to output_dim
        features = self.projection(combined)  # (N, 320)
        
        return features


# ============================================================================
# Simple Molecule Decomposer (alternative to full MolDecomposition)
# ============================================================================
def simple_mol_decompose(smiles: str) -> dict:
    """
    Simple molecule decomposition without full MAGNet algorithm.
    Returns basic molecular properties and topology information.
    
    Args:
        smiles: SMILES string
        
    Returns:
        dict with keys: atoms, bonds, ring_info, aromatic_info, num_motifs
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    Chem.SanitizeMol(mol)
    
    result = {
        'atoms': [atom.GetSymbol() for atom in mol.GetAtoms()],
        'num_atoms': mol.GetNumAtoms(),
        'bonds': [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondType())
                  for bond in mol.GetBonds()],
        'num_bonds': mol.GetNumBonds(),
        'ring_info': mol.GetRingInfo().AtomRings(),
        'aromatic_atoms': [i for i, atom in enumerate(mol.GetAtoms()) if atom.GetIsAromatic()],
    }
    
    # Simple motif count = number of rings + 1 (core structure)
    result['num_motifs'] = len(result['ring_info']) + 1
    
    return result


# ============================================================================
# Fingerprint helpers (simplified)
# ============================================================================
def compute_fingerprint(mol, fp_size: int = 2048) -> np.ndarray:
    """
    Compute Morgan fingerprint for a molecule.
    
    Args:
        mol: RDKit molecule object
        fp_size: fingerprint size (default 2048)
        
    Returns:
        numpy array of shape (fp_size,) with dtype float32
    """
    from rdkit.Chem import AllChem
    
    if mol is None:
        return np.zeros(fp_size, dtype=np.float32)
    
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=fp_size)
        fp_array = np.array(fp, dtype=np.float32)
    except Exception:
        fp_array = np.zeros(fp_size, dtype=np.float32)
    
    return fp_array


# ============================================================================
# Graph building utilities
# ============================================================================
def mol_to_graph(mol, node_features: torch.Tensor = None):
    """
    Convert RDKit molecule to graph representation.
    
    Args:
        mol: RDKit molecule object
        node_features: optional (N, D) tensor of node features
        
    Returns:
        tuple: (edge_index, edge_attr, num_nodes)
        where edge_index is (2, num_edges), edge_attr is (num_edges,)
    """
    edges = []
    edge_attrs = []
    
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edges.append([i, j])
        edges.append([j, i])
        
        bond_type = int(bond.GetBondType())
        edge_attrs.append(bond_type)
        edge_attrs.append(bond_type)
    
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.long)
    else:
        num_atoms = mol.GetNumAtoms()
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0,), dtype=torch.long)
    
    return edge_index, edge_attr, mol.GetNumAtoms()


# ============================================================================
# SMILES utilities
# ============================================================================
def smiles_to_mol(smiles: str) -> Chem.Mol:
    """Convert SMILES to RDKit molecule object with sanitization."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def get_atom_charges(mol) -> np.ndarray:
    """Get formal charges of all atoms in a molecule."""
    if mol is None:
        return np.array([], dtype=np.int32)
    charges = np.array([atom.GetFormalCharge() for atom in mol.GetAtoms()], dtype=np.int32)
    return charges
