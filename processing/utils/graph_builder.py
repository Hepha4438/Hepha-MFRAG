"""Build molecular graphs G(V,E,X) independently without MAGNet dependency"""
import torch
import pickle
from torch_geometric.data import Data
from rdkit import Chem
import numpy as np
from pathlib import Path

# Import standalone utilities copied from MAGNet
from .magnet_utils import (
    SimpleAtomFeaturizer,
    simple_mol_decompose,
    get_atom_charges,
    ATOM_LIST,
    mol_to_graph,
    smiles_to_mol
)

def build_graph_simple(smiles):
    """
    Build simple G(V,E,X) graph from SMILES
    V = atoms, E = bonds, X = atom types (1-dim)
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # V: Vertices (atoms)
        num_atoms = mol.GetNumAtoms()
        atom_types = torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.long)
        
        # E: Edges (bonds) 
        edge_list = []
        edge_attrs = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_list.append([i, j])
            edge_list.append([j, i])
            bond_type = int(bond.GetBondType())
            edge_attrs.append(bond_type)
            edge_attrs.append(bond_type)
        
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros(0, dtype=torch.long)
        
        # X: Node features (simple: atomic number)
        data = Data(
            x=atom_types.unsqueeze(-1),  # (N, 1)
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=num_atoms,
            smiles=smiles,
        )
        return data
    except Exception as e:
        print(f"Error building simple graph: {e}")
        return None

def build_graph_enhanced(smiles):
    """
    Build enhanced G(V,E,X) graph with richer node features
    Uses SimpleAtomFeaturizer for 320-dim node features
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Initialize featurizer
        featurizer = SimpleAtomFeaturizer(output_dim=320)
        
        # Generate node features (X)
        with torch.no_grad():
            node_features = featurizer(mol, device='cpu')  # (N, 320)
        
        num_atoms = mol.GetNumAtoms()
        
        # Build edges (E)
        edge_list = []
        edge_attrs = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_list.append([i, j])
            edge_list.append([j, i])
            bond_type = int(bond.GetBondType())
            edge_attrs.append(bond_type)
            edge_attrs.append(bond_type)
        
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros(0, dtype=torch.long)
        
        # Create PyG Data object
        data = Data(
            x=node_features,  # (N, 320)
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=num_atoms,
            smiles=smiles,
        )
        
        return data
        
    except Exception as e:
        print(f"[WARNING] Error building enhanced graph: {e}")
        # Fallback to simple graph
        return build_graph_simple(smiles)

def build_graph_magnet_style(smiles, hash_to_id=None, atom_featurizer=None, dim_config=None):
    """
    Build G(V,E,X) in MAGNet style (with enhanced features)
    Note: Full MAGNet decomposition requires torch_sparse which is optional
    Falls back to enhanced graph with SimpleAtomFeaturizer
    """
    # Use enhanced featurizer as fallback
    return build_graph_enhanced(smiles)

def save_graph(data, output_path):
    """Save PyG Data object to disk"""
    torch.save(data, output_path)

def load_graph(graph_path):
    """Load PyG Data object from disk"""
    return torch.load(graph_path)

