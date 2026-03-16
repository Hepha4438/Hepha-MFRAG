"""
Scaffold extraction utilities for HES data generation.
Wraps MAGNet decomposition to extract scaffolds and create scaffold graphs.
"""
import numpy as np
import networkx as nx
from rdkit import Chem
from copy import deepcopy


def extract_atoms_in_motif(nodes_dict, motif_id):
    """
    Extract atoms belonging to a specific motif ID.
    
    Args:
        nodes_dict: Dict mapping atom_idx -> list of motif IDs it belongs to
        motif_id: ID of the motif to extract
    
    Returns:
        List of atom indices in this motif
    """
    return [atom_idx for atom_idx, motif_ids in nodes_dict.items() if motif_id in motif_ids]


def extract_ring_and_junction_atoms(mol_decomp):
    """
    Extract atoms that belong to ring systems and junctions (scaffold atoms).
    Excludes leaf atoms.
    
    This forms the scaffold by removing all acyclic leaves while preserving
    the ring systems and their bridging atoms.
    
    Args:
        mol_decomp: MolDecomposition instance
    
    Returns:
        Set of atom indices that form the scaffold
    """
    scaffold_atoms = set()
    
    # Get all motif IDs (excluding -1 which represents leaf atoms)
    all_motif_ids = set()
    for atom_idx, motif_ids in mol_decomp.nodes.items():
        for mid in motif_ids:
            if mid >= 0:  # Skip leaf atoms
                all_motif_ids.add(mid)
    
    # For each motif ID, if it contains any ring atoms, include all its atoms in scaffold
    for motif_id in all_motif_ids:
        atoms_in_motif = extract_atoms_in_motif(mol_decomp.nodes, motif_id)
        
        # Check if this motif contains any ring atoms
        has_ring_atoms = any(
            mol_decomp.mol.GetAtomWithIdx(atom_idx).IsInRing() 
            for atom_idx in atoms_in_motif
        )
        
        if has_ring_atoms:
            scaffold_atoms.update(atoms_in_motif)
    
    return scaffold_atoms


def create_scaffold_mol(mol, scaffold_atom_indices):
    """
    Create an RDKit molecule object containing only scaffold atoms.
    
    Args:
        mol: Original RDKit molecule
        scaffold_atom_indices: Set of atom indices to keep
    
    Returns:
        RDKit molecule object with only scaffold atoms and their bonds,
        or None if scaffold has no atoms
    """
    if not scaffold_atom_indices:
        return None
    
    # Create list of indices to keep, sorted
    keep_indices = sorted(list(scaffold_atom_indices))
    
    # Build new molecule with only these atoms
    editeable_mol = Chem.RWMol()
    atom_mapping = {}  # Maps old indices to new indices
    
    for new_idx, old_idx in enumerate(keep_indices):
        old_atom = mol.GetAtomWithIdx(old_idx)
        editeable_mol.AddAtom(Chem.Atom(old_atom.GetSymbol()))
        new_atom = editeable_mol.GetAtomWithIdx(new_idx)
        new_atom.SetFormalCharge(old_atom.GetFormalCharge())
        atom_mapping[old_idx] = new_idx
    
    # Add bonds between atoms in scaffold
    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        
        if begin_idx in atom_mapping and end_idx in atom_mapping:
            editeable_mol.AddBond(
                atom_mapping[begin_idx],
                atom_mapping[end_idx],
                bond.GetBondType()
            )
    
    scaffold = editeable_mol.GetMol()
    
    # Sanitize the scaffold molecule
    try:
        Chem.SanitizeMol(scaffold)
    except:
        # If sanitization fails, try clearing aromaticity and kekulizing
        for atom in scaffold.GetAtoms():
            atom.SetIsAromatic(False)
        for bond in scaffold.GetBonds():
            bond.SetIsAromatic(False)
        try:
            Chem.Kekulize(scaffold)
            Chem.SanitizeMol(scaffold)
        except:
            return None
    
    return scaffold, atom_mapping


def get_scaffold_from_decomposition(mol_decomp):
    """
    Create scaffold from a decomposed molecule.
    
    The scaffold is the subgraph containing ring systems and junctions,
    excluding acyclic leaf atoms.
    
    Args:
        mol_decomp: MolDecomposition instance
    
    Returns:
        Tuple of (scaffold_mol, atom_mapping, scaffold_motif_ids)
        - scaffold_mol: RDKit Mol object or None
        - atom_mapping: Dict mapping {old_index -> new_index} in scaffold
        - scaffold_motif_ids: List of motif IDs present in scaffold
    """
    # Extract atoms that form the scaffold
    scaffold_atoms = extract_ring_and_junction_atoms(mol_decomp)
    
    if not scaffold_atoms:
        return None, {}, []
    
    # Create scaffold molecule
    result = create_scaffold_mol(mol_decomp.mol, scaffold_atoms)
    if result is None:
        return None, {}, []
    
    scaffold_mol, atom_mapping = result
    
    # Get motif IDs present in scaffold
    scaffold_motif_ids = []
    for atom_idx in scaffold_atoms:
        for motif_id in mol_decomp.nodes[atom_idx]:
            if motif_id >= 0 and motif_id not in scaffold_motif_ids:
                scaffold_motif_ids.append(motif_id)
    
    scaffold_motif_ids.sort()
    
    return scaffold_mol, atom_mapping, scaffold_motif_ids


def get_motif_decomposition(mol_decomp):
    """
    Get motif IDs for all atoms in the original molecule.
    
    Args:
        mol_decomp: MolDecomposition instance
    
    Returns:
        List where index is atom_idx and value is list of motif IDs containing that atom
    """
    num_atoms = mol_decomp.mol.GetNumAtoms()
    motif_decomposition = [[] for _ in range(num_atoms)]
    
    for atom_idx, motif_ids in mol_decomp.nodes.items():
        # Filter out leaf atoms (-1)
        motif_decomposition[atom_idx] = [mid for mid in motif_ids if mid >= 0]
    
    return motif_decomposition


def get_scaffold_topology_features(scaffold_mol):
    """
    Extract topology features (Weisfeiler-Lehman graph hash) from scaffold.
    
    Args:
        scaffold_mol: RDKit molecule object
    
    Returns:
        Tuple of (graph_hash, adjacency_matrix)
    """
    if scaffold_mol is None:
        return None, None
    
    try:
        adjacency = Chem.GetAdjacencyMatrix(scaffold_mol)
        graph = nx.from_numpy_array(np.triu(adjacency), create_using=nx.Graph)
        graph_hash = nx.weisfeiler_lehman_graph_hash(graph)
        return graph_hash, adjacency
    except:
        return None, None
