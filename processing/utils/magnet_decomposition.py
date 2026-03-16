"""
Independent copy of MAGNet decomposition code
Extracted from MAGNet project to eliminate external dependency
All functionality self-contained here
"""
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from itertools import chain, combinations
from typing import Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
import rdkit.DataStructs as DataStructs
from rdkit import RDLogger
from rdkit.Chem.rdchem import Mol

RDLogger.DisableLog("rdApp.*")

# ============================================================================
# Constants (from magnet.src.chemutils.constants)
# ============================================================================
ATOM_LIST = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "Si", "B", "Se"]
BOND_LIST = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
]

# ============================================================================
# Fingerprint computation (from magnet.src.chemutils.rdkit_helpers)
# ============================================================================
def compute_fingerprint(input: Union[str, Chem.rdchem.Mol]) -> np.array:
    """Compute combined topological and circular Morgan fingerprint"""
    if isinstance(input, str):
        mol = Chem.MolFromSmiles(input)
    else:
        mol = deepcopy(input)
    
    # Topological fingerprint
    top_feats = np.packbits(Chem.RDKFingerprint(mol, fpSize=2048)) / 255
    
    # Circular Morgan fingerprint
    circ_feats = AllChem.GetHashedMorganFingerprint(mol, 3, nBits=256)
    array = np.zeros((0,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(circ_feats, array)
    circ_feats = array
    
    # Concatenate
    mol_fingerprint = np.concatenate([top_feats, circ_feats])
    return mol_fingerprint

# ============================================================================
# Helper functions for decomposition (from magnet.src.chemutils.hypergraph)
# ============================================================================

def is_all_cyclic(mol):
    """Check if all atoms and bonds in molecule are in rings"""
    all_bonds = all([b.IsInRing() for b in mol.GetBonds()])
    all_atoms = all([a.IsInRing() for a in mol.GetAtoms()])
    return all_bonds and all_atoms


def is_cyclic_junction(mol):
    """Check if molecule has a cyclic junction"""
    return any([is_atom_cyclic_junction(a) for a in mol.GetAtoms()])


def is_atom_cyclic_junction(atom):
    """Check if atom is both cyclic and a junction"""
    return atom.IsInRing() and is_atom_junction(atom)


def is_atom_junction(atom):
    """Check if atom is a junction (degree 3 or 4, not in ring)"""
    if atom.IsInRing():
        return False
    else:
        return atom.GetDegree() in [4, 3]


def sanitize_with_catch(mol):
    """
    Sanitize molecule with error handling for aromaticity issues
    from fused bonds separation
    """
    atoms_before = deepcopy([a.GetSymbol() for a in mol.GetAtoms()])
    charges_before = deepcopy([a.GetFormalCharge() for a in mol.GetAtoms()])
    adjacency_before = deepcopy(Chem.rdmolops.GetAdjacencyMatrix(mol))
    try:
        Chem.SanitizeMol(mol)
    except:
        for atom in mol.GetAtoms():
            atom.SetIsAromatic(False)
        for bond in mol.GetBonds():
            bond.SetIsAromatic(False)
        Chem.Kekulize(mol)
        Chem.SanitizeMol(mol)
    assert atoms_before == [a.GetSymbol() for a in mol.GetAtoms()]
    assert charges_before == [a.GetFormalCharge() for a in mol.GetAtoms()]
    assert np.all(adjacency_before == Chem.rdmolops.GetAdjacencyMatrix(mol))


def find_all_neighbours(mol: Chem.Mol, atom_idx: list, exclude_idx: list):
    """
    In a given molecule, find all nodes reachable from atom_idx 
    without a path over exclude_idx
    """
    start_set = list(set(atom_idx) - set(exclude_idx))
    previous = 0
    neighbours = deepcopy(start_set)
    while True:
        new_neighbours = []
        for aid in neighbours:
            new_neighbours.extend([n.GetIdx() for n in mol.GetAtomWithIdx(aid).GetNeighbors()])
        neighbours.extend(new_neighbours)
        neighbours = list(set([n for n in neighbours if n not in exclude_idx]))
        if len(neighbours) == previous:
            break
        previous = len(neighbours)
    return neighbours


def intersect(a: list, b: list):
    """Find intersection of two lists"""
    return list(set(a).intersection(b))


def extract_valid_fragment(mol, extract_atom_ids):
    """Extract valid fragment from molecule by atom indices"""
    editeable_mol = Chem.RWMol()
    for i, eai in enumerate(extract_atom_ids):
        editeable_mol.AddAtom(Chem.Atom(mol.GetAtomWithIdx(eai).GetSymbol()))
        atom = editeable_mol.GetAtomWithIdx(i)
        atom.SetFormalCharge(mol.GetAtomWithIdx(eai).GetFormalCharge())
    for bond in mol.GetBonds():
        if bond.GetBeginAtom().GetIdx() in extract_atom_ids:
            if bond.GetEndAtom().GetIdx() in extract_atom_ids:
                ba = bond.GetBeginAtom().GetIdx()
                ba = extract_atom_ids.index(ba)
                ea = bond.GetEndAtom().GetIdx()
                ea = extract_atom_ids.index(ea)
                editeable_mol.AddBond(ba, ea, bond.GetBondType())
    return editeable_mol.GetMol()


def extract_fragment_from_mol(mol, extract_atom_ids):
    """Extract fragment from molecule preserving mapping to original atoms"""
    bonds_to_cut = []
    for bond in mol.GetBonds():
        atom_begin = bond.GetBeginAtom().GetIdx()
        atom_end = bond.GetEndAtom().GetIdx()
        if (atom_begin in extract_atom_ids) ^ (atom_end in extract_atom_ids):
            bonds_to_cut.append(bond.GetIdx())
    fragmented_molecule = Chem.FragmentOnBonds(mol, bonds_to_cut, addDummies=False)
    frag_idx = []
    frags = Chem.GetMolFrags(
        fragmented_molecule,
        asMols=True,
        sanitizeFrags=False,
        fragsMolAtomMapping=frag_idx,
    )
    for idx, frag in zip(frag_idx, frags):
        if sorted(list(idx)) == sorted(extract_atom_ids):
            return idx, frag
    warnings.warn("No Matching found")

# ============================================================================
# Decomposer classes (from magnet.src.chemutils.hypergraph)
# ============================================================================

class GeneralDecomposer(ABC):
    """Base class for molecular decomposers"""
    def __init__(self, frag_idx, core_mol):
        self.core_mol = core_mol
        self.frag_idx = frag_idx

    def decompose(self):
        """
        General Decomposition Function that iterates over current fragmented state
        """
        while True:
            new_frag_idx = []
            found = False
            for idx_in_core in self.frag_idx:
                # extract current fragment
                fragment = extract_valid_fragment(self.core_mol, idx_in_core)
                # apply check function to identify "object of interest"
                if self.check_func(fragment, idx_in_core):
                    found = True
                    # apply decompose function
                    output_frag_idx = self.fragment_func(fragment, idx_in_core)
                    for ofi in output_frag_idx:
                        # map back fragment_idx to molecule_idx
                        new_frag_idx.append(tuple([idx_in_core[f] for f in ofi]))
                else:
                    new_frag_idx.append(idx_in_core)
            self.frag_idx = new_frag_idx
            if not found:
                break
        return new_frag_idx

    @abstractmethod
    def check_func(self, fragment, idx_in_core):
        pass

    @abstractmethod
    def fragment_func(self, fragment, idx_in_core):
        pass


class BBBDecomposer(GeneralDecomposer):
    """
    Bond-Breaking Block (BBB) Decomposer
    Separate rings that are attached on only one joint atom
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_cut_bonds(self, fragment):
        ids_of_bonds_to_cut = []
        for bond in fragment.GetBonds():
            if bond.IsInRing():
                continue
            atom_begin = bond.GetBeginAtom()
            atom_end = bond.GetEndAtom()
            if not atom_begin.IsInRing() and not atom_end.IsInRing():
                continue
            ids_of_bonds_to_cut.append(bond.GetIdx())
        return ids_of_bonds_to_cut

    def check_func(self, fragment, idx_in_core):
        # check if a mixture of cyclic and acyclic bonds is present
        valid_fragment = extract_valid_fragment(self.core_mol, idx_in_core)
        is_in_ring = [a.IsInRing() for a in valid_fragment.GetAtoms()]
        ids_of_bonds_to_cut = self.get_cut_bonds(fragment)
        return (not all(is_in_ring) or any(is_in_ring)) and len(ids_of_bonds_to_cut) > 0

    def fragment_func(self, fragment, idx_in_core):
        ids_of_bonds_to_cut = self.get_cut_bonds(fragment)
        core_mol_frags = Chem.FragmentOnBonds(fragment, ids_of_bonds_to_cut, addDummies=False)
        frag_idx = Chem.GetMolFrags(core_mol_frags)
        # assign idx double to create hypernodes
        updated_frag_idx = []
        for f_idx in frag_idx:
            # chains and junctions need hypernodes to be added
            f_idx = set(f_idx)
            add_ids = []
            for b_id in ids_of_bonds_to_cut:
                bond = fragment.GetBondWithIdx(b_id)
                ba, ea = bond.GetBeginAtom(), bond.GetEndAtom()
                if ba.IsInRing() and ea.IsInRing():
                    # we handle this case later
                    continue
                bond_set = set([ba.GetIdx(), ea.GetIdx()])
                shared_nodes = f_idx.intersection(bond_set)
                assert len(shared_nodes) <= 1
                if shared_nodes:
                    shared_node_atom = fragment.GetAtomWithIdx(list(shared_nodes)[0])
                    if fragment.GetAtomWithIdx(list(shared_nodes)[0]).IsInRing():
                        continue
                    add_ids.append(list(bond_set - shared_nodes)[0])
            f_idx = tuple(f_idx.union(set(add_ids)))
            updated_frag_idx.append(f_idx)
        # additionally, add single-bond-bridges between rings as new fragment
        for b_id in ids_of_bonds_to_cut:
            bond = fragment.GetBondWithIdx(b_id)
            # bonds in ids to cut are always not in-rings
            begin_atom = bond.GetBeginAtom()
            end_atom = bond.GetEndAtom()
            if begin_atom.IsInRing() and end_atom.IsInRing():
                updated_frag_idx.append((begin_atom.GetIdx(), end_atom.GetIdx()))
        return updated_frag_idx


class JointRingDecomposer(GeneralDecomposer):
    """
    Joint Ring Decomposer
    Separate rings that are attached on a joint atom with degree 4
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_first_joint_ring(self, fragment, idx_in_core):
        # look for an atom of degree 4 in a ring
        for atom in fragment.GetAtoms():
            if atom.GetDegree() == 4 and atom.IsInRing():
                # check whether it connects two rings
                ri = fragment.GetRingInfo()
                neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
                check_ring_connector = [
                    [(t, n) for t in neighbors if (ri.AreAtomsInSameRing(t, n) and t != n)] for n in neighbors
                ]
                # out of all neighbors, we expect exactly two to be in the same ring
                if all([len(cr) == 1 for cr in check_ring_connector]):
                    if self.core_mol.GetAtomWithIdx(idx_in_core[atom.GetIdx()]).GetDegree() == 5:
                        return None
                    return atom

    def check_func(self, fragment, idx_in_core):
        return self.get_first_joint_ring(fragment, idx_in_core) is not None

    def fragment_func(self, fragment, idx_in_core):
        output_frag_idx = []
        atom = self.get_first_joint_ring(fragment, idx_in_core)
        atom_idx = atom.GetIdx()
        # detach joint atom and fragment, central atom will be added back later
        cut_bonds = [b.GetIdx() for b in atom.GetBonds()]
        core_mol_frags = Chem.FragmentOnBonds(fragment, cut_bonds, addDummies=False)
        new_frag_idx = Chem.GetMolFrags(core_mol_frags)
        output_frag_idx.extend([tuple(list(nfi) + [atom_idx]) for nfi in new_frag_idx if atom_idx not in nfi])
        return output_frag_idx


class JunctionDecomposer(GeneralDecomposer):
    """
    Junction Decomposer
    Separate junctions (atoms with degree >= 3 not in ring)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_junction_cut_bonds(self, fragment):
        # if it is a cyclic structure, it can not have a junction
        fragment_atoms = [a for a in fragment.GetAtoms()]
        junction_atoms = [a.GetIdx() for a in fragment_atoms if is_atom_junction(a)]
        # if there are no junction atoms, we can abort early
        if len(junction_atoms) == 0:
            return junction_atoms, None
        # connected junctions should stay intact
        current_junction = [junction_atoms.pop(0)]
        # find all neighboring junctions
        while True:
            neighbor_found = False
            for start_node in current_junction:
                for n in fragment.GetAtomWithIdx(start_node).GetNeighbors():
                    if is_atom_junction(n):
                        if n.GetIdx() not in current_junction:
                            neighbor_found = True
                            junction_atoms.remove(n.GetIdx())
                            current_junction.append(n.GetIdx())
            if not neighbor_found:
                break

        # find all neighbors of junction to add to fragment
        junction_neighbors = []
        for j in current_junction:
            j_atom = fragment.GetAtomWithIdx(j)
            junction_neighbors.extend([n.GetIdx() for n in j_atom.GetNeighbors()])
        junction_members = junction_neighbors + current_junction

        # cut all bonds that go outside of fragment except
        # those in rings because they constitute ring junctions
        cut_bonds = []
        for b in fragment.GetBonds():
            ba, ea = b.GetBeginAtom().GetIdx(), b.GetEndAtom().GetIdx()
            if (ba in junction_members) ^ (ea in junction_members):
                assert not b.IsInRing()
                cut_bonds.append(b.GetIdx())
        return cut_bonds, junction_members

    def check_func(self, fragment, idx_in_core):
        return len(self.get_junction_cut_bonds(fragment)[0]) > 0

    def fragment_func(self, fragment, idx_in_core):
        new_frag_idx = []
        cut_bonds, junction_members = self.get_junction_cut_bonds(fragment)
        if cut_bonds:
            core_mol_frags = Chem.FragmentOnBonds(fragment, cut_bonds, addDummies=False)
            cut_frag_idx = Chem.GetMolFrags(core_mol_frags)
            # assign idx double to create hypernodes
            for f_idx in cut_frag_idx:
                # junction fragment encountered, we just update fragment idx
                if not any([f in junction_members for f in f_idx]):
                    # for other cut fragments, we need to determine hypernodes
                    f_idx = set(f_idx)
                    add_ids = []
                    for b_id in cut_bonds:
                        # if any of the cut bonds coincides with atom in other fragment
                        bond = fragment.GetBondWithIdx(b_id)
                        ba, ea = bond.GetBeginAtom(), bond.GetEndAtom()
                        bond_set = set([ba.GetIdx(), ea.GetIdx()])
                        shared_nodes = f_idx.intersection(bond_set)
                        # add end node of bond as hypernode
                        if shared_nodes:
                            add_ids.append(list(bond_set - shared_nodes)[0])
                    f_idx = tuple(f_idx.union(set(add_ids)))
                new_frag_idx.append(f_idx)
        else:
            return [tuple(range(fragment.GetNumAtoms()))]
        return new_frag_idx

# ============================================================================
# Main MolDecomposition class (from magnet.src.chemutils.hypergraph)
# ============================================================================

MAX_JOIN_DEGREE = 3

class MolDecomposition:
    """
    Decompose molecule into motifs using MAGNet algorithm.
    Motifs are obtained through systematic decomposition into:
    - Bond-Breaking Blocks (BBB)
    - Joint Rings
    - Junctions
    """
    
    def __init__(self, input_mol: str, plot_decomp_steps: bool = False):
        mol = Chem.MolFromSmiles(input_mol)
        Chem.Kekulize(mol)
        self.mol = mol
        self.plot_decomp_steps = plot_decomp_steps

        # initialize node mapping
        self.nodes = {i: [] for i in range(self.mol.GetNumAtoms())}

        # apply decomposition to molecule input
        self.decompose()

        # check soundness of decomposition
        self.verify_sound_decomposition()

        # already prepare costly features for later use
        self.create_motif_map()
        self.prepare_fingerprints()

    def decompose(self):
        """Apply MAGNet decomposition algorithm"""
        # identify leaf atoms and set core molecule
        idx_in_full_mol = self.set_leaf_atoms()
        frag_idx = [tuple(range(len(idx_in_full_mol)))]

        # decompose according to MAGNet algorithm
        decomposers = [BBBDecomposer, JointRingDecomposer, JunctionDecomposer]
        for decomp in decomposers:
            frag_idx = decomp(frag_idx=frag_idx, core_mol=self.core_mol).decompose()

        # map shapes back to full molecule
        for k, idx in enumerate(frag_idx):
            for idx_in_core in idx:
                original_idx = idx_in_full_mol[idx_in_core]
                self.nodes[original_idx] = self.nodes[original_idx] + [k]

    def set_leaf_atoms(self):
        """Identify and remove leaf atoms from decomposition"""
        adj = Chem.rdmolops.GetAdjacencyMatrix(self.mol)
        graph_no_leaf = nx.from_numpy_array(np.triu(adj), create_using=nx.Graph)
        for atom in self.mol.GetAtoms():
            graph_no_leaf.nodes[atom.GetIdx()]["label"] = atom.GetSymbol()
        atom_types, leaf_atoms = [], []
        for k in range(graph_no_leaf.number_of_nodes()):
            atom_types.append(ATOM_LIST.index(graph_no_leaf.nodes[k]["label"]))
        sorted_idx = np.flip(np.argsort(atom_types))

        # molecules consisting of just small chains mess up the leaf finding
        if self.mol.GetNumAtoms() > 2:
            for idx in sorted_idx:
                if graph_no_leaf.degree[idx.item()] == 1:
                    neighbour = list(graph_no_leaf.neighbors(idx.item()))[0]
                    if graph_no_leaf.degree[neighbour] not in [2, 4]:
                        graph_no_leaf.remove_node(idx.item())
                        leaf_atoms.append(idx.item())

        # set leaf atoms in global mapping
        for idx in leaf_atoms:
            self.nodes[idx] = self.nodes[idx] + [-1]

        # extract core molecule -> molecule without leafs
        core_mol_idx = [k for k, v in self.nodes.items() if len(v) == 0]
        if leaf_atoms:
            idx_in_mol, self.core_mol = extract_fragment_from_mol(self.mol, core_mol_idx)
            self.valid_core_mol = extract_valid_fragment(self.mol, core_mol_idx)
        else:
            self.core_mol, self.valid_core_mol = self.mol, self.mol
            idx_in_mol = tuple(core_mol_idx)
        return idx_in_mol

    def prepare_fingerprints(self):
        """Precompute fingerprints for the molecule"""
        self.fingerprint_mol = compute_fingerprint(self.mol)
        self.fingerprint_mol = np.array(self.fingerprint_mol, dtype=np.float32)
        if np.any(np.isnan(self.fingerprint_mol)):
            self.fingerprint_mol[np.isnan(self.fingerprint_mol)] = 0

    def verify_sound_decomposition(self):
        """Verify that decomposition satisfies MAGNet constraints"""
        # check: hypernode can only be in 2 motifs
        for k, v in self.nodes.items():
            if not 1 <= len(v) <= 2:
                if not self.mol.GetAtomWithIdx(k).IsInRing():
                    raise ValueError("Acyclic hypernode can only be in 2 motifs")
        self.create_motif_map()
        # check: overlap of motifs can only be 1 node
        for key1 in self.id_to_hash.keys():
            shape_node_outer = [k for (k, v) in self.nodes.items() if key1 in v]
            for key2 in self.id_to_hash.keys():
                shape_node_inner = [k for (k, v) in self.nodes.items() if key2 in v]
                if key1 != key2:
                    intersection_len = len(set(shape_node_outer).intersection(set(shape_node_inner)))
                    assert intersection_len in [0, 1]

    def create_motif_map(self):
        """Create mapping between motif IDs, fragments, and hashes"""
        self.id_to_fragment, self.id_to_hash, self.hash_to_id = dict(), dict(), dict()
        self.id_to_hash[-1] = -1
        self.hash_to_id[-1] = -1
        
        # Get maximum motif ID (excluding leaf atoms marked as -1)
        all_ids = []
        for node_list in self.nodes.values():
            all_ids.extend([nid for nid in node_list if nid >= 0])
        
        if not all_ids:
            return  # No motifs to process
        
        num_classes = max(all_ids)
        
        # Iterate over all motif IDs from 0 to num_classes (original behavior)
        for i in range(num_classes + 1):
            atoms_in_motif = [k for k, v in self.nodes.items() if (i in v)]
            
            frag = extract_valid_fragment(self.mol, atoms_in_motif)
            # Attention: since we sanitize, we can not rely on ordering of SMILES
            Chem.SanitizeMol(frag)
            adjacency = Chem.GetAdjacencyMatrix(frag)
            graph = nx.from_numpy_array(np.triu(adjacency), create_using=nx.Graph)
            graph_hash = nx.weisfeiler_lehman_graph_hash(graph)
            self.id_to_hash[i] = graph_hash
            self.id_to_fragment[i] = Chem.MolToSmiles(frag)
            self.hash_to_id[graph_hash] = i

    def get_motifs(self):
        """Return list of motif SMILES strings"""
        motifs = []
        for i in self.id_to_fragment.keys():
            if i >= 0:  # Skip leaf atoms (marked as -1)
                motifs.append(self.id_to_fragment[i])
        return motifs
