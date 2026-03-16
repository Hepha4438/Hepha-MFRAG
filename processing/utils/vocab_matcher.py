"""
Vocabulary matching utilities for HES data generation.
Maps decomposed motifs/shapes to their corresponding vocabulary IDs.
"""
import pickle
import numpy as np
from rdkit import Chem
from pathlib import Path


class VocabularyMatcher:
    """Handles matching of motifs and shapes to vocabulary IDs"""
    
    def __init__(self, motif_vocab_path, shape_vocab_path):
        """
        Initialize vocabulary matcher.
        
        Args:
            motif_vocab_path: Path to motif_vocab.pkl (dict: motif_smiles -> ecfp)
            shape_vocab_path: Path to shape_vocab.pkl (dict: topology_hash -> motif_ids)
        """
        self.motif_vocab_path = Path(motif_vocab_path)
        self.shape_vocab_path = Path(shape_vocab_path)
        
        # Load vocabularies
        with open(self.motif_vocab_path, 'rb') as f:
            motif_vocab_raw = pickle.load(f)
        
        with open(self.shape_vocab_path, 'rb') as f:
            shape_vocab_raw = pickle.load(f)
        
        # Handle motif vocab: could be dict or list
        if isinstance(motif_vocab_raw, dict):
            # Format: {motif_smiles: ecfp_array}
            self.motif_vocab = list(motif_vocab_raw.items())
        else:
            # Already a list/tuple
            self.motif_vocab = motif_vocab_raw
        
        # Handle shape vocab: could be dict or list
        if isinstance(shape_vocab_raw, dict):
            # Format: {topology_hash: motif_ids}
            self.shape_vocab = list(shape_vocab_raw.items())
        else:
            # Already a list/tuple
            self.shape_vocab = shape_vocab_raw
        
        # Build reverse mapping: SMILES -> ID for faster lookup
        self.smiles_to_motif_id = {}
        for i, (motif_smiles, _) in enumerate(self.motif_vocab):
            # Canonicalize SMILES for consistent matching
            canonical_smiles = self._canonicalize_smiles(motif_smiles)
            if canonical_smiles:
                self.smiles_to_motif_id[canonical_smiles] = i
        
        # Build reverse mapping: topology hash -> ID for shapes
        self.hash_to_shape_id = {}
        for i, shape_entry in enumerate(self.shape_vocab):
            # shape_entry is a tuple: (hash/key, value)
            if isinstance(shape_entry, tuple) and len(shape_entry) == 2:
                shape_key = shape_entry[0]
                self.hash_to_shape_id[shape_key] = i
    
    def _canonicalize_smiles(self, smiles):
        """
        Canonicalize SMILES string.
        
        Args:
            smiles: SMILES string
        
        Returns:
            Canonical SMILES or None if invalid
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return Chem.MolToSmiles(mol)
        except:
            return None
    
    def get_motif_id(self, motif_smiles):
        """
        Get vocabulary ID for a motif SMILES.
        
        Args:
            motif_smiles: SMILES string of the motif
        
        Returns:
            Integer vocabulary ID, or -1 if not found
        """
        canonical = self._canonicalize_smiles(motif_smiles)
        if canonical is None:
            return -1
        
        return self.smiles_to_motif_id.get(canonical, -1)
    
    def get_shape_id(self, topology_hash):
        """
        Get vocabulary ID for a shape (topology group).
        
        Args:
            topology_hash: Weisfeiler-Lehman graph hash
        
        Returns:
            Integer vocabulary ID, or -1 if not found
        """
        return self.hash_to_shape_id.get(topology_hash, -1)
    
    def get_motif_ids_from_decomposition(self, mol_decomp):
        """
        Extract motif vocabulary IDs from a decomposed molecule.
        
        Args:
            mol_decomp: MolDecomposition instance
        
        Returns:
            List of motif vocabulary IDs (one per motif in the molecule)
        """
        motif_ids = []
        
        for i in range(len(mol_decomp.id_to_fragment)):
            if i >= 0:  # Skip leaf atoms
                motif_smiles = mol_decomp.id_to_fragment[i]
                vocab_id = self.get_motif_id(motif_smiles)
                motif_ids.append(vocab_id)
        
        return motif_ids
    
    def get_shape_ids_from_motif_ids(self, motif_ids, mol_decomp):
        """
        Get shape vocabulary IDs from motif IDs.
        
        Groups motifs by their topology hash into shapes.
        
        Args:
            motif_ids: List of motif vocab IDs
            mol_decomp: MolDecomposition instance (for hash lookup)
        
        Returns:
            List of shape vocabulary IDs (one per unique topology in the molecule)
        """
        seen_hashes = {}
        shape_ids = []
        
        # Map from topological hash to shape ID
        for motif_idx in sorted(mol_decomp.id_to_hash.keys()):
            if motif_idx >= 0:  # Skip leaf atoms
                topo_hash = mol_decomp.id_to_hash[motif_idx]
                
                if topo_hash not in seen_hashes:
                    shape_id = self.get_shape_id(topo_hash)
                    if shape_id >= 0:
                        seen_hashes[topo_hash] = shape_id
                        shape_ids.append(shape_id)
        
        return shape_ids
    
    def get_stats(self):
        """Return vocabulary statistics"""
        return {
            'num_motifs': len(self.motif_vocab),
            'num_shapes': len(self.shape_vocab),
        }
