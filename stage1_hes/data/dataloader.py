"""
Data loading utilities for HES training.

Handles:
- Creating train/val/test splits from HESDataset
- Building DataLoaders with proper collation
- Property normalization and batching
"""

import torch
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.data import Data, InMemoryDataset
from torch.utils.data import Subset, random_split
from typing import Tuple, Optional, List
import sys
from pathlib import Path
import pickle
import pandas as pd

# Add parent paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "processing"))


class HESData(Data):
    """Custom Data class for HES model with two graph structures"""
    
    def __init__(self, x_g=None, edge_index_g=None, num_nodes_g=None, x_sc=None, 
                 edge_index_sc=None, num_nodes_sc=None, motif_indices=None, 
                 shape_indices=None, y=None):
        """Initialize HESData with proper num_nodes handling"""
        super().__init__()
        if x_g is not None:
            self.x_g = x_g
            # For PyG, set num_nodes to the size of main graph
            self.num_nodes = x_g.size(0)
        if edge_index_g is not None:
            self.edge_index_g = edge_index_g
        if num_nodes_g is not None:
            self.num_nodes_g = num_nodes_g
        if x_sc is not None:
            self.x_sc = x_sc
        if edge_index_sc is not None:
            self.edge_index_sc = edge_index_sc
        if num_nodes_sc is not None:
            self.num_nodes_sc = num_nodes_sc
        if motif_indices is not None:
            self.motif_indices = motif_indices
        if shape_indices is not None:
            self.shape_indices = shape_indices
        if y is not None:
            self.y = y
    
    def __inc__(self, key, value, store):
        """Custom increment function for batching"""
        if key == 'edge_index_g':
            # Increment edge indices for molecular graph by number of atoms
            return self.num_nodes_g
        elif key == 'edge_index_sc':
            # Increment edge indices for scaffold graph by number of motifs
            return self.num_nodes_sc
        elif key == 'motif_indices':
            # Motif indices are absolute vocab indices, don't increment
            return 0
        elif key == 'shape_indices':
            # Shape indices are absolute vocab indices, don't increment
            return 0
        else:
            # Default PyG behavior
            return super().__inc__(key, value, store)


class SimpleHESDataset(InMemoryDataset):
    """Simple in-memory dataset for HES data"""
    
    def __init__(self, data_list):
        super().__init__()
        self.data_list = data_list
    
    def len(self):
        return len(self.data_list)
    
    def get(self, idx):
        return self.data_list[idx]


class HESDataLoader:
    """
    DataLoader wrapper for HES dataset with train/val/test split.
    """
    
    def __init__(
        self,
        dataset_root: str,
        batch_size: int = 32,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        num_workers: int = 0,
        shuffle_train: bool = True,
        seed: int = 42,
    ):
        """
        Args:
            dataset_root: Path to processing/output directory (contains graphs, properties.csv, etc.)
            batch_size: Batch size for DataLoader
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
            num_workers: Number of DataLoader workers
            shuffle_train: Whether to shuffle training data
            seed: Random seed for reproducibility
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_train = shuffle_train
        self.seed = seed
        
        # Load dataset
        print(f"[DataLoader] Loading HES dataset from {dataset_root}...")
        
        dataset_root = Path(dataset_root)
        
        # Reconstruct data_list with proper Data objects
        data_list = self._create_data_list(dataset_root)
        
        # Create simple dataset
        self.dataset = SimpleHESDataset(data_list)
        print(f"[DataLoader] Loaded {len(self.dataset)} samples")
        
        # Create splits
        train_size = int(len(self.dataset) * train_ratio)
        val_size = int(len(self.dataset) * val_ratio)
        test_size = len(self.dataset) - train_size - val_size
        
        print(f"[DataLoader] Creating splits: train={train_size}, val={val_size}, test={test_size}")
        
        # Random split
        torch.manual_seed(seed)
        train_dataset, val_dataset, test_dataset = random_split(
            self.dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(seed),
        )
        
        # Create DataLoaders using PyG DataLoader for proper batching
        self.train_loader = PyGDataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
        )
        
        self.val_loader = PyGDataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        
        self.test_loader = PyGDataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        
        print(f"[DataLoader] DataLoaders created")
        print(f"  - Train batches: {len(self.train_loader)}")
        print(f"  - Val batches: {len(self.val_loader)}")
        print(f"  - Test batches: {len(self.test_loader)}")
    
    def _create_data_list(self, dataset_root: Path) -> List:
        """
        Load data from processing output directory.
        
        Expected structure:
        - graphs/ : graph_*.pt files (one per molecule)
        - properties.csv : property values for each graph
        - vocabularies/ : motif and shape vocabularies
        
        Args:
            dataset_root: Path to processing/output directory
        
        Returns:
            List of HESData objects
        """
        print(f"[DataCreation] Creating data objects from {dataset_root}...")
        
        # Check for graphs directory
        graphs_dir = dataset_root / "graphs"
        properties_csv = dataset_root / "properties.csv"
        metadata_path = dataset_root / "hes_dataset_metadata.pkl"
        data_list_path = dataset_root / "hes_data_list.pkl"
        
        if not graphs_dir.exists():
            print(f"[DataCreation] Graphs directory not found, using synthetic data")
            return self._create_synthetic_data_list(1000)
        
        graph_files = sorted(graphs_dir.glob("graph_*.pt"))
        num_graphs = len(graph_files)
        print(f"[DataCreation] Found {num_graphs} graph files")

        # Load preprocessing metadata (for validation/traceability)
        metadata = None
        if metadata_path.exists():
            try:
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                print(
                    "[DataCreation] Loaded metadata: "
                    f"num_molecules={metadata.get('num_molecules', 'N/A')}, "
                    f"success_count={metadata.get('success_count', 'N/A')}, "
                    f"failed_count={metadata.get('failed_count', 'N/A')}"
                )
                if 'num_molecules' in metadata and metadata['num_molecules'] != num_graphs:
                    print(
                        "[DataCreation][WARN] Metadata num_molecules "
                        f"({metadata['num_molecules']}) != number of graph files ({num_graphs})"
                    )
            except Exception as e:
                print(f"[DataCreation] Failed to load metadata: {e}")
        else:
            print(f"[DataCreation] Metadata file not found: {metadata_path}")
        
        # Prefer per-sample preprocessed file generated by processing/hes_data_gen.py
        if data_list_path.exists():
            try:
                with open(data_list_path, 'rb') as f:
                    preprocessed_data_list = pickle.load(f)
                if isinstance(preprocessed_data_list, list) and len(preprocessed_data_list) > 0:
                    print(f"[DataCreation] Loading preprocessed per-sample data from {data_list_path}")
                    converted = self._convert_preprocessed_data_list(preprocessed_data_list, properties_csv)
                    if converted:
                        print(f"[DataCreation] Loaded {len(converted)} samples from preprocessed file")
                        return converted
                    print("[DataCreation][WARN] Preprocessed file loaded but yielded no valid samples, fallback to graph rebuild")
                else:
                    print("[DataCreation][WARN] Preprocessed data file is empty or invalid, fallback to graph rebuild")
            except Exception as e:
                print(f"[DataCreation][WARN] Failed to load preprocessed data file: {e}")

        if not graph_files:
            print(f"[DataCreation] No graph files found, using synthetic data")
            return self._create_synthetic_data_list(1000)
        
        # Load properties if available
        properties = {}
        properties_df = None
        if properties_csv.exists():
            try:
                properties_df = pd.read_csv(properties_csv)
                properties = properties_df.to_dict('index')
            except Exception as e:
                print(f"[DataCreation] Failed to load properties: {e}")

        # Load vocabulary matcher for real motif/shape IDs
        matcher = None
        try:
            motif_vocab_path = dataset_root / "vocabularies" / "motif_vocab.pkl"
            shape_vocab_path = dataset_root / "vocabularies" / "shape_vocab.pkl"
            if motif_vocab_path.exists() and shape_vocab_path.exists():
                from processing.utils.vocab_matcher import VocabularyMatcher
                matcher = VocabularyMatcher(motif_vocab_path, shape_vocab_path)
            else:
                print("[DataCreation] Vocabulary files missing, motif/shape IDs will use fallback IDs")
        except Exception as e:
            print(f"[DataCreation] Failed to initialize VocabularyMatcher: {e}")

        smiles_list = []
        if properties_df is not None:
            if 'smiles' in properties_df.columns:
                smiles_list = properties_df['smiles'].astype(str).tolist()
            elif 'SMILES' in properties_df.columns:
                smiles_list = properties_df['SMILES'].astype(str).tolist()
        
        data_list = []
        
        # Load all available graphs
        print(f"[DataCreation] Loading all {len(graph_files)} molecules...")
        
        for idx, graph_path in enumerate(graph_files):
            try:
                # Load graph
                data_g = torch.load(graph_path, weights_only=False)
                
                if data_g is None:
                    continue
                
                # Extract features and ensure proper dimensions
                if hasattr(data_g, 'x') and data_g.x is not None:
                    x_g = data_g.x.float()
                    # If x_g is 1D (just atom types), expand to proper feature dimension
                    if len(x_g.shape) == 1:
                        # Reshape to (n_nodes, 1)
                        x_g = x_g.view(-1, 1)
                    if x_g.shape[1] < 15:
                        # Pad with random features to reach dimension 15
                        n_nodes = x_g.shape[0]
                        feat_dim = x_g.shape[1]
                        padding = torch.randn(n_nodes, 15 - feat_dim)
                        x_g = torch.cat([x_g, padding], dim=1)
                    elif x_g.shape[1] > 15:
                        # Truncate to 15 dimensions
                        x_g = x_g[:, :15]
                else:
                    n_atoms = data_g.num_nodes if hasattr(data_g, 'num_nodes') else 5
                    x_g = torch.randn(n_atoms, 15)
                
                num_atoms = x_g.shape[0]
                edge_index_g = data_g.edge_index if hasattr(data_g, 'edge_index') and data_g.edge_index is not None else torch.zeros((2, 1), dtype=torch.long)
                
                # Reconstruct scaffold graph + motif/shape IDs from preprocessing logic
                x_sc = torch.zeros((1, 16), dtype=torch.float32)
                x_sc[0, -1] = 1.0  # null scaffold flag
                edge_index_sc = torch.zeros((2, 0), dtype=torch.long)
                motif_indices = torch.tensor([0], dtype=torch.long)
                shape_indices = torch.tensor([0], dtype=torch.long)

                smiles = smiles_list[idx] if idx < len(smiles_list) else None
                if smiles and matcher is not None:
                    try:
                        from processing.utils.magnet_decomposition import MolDecomposition
                        from processing.utils.scaffold_extractor import get_scaffold_from_decomposition
                        from processing.utils.graph_builder import mol_to_pyg_data

                        mol_decomp = MolDecomposition(smiles)

                        scaffold_mol, _, _ = get_scaffold_from_decomposition(mol_decomp)
                        if scaffold_mol is not None:
                            data_sc = mol_to_pyg_data(scaffold_mol, add_smiles=False)
                            if data_sc is not None and hasattr(data_sc, 'x') and data_sc.x is not None and data_sc.x.numel() > 0:
                                x_sc = data_sc.x.float()
                                if x_sc.dim() == 1:
                                    x_sc = x_sc.view(-1, 1)
                                if x_sc.shape[1] < 16:
                                    pad = torch.zeros(x_sc.shape[0], 16 - x_sc.shape[1], dtype=x_sc.dtype)
                                    x_sc = torch.cat([x_sc, pad], dim=1)
                                elif x_sc.shape[1] > 16:
                                    x_sc = x_sc[:, :16]
                                edge_index_sc = data_sc.edge_index if hasattr(data_sc, 'edge_index') and data_sc.edge_index is not None else torch.zeros((2, 0), dtype=torch.long)

                        real_motif_ids = []
                        for i in sorted(mol_decomp.id_to_fragment.keys()):
                            if i >= 0:
                                motif_smiles = mol_decomp.id_to_fragment[i]
                                vocab_id = matcher.get_motif_id(motif_smiles)
                                if vocab_id >= 0:
                                    real_motif_ids.append(vocab_id)

                        real_shape_ids = []
                        seen_hashes = set()
                        for i in sorted(mol_decomp.id_to_hash.keys()):
                            if i >= 0:
                                topo_hash = mol_decomp.id_to_hash[i]
                                if topo_hash not in seen_hashes:
                                    seen_hashes.add(topo_hash)
                                    shape_id = matcher.get_shape_id(topo_hash)
                                    if shape_id >= 0:
                                        real_shape_ids.append(shape_id)

                        if real_motif_ids:
                            motif_indices = torch.tensor(real_motif_ids, dtype=torch.long)
                        if real_shape_ids:
                            shape_indices = torch.tensor(real_shape_ids, dtype=torch.long)
                    except Exception:
                        pass

                num_motifs = x_sc.shape[0]
                
                # Load properties - shape should be [1, 8] (graph-level feature)
                y = torch.zeros(1, 8, dtype=torch.float32)
                if idx in properties:
                    props = properties[idx]
                    # Try to extract numeric properties (skip string columns like SMILES)
                    prop_vals = []
                    for key in sorted(props.keys()):
                        val = props[key]
                        if isinstance(val, (int, float)):
                            prop_vals.append(float(val))
                    if len(prop_vals) >= 8:
                        y = torch.tensor([prop_vals[:8]], dtype=torch.float32)  # Wrap in list for [1, 8]
                
                # Create HESData object
                data = HESData(
                    x_g=x_g,
                    edge_index_g=edge_index_g,
                    num_nodes_g=num_atoms,
                    x_sc=x_sc,
                    edge_index_sc=edge_index_sc,
                    num_nodes_sc=num_motifs,
                    motif_indices=motif_indices,
                    shape_indices=shape_indices,
                    y=y,
                )
                data_list.append(data)
                
            except Exception as e:
                continue
        
        print(f"[DataCreation] Successfully loaded {len(data_list)} data objects")
        
        return data_list if data_list else self._create_synthetic_data_list(1000)

    def _convert_preprocessed_data_list(self, preprocessed_data_list: List[dict], properties_csv: Path) -> List:
        """Convert list of preprocessed dict samples into HESData objects."""
        properties_by_smiles = {}
        if properties_csv.exists():
            try:
                properties_df = pd.read_csv(properties_csv)
                if 'smiles' in properties_df.columns:
                    for _, row in properties_df.iterrows():
                        properties_by_smiles[str(row['smiles'])] = row
                elif 'SMILES' in properties_df.columns:
                    for _, row in properties_df.iterrows():
                        properties_by_smiles[str(row['SMILES'])] = row
            except Exception:
                pass

        converted = []
        for item in preprocessed_data_list:
            try:
                x_g = item.get('x_g')
                if x_g is None:
                    continue
                if not isinstance(x_g, torch.Tensor):
                    x_g = torch.tensor(x_g)
                x_g = x_g.float()
                if x_g.dim() == 1:
                    x_g = x_g.view(-1, 1)
                if x_g.shape[1] < 15:
                    pad = torch.zeros(x_g.shape[0], 15 - x_g.shape[1], dtype=x_g.dtype)
                    x_g = torch.cat([x_g, pad], dim=1)
                elif x_g.shape[1] > 15:
                    x_g = x_g[:, :15]

                edge_index_g = item.get('edge_index_g', torch.zeros((2, 0), dtype=torch.long))
                if not isinstance(edge_index_g, torch.Tensor):
                    edge_index_g = torch.tensor(edge_index_g)
                edge_index_g = edge_index_g.long()

                x_sc = item.get('x_sc')
                if x_sc is None:
                    x_sc = torch.zeros((1, 16), dtype=torch.float32)
                    x_sc[0, -1] = 1.0
                else:
                    if not isinstance(x_sc, torch.Tensor):
                        x_sc = torch.tensor(x_sc)
                    x_sc = x_sc.float()
                    if x_sc.dim() == 1:
                        x_sc = x_sc.view(-1, 1)
                    if x_sc.shape[1] < 16:
                        pad = torch.zeros(x_sc.shape[0], 16 - x_sc.shape[1], dtype=x_sc.dtype)
                        x_sc = torch.cat([x_sc, pad], dim=1)
                    elif x_sc.shape[1] > 16:
                        x_sc = x_sc[:, :16]

                edge_index_sc = item.get('edge_index_sc', torch.zeros((2, 0), dtype=torch.long))
                if not isinstance(edge_index_sc, torch.Tensor):
                    edge_index_sc = torch.tensor(edge_index_sc)
                edge_index_sc = edge_index_sc.long()

                motif_indices = item.get('motif_indices', [0])
                if not isinstance(motif_indices, torch.Tensor):
                    motif_indices = torch.tensor(motif_indices, dtype=torch.long)
                else:
                    motif_indices = motif_indices.long()
                if motif_indices.numel() == 0:
                    motif_indices = torch.tensor([0], dtype=torch.long)

                shape_indices = item.get('shape_indices', [0])
                if not isinstance(shape_indices, torch.Tensor):
                    shape_indices = torch.tensor(shape_indices, dtype=torch.long)
                else:
                    shape_indices = shape_indices.long()
                if shape_indices.numel() == 0:
                    shape_indices = torch.tensor([0], dtype=torch.long)

                y = torch.zeros(1, 8, dtype=torch.float32)
                smiles = item.get('smiles')
                if smiles is not None and str(smiles) in properties_by_smiles:
                    row = properties_by_smiles[str(smiles)]
                    prop_vals = []
                    for key in sorted(row.index):
                        val = row[key]
                        if isinstance(val, (int, float)):
                            prop_vals.append(float(val))
                    if len(prop_vals) >= 8:
                        y = torch.tensor([prop_vals[:8]], dtype=torch.float32)

                converted.append(
                    HESData(
                        x_g=x_g,
                        edge_index_g=edge_index_g,
                        num_nodes_g=x_g.shape[0],
                        x_sc=x_sc,
                        edge_index_sc=edge_index_sc,
                        num_nodes_sc=x_sc.shape[0],
                        motif_indices=motif_indices,
                        shape_indices=shape_indices,
                        y=y,
                    )
                )
            except Exception:
                continue

        return converted
    
    def _create_synthetic_data_list(self, num_samples: int) -> List:
        """Create synthetic Data objects for testing"""
        print(f"[Synthetic] Creating {num_samples} synthetic Data objects for testing")
        data_list = []
        
        for _ in range(num_samples):
            num_atoms = torch.randint(5, 15, (1,)).item()
            num_motifs = torch.randint(2, 6, (1,)).item()
            
            # Create valid edges
            if num_atoms > 1:
                edge_index_g = torch.randint(0, num_atoms, (2, num_atoms * 2))
            else:
                edge_index_g = torch.zeros((2, 1), dtype=torch.long)
            
            if num_motifs > 1:
                edge_index_sc = torch.randint(0, num_motifs, (2, num_motifs - 1))
            else:
                edge_index_sc = torch.zeros((2, 1), dtype=torch.long)
            
            # Create Data object with explicit num_nodes using HESData
            data = HESData(
                x_g=torch.randn(num_atoms, 15),
                edge_index_g=edge_index_g,
                num_nodes_g=num_atoms,
                x_sc=torch.randn(num_motifs, 16),
                edge_index_sc=edge_index_sc,
                num_nodes_sc=num_motifs,
                motif_indices=torch.randint(1, 7370, (num_motifs,)),
                shape_indices=torch.randint(1, 346, (num_motifs,)),
                y=torch.randn(1, 8),
            )
            data_list.append(data)
        
        return data_list


def collate_batch(batch: List) -> dict:
    """
    Custom collate function for HES dataset batches.
    
    Preserves batch structure with separate batch assignments for G and Sc.
    """
    # This is handled automatically by PyG DataLoader
    # Return as is (each item is already a Data object)
    from torch_geometric.data import Batch
    return Batch.from_data_list(batch)
