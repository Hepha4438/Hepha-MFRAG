"""
PyTorch Geometric Dataset for HES (Hierarchical Equivariant Scaffold) molecules.
Returns atomic graphs, scaffold graphs, and component indices with properties.
"""
import pickle
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader as PyGDataLoader


class HESDataset(Dataset):
    """
    PyTorch Geometric dataset for HES data generation.
    
    Each data point contains:
    - Atomic graph G: Full molecular graph with atoms as nodes
    - Scaffold graph Sc: Subgraph containing ring systems and junctions
    - Motif indices: IDs of motifs in G mapped to vocabulary
    - Shape indices: IDs of shapes in Sc mapped to vocabulary
    - Properties: 8-dim tensor (logP, qed, SAS, docking_parp1, docking_fa7, docking_5ht1b, docking_braf, docking_jak2)
    """
    
    def __init__(self, data_list, properties_df=None, lazy_load=True):
        """
        Initialize HES dataset.
        
        Args:
            data_list: List of preprocessed data dictionaries, each containing:
                - 'smiles': SMILES string
                - 'x_g': Node features tensor for atomic graph
                - 'edge_index_g': Edge indices for atomic graph
                - 'edge_attr_g': Edge attributes for atomic graph
                - 'x_sc': Node features tensor for scaffold graph (or None)
                - 'edge_index_sc': Edge indices for scaffold graph (or None)
                - 'edge_attr_sc': Edge attributes for scaffold graph (or None)
                - 'motif_indices': Tensor of motif vocabulary IDs
                - 'shape_indices': Tensor of shape vocabulary IDs
            properties_df: DataFrame with properties indexed by SMILES
            lazy_load: If True, don't load all data into memory
        """
        super().__init__()
        self.data_list = data_list
        self.properties_df = properties_df
        self.lazy_load = lazy_load
        
        # For lazy loading, keep data on disk references
        # For eager loading, preprocess all
        if not lazy_load:
            self.preprocessed_data = self._preprocess_all()
        else:
            self.preprocessed_data = None
    
    def _preprocess_all(self):
        """Preprocess all data into memory (if not lazy loading)"""
        return [self._process_single(i) for i in range(len(self.data_list))]
    
    def _process_single(self, idx):
        """Process a single data point into PyG Data object"""
        data_dict = self.data_list[idx]
        
        # Create Data object for atomic graph
        data_g = Data(
            x=data_dict['x_g'],
            edge_index=data_dict['edge_index_g'],
            edge_attr=data_dict['edge_attr_g'],
            num_nodes=data_dict['x_g'].shape[0] if data_dict['x_g'] is not None else 0,
        )
        
        # Create Data object for scaffold graph (may be None for single-atom molecules)
        if data_dict['x_sc'] is not None:
            data_sc = Data(
                x=data_dict['x_sc'],
                edge_index=data_dict['edge_index_sc'],
                edge_attr=data_dict['edge_attr_sc'],
                num_nodes=data_dict['x_sc'].shape[0],
            )
        else:
            data_sc = None
        
        # Get properties if available
        properties = None
        if self.properties_df is not None:
            smiles = data_dict.get('smiles')
            if smiles in self.properties_df.index:
                props = self.properties_df.loc[smiles]
                # Select required property columns
                property_cols = ['logP', 'qed', 'SAS', 'docking_parp1', 'docking_fa7', 
                                'docking_5ht1b', 'docking_braf', 'docking_jak2']
                properties = torch.tensor(
                    [props[col] if col in props.index else 0.0 for col in property_cols],
                    dtype=torch.float32
                )
        
        # If properties not found, create zero tensor
        if properties is None:
            properties = torch.zeros(8, dtype=torch.float32)
        
        # Create combined Data object with all fields
        combined_data = Data(
            # Atomic graph
            x_g=data_g.x,
            edge_index_g=data_g.edge_index,
            edge_attr_g=data_g.edge_attr,
            num_nodes_g=data_g.num_nodes,
            # Scaffold graph
            x_sc=data_sc.x if data_sc is not None else torch.zeros((0, 1), dtype=torch.long),
            edge_index_sc=data_sc.edge_index if data_sc is not None else torch.zeros((2, 0), dtype=torch.long),
            edge_attr_sc=data_sc.edge_attr if data_sc is not None else torch.zeros(0, dtype=torch.long),
            num_nodes_sc=data_sc.num_nodes if data_sc is not None else 0,
            # Component indices
            motif_indices=torch.tensor(data_dict['motif_indices'], dtype=torch.long),
            shape_indices=torch.tensor(data_dict['shape_indices'], dtype=torch.long),
            # Properties
            y=properties,
        )
        
        return combined_data
    
    def len(self):
        """Return dataset length"""
        return len(self.data_list)
    
    def get(self, idx):
        """Get item at index"""
        if self.preprocessed_data is not None:
            return self.preprocessed_data[idx]
        else:
            return self._process_single(idx)


class HESDataLoader(PyGDataLoader):
    """
    Custom DataLoader for HES dataset with proper batching.
    Inherits from PyG DataLoader for automatic batching.
    """
    
    def __init__(self, dataset, batch_size=32, shuffle=True, num_workers=0, **kwargs):
        """
        Initialize HES DataLoader.
        
        Args:
            dataset: HESDataset instance
            batch_size: Batch size
            shuffle: Whether to shuffle
            num_workers: Number of workers (0 for single-process)
            **kwargs: Additional arguments for DataLoader
        """
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            **kwargs
        )


def create_hes_dataset(data_list, properties_csv=None, lazy_load=True):
    """
    Create HES dataset from preprocessed data.
    
    Args:
        data_list: List of preprocessed data dictionaries
        properties_csv: Path to properties CSV file
        lazy_load: If True, use lazy loading
    
    Returns:
        HESDataset instance
    """
    properties_df = None
    if properties_csv is not None:
        properties_csv = Path(properties_csv)
        if properties_csv.exists():
            properties_df = pd.read_csv(properties_csv)
            # Set smiles as index if it's a column
            if 'smiles' in properties_df.columns:
                properties_df = properties_df.set_index('smiles')
            elif 'SMILES' in properties_df.columns:
                properties_df = properties_df.set_index('SMILES')
    
    return HESDataset(data_list, properties_df=properties_df, lazy_load=lazy_load)
