"""
HES Model: Hierarchical Embedding Space Model for Molecular Graphs

Architecture:
- Two independent MPNN encoders (GIN) for atomic graph (G) and scaffold graph (Sc)
- 4-layer embeddings: Emb_G_mole, Emb_G_frag, Emb_Sc_mole, Emb_Sc_shape
- Multi-task loss combining alignment, global, and property losses

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool
from typing import Tuple, Dict, Optional
import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "processing"))
from utils.magnet_utils import SimpleAtomFeaturizer


class MPNN_Encoder(nn.Module):
    """
    Graph Isomorphism Network (GIN) encoder for molecular graphs.
    
    Extracts multi-layer embeddings:
    - Emb_mole (molecular-level): global mean pooling
    - Emb_frag (fragment-level): node-wise features from MPN layers
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        """
        Args:
            input_dim: Input node feature dimension
            hidden_dim: Hidden dimension for MPN layers
            embedding_dim: Output embedding dimension
            num_layers: Number of GIN layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        
        # Initial linear layer to project to hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GIN layers: Graph Isomorphism Network
        # Each layer: MLP(x + sum(neighbor embeddings))
        self.gin_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.gin_layers.append(GINConv(mlp, train_eps=True))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Output linear layers
        # For fragment-level embeddings (node embeddings from last GIN layer)
        self.frag_proj = nn.Linear(hidden_dim, embedding_dim)
        
        # For molecular-level embeddings (global pooling)
        self.mole_proj = nn.Linear(hidden_dim, embedding_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the encoder.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge attributes (used for weighted edges if needed)
            batch: Batch assignment vector for handling multiple graphs [num_nodes]
        
        Returns:
            emb_mole: Molecular-level embeddings [batch_size, embedding_dim]
            emb_frag: Fragment-level node embeddings [num_nodes, embedding_dim]
            hidden_states: List of hidden states from each GIN layer (for analysis)
        """
        # Project input features to hidden dimension
        h = self.input_proj(x)
        h = F.relu(h)
        h = self.dropout(h)
        
        # Store hidden states from each layer for analysis
        hidden_states = [h]
        
        # Forward through GIN layers
        for i, (gin_layer, batch_norm) in enumerate(zip(self.gin_layers, self.batch_norms)):
            h = gin_layer(h, edge_index)
            h = batch_norm(h)
            h = F.relu(h)
            h = self.dropout(h)
            hidden_states.append(h)
        
        # Extract embeddings
        # Fragment-level: project final hidden state
        emb_frag = self.frag_proj(h)
        emb_frag = F.normalize(emb_frag, p=2, dim=1)  # L2 normalization
        
        # Molecular-level: global mean pooling
        h_global = global_mean_pool(h, batch)
        emb_mole = self.mole_proj(h_global)
        emb_mole = F.normalize(emb_mole, p=2, dim=1)  # L2 normalization
        
        return emb_mole, emb_frag, hidden_states


class HESModel(nn.Module):
    """
    Hierarchical Embedding Space (HES) Model.
    
    Learns aligned embeddings for:
    - Atomic graph (G): nodes=atoms, edges=bonds
      - Emb_G_mole: molecular (global) level
      - Emb_G_frag: fragment (atomic) level
    - Scaffold graph (Sc): nodes=motifs (or null), edges=adjacency
      - Emb_Sc_mole: molecular (global) level
      - Emb_Sc_shape: shape (motif) level
    
    Additionally:
    - Property prediction heads for 8 molecular properties
    - Embedding projection from motif/shape IDs to embedding space
    """
    
    def __init__(
        self,
        atom_feature_dim: int,
        scaffold_node_feature_dim: int,
        embedding_dim: int = 256,
        num_mpn_layers: int = 3,
        hidden_dim: int = 256,
        num_motif_ids: int = 7370,
        num_shape_ids: int = 346,
        num_properties: int = 8,
        dropout: float = 0.1,
    ):
        """
        Args:
            atom_feature_dim: Dimension of atomic features
            scaffold_node_feature_dim: Dimension of scaffold node features (atom_features + null_flag)
            embedding_dim: Embedding dimension (d=256)
            num_mpn_layers: Number of GIN layers
            hidden_dim: Hidden dimension
            num_motif_ids: Number of motif vocabulary IDs
            num_shape_ids: Number of shape vocabulary IDs
            num_properties: Number of molecular properties (8)
            dropout: Dropout rate
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_properties = num_properties
        
        # ====================================================================
        # MPNN Encoders
        # ====================================================================
        # Encoder for atomic graph (G)
        self.encoder_g = MPNN_Encoder(
            input_dim=atom_feature_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            num_layers=num_mpn_layers,
            dropout=dropout,
        )
        
        # Encoder for scaffold graph (Sc)
        self.encoder_sc = MPNN_Encoder(
            input_dim=scaffold_node_feature_dim,  # atom features + null flag
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            num_layers=num_mpn_layers,
            dropout=dropout,
        )
        
        # ====================================================================
        # EMBEDDING PROJECTIONS (for motif/shape vocabulary)
        # ====================================================================
        # Motif ID embedding: maps motif indices to embedding space
        self.motif_embedding = nn.Embedding(num_motif_ids, embedding_dim, padding_idx=0)
        
        # Shape ID embedding: maps shape indices to embedding space
        self.shape_embedding = nn.Embedding(num_shape_ids, embedding_dim, padding_idx=0)
        
        # ====================================================================
        # PROPERTY PREDICTION HEADS
        # ====================================================================
        # Dense network to predict 8 properties from molecular embeddings
        self.property_head = nn.Sequential(
            nn.Linear(embedding_dim * 4, 512),  # Concat 4 embeddings: Emb_G_mole, Emb_G_frag (mean), Emb_Sc_mole, Emb_Sc_shape (mean)
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_properties),  # Output 8 properties
        )
    
    def forward(
        self,
        x_g: torch.Tensor,
        edge_index_g: torch.Tensor,
        edge_attr_g: Optional[torch.Tensor],
        x_sc: torch.Tensor,
        edge_index_sc: torch.Tensor,
        edge_attr_sc: Optional[torch.Tensor],
        motif_indices: Optional[torch.Tensor] = None,
        shape_indices: Optional[torch.Tensor] = None,
        batch_g: Optional[torch.Tensor] = None,
        batch_sc: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through HES model.
        
        Args:
            x_g: Atomic graph node features [num_atoms, atom_feature_dim]
            edge_index_g: Atomic graph edges [2, num_bonds]
            edge_attr_g: Atomic graph edge attributes (optional)
            x_sc: Scaffold graph node features [num_motifs, scaffold_node_feature_dim]
            edge_index_sc: Scaffold graph edges [2, num_motif_edges]
            edge_attr_sc: Scaffold graph edge attributes (optional)
            motif_indices: Motif vocabulary IDs for nodes [num_motifs] (optional)
            shape_indices: Shape vocabulary IDs for motifs [num_motifs] (optional)
            batch_g: Batch assignment for atomic graph [num_atoms]
            batch_sc: Batch assignment for scaffold graph [num_motifs]
        
        Returns:
            Dictionary containing:
            - emb_g_mole: Molecular embeddings from atomic graph [batch_size, d]
            - emb_g_frag: Fragment embeddings from atomic graph [num_atoms, d]
            - emb_sc_mole: Molecular embeddings from scaffold graph [batch_size, d]
            - emb_sc_shape: Shape embeddings from scaffold graph [num_motifs, d]
            - emb_motif: Motif vocabulary embeddings [num_motifs, d]
            - emb_shape: Shape vocabulary embeddings [num_motifs, d]
            - prop_pred: Property predictions [batch_size, 8]
        """
        # ====================================================================
        # ENCODE ATOMIC GRAPH
        # ====================================================================
        emb_g_mole, emb_g_frag, _ = self.encoder_g(
            x=x_g,
            edge_index=edge_index_g,
            edge_attr=edge_attr_g,
            batch=batch_g,
        )
        
        # ====================================================================
        # ENCODE SCAFFOLD GRAPH
        # ====================================================================
        emb_sc_mole, emb_sc_shape, _ = self.encoder_sc(
            x=x_sc,
            edge_index=edge_index_sc,
            edge_attr=edge_attr_sc,
            batch=batch_sc,
        )
        
        # ====================================================================
        # VOCABULARY EMBEDDINGS
        # ====================================================================
        emb_motif = None
        emb_shape = None
        
        if motif_indices is not None:
            # Project motif IDs to embedding space
            emb_motif = self.motif_embedding(motif_indices)
            emb_motif = F.normalize(emb_motif, p=2, dim=1)
        
        if shape_indices is not None:
            # Project shape IDs to embedding space
            emb_shape = self.shape_embedding(shape_indices)
            emb_shape = F.normalize(emb_shape, p=2, dim=1)
        
        # ====================================================================
        # PROPERTY PREDICTION
        # ====================================================================
        # Aggregate fragment embeddings to molecular level
        emb_g_frag_mean = global_mean_pool(emb_g_frag, batch_g)
        emb_sc_shape_mean = global_mean_pool(emb_sc_shape, batch_sc)
        
        # Concatenate all molecular embeddings
        emb_concat = torch.cat(
            [emb_g_mole, emb_g_frag_mean, emb_sc_mole, emb_sc_shape_mean],
            dim=1
        )
        
        # Predict properties
        prop_pred = self.property_head(emb_concat)
        
        return {
            "emb_g_mole": emb_g_mole,
            "emb_g_frag": emb_g_frag,
            "emb_sc_mole": emb_sc_mole,
            "emb_sc_shape": emb_sc_shape,
            "emb_motif": emb_motif,
            "emb_shape": emb_shape,
            "prop_pred": prop_pred,
        }
    
    def get_embeddings(
        self,
        x_g: torch.Tensor,
        edge_index_g: torch.Tensor,
        x_sc: torch.Tensor,
        edge_index_sc: torch.Tensor,
        batch_g: Optional[torch.Tensor] = None,
        batch_sc: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Get only embeddings (no property prediction).
        Useful for RL stage where we use learned embeddings.
        """
        emb_g_mole, emb_g_frag, _ = self.encoder_g(
            x=x_g,
            edge_index=edge_index_g,
            batch=batch_g,
        )
        
        emb_sc_mole, emb_sc_shape, _ = self.encoder_sc(
            x=x_sc,
            edge_index=edge_index_sc,
            batch=batch_sc,
        )
        
        return {
            "emb_g_mole": emb_g_mole,
            "emb_g_frag": emb_g_frag,
            "emb_sc_mole": emb_sc_mole,
            "emb_sc_shape": emb_sc_shape,
        }
