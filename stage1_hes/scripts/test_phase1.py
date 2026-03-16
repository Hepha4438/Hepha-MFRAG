"""
Test HES Model on sample batch from HESDataset.

This verifies:
1. Model initialization
2. Forward pass on real data
3. Output shapes and properties
4. Loss computation (placeholder)
"""

import sys
from pathlib import Path

# Add parent directories to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "processing"))
sys.path.insert(0, str(project_root / "stage1_hes"))

import torch
from torch_geometric.data import DataLoader
import json

from models.hes_model import HESModel
from training.config import config


def test_model():
    """Test HES model with real data"""
    print("=" * 80)
    print("TESTING HES MODEL - PHASE 1")
    print("=" * 80)
    
    # ========================================================================
    # LOAD HES DATASET
    # ========================================================================
    print("\n[1/4] Loading HESDataset...")
    try:
        from utils.hes_dataset import HESDataset
        
        dataset = HESDataset(
            root=str(config.HES_DATA_DIR),
            mode="train",
            preprocess=False,  # Use lazy loading
        )
        print(f"✓ Loaded {len(dataset)} training samples")
        
        # Get one sample to check structure
        sample = dataset[0]
        print(f"\nSample structure:")
        print(f"  - x_g shape: {sample.x_g.shape}")
        print(f"  - edge_index_g shape: {sample.edge_index_g.shape}")
        print(f"  - x_sc shape: {sample.x_sc.shape}")
        print(f"  - edge_index_sc shape: {sample.edge_index_sc.shape}")
        print(f"  - motif_indices shape: {sample.motif_indices.shape}")
        print(f"  - shape_indices shape: {sample.shape_indices.shape}")
        print(f"  - y shape: {sample.y.shape}")
        
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        print("  (This is expected if dataset hasn't been preprocessed)")
        
        # Create dummy data for testing
        print("\n[FALLBACK] Creating synthetic batch for architecture testing...")
        batch_size = 4
        num_atoms_list = [8, 10, 6, 12]  # variable number of atoms
        num_motifs_list = [3, 4, 2, 5]
        
        # Create mock batch
        class MockData:
            pass
        
        sample = MockData()
        # Concatenate all graphs in batch
        sample.x_g = torch.randn(sum(num_atoms_list), config.ATOM_FEATURE_DIM)
        
        # Edge indices with batch offset
        edge_list_g = []
        atom_offset = 0
        batch_g = []
        for i, n_atoms in enumerate(num_atoms_list):
            # Add some random edges
            n_edges = n_atoms * 2
            edges = torch.randint(0, n_atoms, (2, n_edges))
            edges[0, :] += atom_offset
            edges[1, :] += atom_offset
            edge_list_g.append(edges)
            batch_g.extend([i] * n_atoms)
            atom_offset += n_atoms
        
        sample.edge_index_g = torch.cat(edge_list_g, dim=1)
        sample.batch_g = torch.tensor(batch_g)
        
        # Scaffold graph
        sample.x_sc = torch.randn(sum(num_motifs_list), config.SCAFFOLD_NODE_FEATURE_DIM)
        
        edge_list_sc = []
        motif_offset = 0
        batch_sc = []
        for i, n_motifs in enumerate(num_motifs_list):
            n_edges = max(1, n_motifs - 1)  # At least one edge (tree structure)
            edges = torch.randint(0, n_motifs, (2, n_edges))
            edges[0, :] += motif_offset
            edges[1, :] += motif_offset
            edge_list_sc.append(edges)
            batch_sc.extend([i] * n_motifs)
            motif_offset += n_motifs
        
        sample.edge_index_sc = torch.cat(edge_list_sc, dim=1)
        sample.batch_sc = torch.tensor(batch_sc)
        
        # Vocabulary indices
        sample.motif_indices = torch.randint(0, config.NUM_MOTIF_IDS, (sum(num_motifs_list),))
        sample.shape_indices = torch.randint(0, config.NUM_SHAPE_IDS, (sum(num_motifs_list),))
        
        # Properties
        sample.y = torch.randn(batch_size, config.NUM_PROPERTIES)
    
    # ========================================================================
    # INITIALIZE MODEL
    # ========================================================================
    print("\n[2/4] Initializing HES Model...")
    model = HESModel(
        atom_feature_dim=config.ATOM_FEATURE_DIM,
        scaffold_node_feature_dim=config.SCAFFOLD_NODE_FEATURE_DIM,
        embedding_dim=config.EMBEDDING_DIM,
        num_mpn_layers=config.NUM_MPN_LAYERS,
        hidden_dim=config.MPN_HIDDEN_DIM,
        num_motif_ids=config.NUM_MOTIF_IDS,
        num_shape_ids=config.NUM_SHAPE_IDS,
        num_properties=config.NUM_PROPERTIES,
        dropout=config.DROPOUT,
    )
    
    print(f"✓ Model initialized")
    print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  - Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # ========================================================================
    # TEST FORWARD PASS
    # ========================================================================
    print("\n[3/4] Testing forward pass...")
    model.eval()
    
    with torch.no_grad():
        # Determine batch info
        if hasattr(sample, 'batch_g'):
            batch_g = sample.batch_g
            batch_sc = sample.batch_sc
        else:
            # Single sample
            batch_g = None
            batch_sc = None
        
        # Forward pass
        outputs = model(
            x_g=sample.x_g,
            edge_index_g=sample.edge_index_g,
            edge_attr_g=None,
            x_sc=sample.x_sc,
            edge_index_sc=sample.edge_index_sc,
            edge_attr_sc=None,
            motif_indices=sample.motif_indices,
            shape_indices=sample.shape_indices,
            batch_g=batch_g,
            batch_sc=batch_sc,
        )
        
        print("✓ Forward pass successful")
        print(f"  - emb_g_mole: {outputs['emb_g_mole'].shape}")
        print(f"  - emb_g_frag: {outputs['emb_g_frag'].shape}")
        print(f"  - emb_sc_mole: {outputs['emb_sc_mole'].shape}")
        print(f"  - emb_sc_shape: {outputs['emb_sc_shape'].shape}")
        print(f"  - emb_motif: {outputs['emb_motif'].shape if outputs['emb_motif'] is not None else 'None'}")
        print(f"  - emb_shape: {outputs['emb_shape'].shape if outputs['emb_shape'] is not None else 'None'}")
        print(f"  - prop_pred: {outputs['prop_pred'].shape}")
        
        # Check embedding dimensions
        assert outputs['emb_g_mole'].shape[-1] == config.EMBEDDING_DIM
        assert outputs['emb_g_frag'].shape[-1] == config.EMBEDDING_DIM
        assert outputs['emb_sc_mole'].shape[-1] == config.EMBEDDING_DIM
        assert outputs['emb_sc_shape'].shape[-1] == config.EMBEDDING_DIM
        assert outputs['prop_pred'].shape[-1] == config.NUM_PROPERTIES
        
        print("✓ All embeddings have correct dimensions")
        
        # Check embedding normalization
        g_mole_norms = torch.norm(outputs['emb_g_mole'], dim=1)
        print(f"\n  Embedding L2 norms (should be ~1.0):")
        print(f"    - emb_g_mole: {g_mole_norms.mean():.4f} ± {g_mole_norms.std():.4f}")
        
        g_frag_norms = torch.norm(outputs['emb_g_frag'], dim=1)
        print(f"    - emb_g_frag: {g_frag_norms.mean():.4f} ± {g_frag_norms.std():.4f}")
    
    # ========================================================================
    # SAVE CONFIG
    # ========================================================================
    print("\n[4/4] Saving configuration...")
    config_path = config.CHECKPOINTS_DIR / "config.json"
    config.save_json(config_path)
    print(f"✓ Config saved to {config_path}")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("PHASE 1 TESTING COMPLETE ✓")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Phase 2: Implement 5 loss functions")
    print("  2. Phase 3: Build training loop with checkpointing")
    print("  3. Phase 4: Run full training on 249K molecules")
    print("\n" + "=" * 80)
    
    return model, outputs


if __name__ == "__main__":
    model, outputs = test_model()
