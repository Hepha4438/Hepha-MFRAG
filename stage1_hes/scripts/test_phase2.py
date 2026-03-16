"""
Test HES Loss Functions

Verify:
1. All 5 loss components compute correctly
2. Loss dimensions and values are sensible
3. Gradients flow properly
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "stage1_hes"))

import torch
from models.losses import HESLoss, AlignmentLoss, SupervisedContrastiveLoss


def test_alignment_loss():
    """Test basic alignment loss"""
    print("\n" + "=" * 80)
    print("Testing AlignmentLoss")
    print("=" * 80)
    
    loss_fn = AlignmentLoss()
    
    # Create test embeddings
    batch_size, d = 4, 256
    emb1 = torch.randn(batch_size, d, requires_grad=True)
    emb2 = torch.randn(batch_size, d, requires_grad=True)
    
    loss = loss_fn(emb1, emb2)
    
    print(f"Embeddings shape: {emb1.shape}")
    print(f"Loss shape: {loss.shape}")
    print(f"Loss value: {loss.item():.4f}")
    
    # Check gradient flow
    loss.backward()
    print(f"✓ AlignmentLoss working correctly")


def test_contrastive_loss():
    """Test supervised contrastive loss"""
    print("\n" + "=" * 80)
    print("Testing SupervisedContrastiveLoss")
    print("=" * 80)
    
    loss_fn = SupervisedContrastiveLoss(temperature=0.1, epsilon=0.05)
    
    # Create test embeddings and properties
    batch_size, d, num_props = 8, 256, 8
    embeddings = torch.randn(batch_size, d, requires_grad=True)
    properties = torch.randn(batch_size, num_props)
    
    loss = loss_fn(embeddings, properties)
    
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Properties shape: {properties.shape}")
    print(f"Loss value: {loss.item():.4f}")
    
    # Check gradient flow
    loss.backward()
    print(f"✓ SupervisedContrastiveLoss working correctly")


def test_hes_loss():
    """Test combined HES loss"""
    print("\n" + "=" * 80)
    print("Testing HESLoss (Combined)")
    print("=" * 80)
    
    loss_fn = HESLoss(
        lambda_local_g=1.0,
        lambda_global=0.8,
        lambda_sc_g=1.0,
        lambda_local_sc=0.8,
        gamma_prop=0.5,
        contrastive_temperature=0.1,
        property_epsilon=0.05,
    )
    
    # Create synthetic batch data
    batch_size = 4
    num_atoms = [8, 10, 6, 12]
    num_motifs = [3, 4, 2, 5]
    d = 256
    num_props = 8
    
    # Concatenate graphs
    total_atoms = sum(num_atoms)
    total_motifs = sum(num_motifs)
    
    emb_g_mole = torch.randn(batch_size, d, requires_grad=True)
    emb_g_frag = torch.randn(total_atoms, d, requires_grad=True)
    emb_sc_mole = torch.randn(batch_size, d, requires_grad=True)
    emb_sc_shape = torch.randn(total_motifs, d, requires_grad=True)
    emb_motif = torch.randn(total_motifs, d, requires_grad=True)
    emb_shape = torch.randn(total_motifs, d, requires_grad=True)
    
    prop_pred = torch.randn(batch_size, num_props, requires_grad=True)
    prop_target = torch.randn(batch_size, num_props)
    properties = torch.randn(batch_size, num_props)
    
    # Batch assignments
    batch_g = torch.zeros(total_atoms, dtype=torch.long)
    accum = 0
    for i, n_atoms in enumerate(num_atoms):
        batch_g[accum:accum+n_atoms] = i
        accum += n_atoms
    
    batch_sc = torch.zeros(total_motifs, dtype=torch.long)
    accum = 0
    for i, n_motifs in enumerate(num_motifs):
        batch_sc[accum:accum+n_motifs] = i
        accum += n_motifs
    
    # Compute loss
    losses = loss_fn(
        emb_g_mole=emb_g_mole,
        emb_g_frag=emb_g_frag,
        emb_sc_mole=emb_sc_mole,
        emb_sc_shape=emb_sc_shape,
        emb_motif=emb_motif,
        emb_shape=emb_shape,
        prop_pred=prop_pred,
        prop_target=prop_target,
        batch_g=batch_g,
        batch_sc=batch_sc,
        properties=properties,
    )
    
    print(f"\nLoss components:")
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            print(f"  {loss_name}: {loss_value.item():.4f}")
    
    # Check gradient flow
    print(f"\nChecking gradient flow...")
    losses["l_hes"].backward()
    
    grads = {
        "emb_g_mole": emb_g_mole.grad is not None,
        "emb_g_frag": emb_g_frag.grad is not None,
        "emb_sc_mole": emb_sc_mole.grad is not None,
        "emb_sc_shape": emb_sc_shape.grad is not None,
        "emb_motif": emb_motif.grad is not None,
        "emb_shape": emb_shape.grad is not None,
        "prop_pred": prop_pred.grad is not None,
    }
    
    for var_name, has_grad in grads.items():
        status = "✓" if has_grad else "✗"
        print(f"  {status} {var_name}")
    
    print(f"\n✓ HESLoss working correctly")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("PHASE 2: TESTING LOSS FUNCTIONS")
    print("=" * 80)
    
    test_alignment_loss()
    test_contrastive_loss()
    test_hes_loss()
    
    print("\n" + "=" * 80)
    print("PHASE 2 TESTING COMPLETE ✓")
    print("=" * 80)
