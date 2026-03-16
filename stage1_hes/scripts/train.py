#!/usr/bin/env python3
"""
Main training script for HES Model

Usage:
    python train.py --batch-size 32 --epochs 100 --lr 3e-4

This orchestrates:
1. Loading HES dataset
2. Creating model and dataloaders
3. Running training with validation
4. Saving checkpoints and artifacts for Stage 2
"""

import argparse
import sys
from pathlib import Path
import json
from datetime import datetime
import time

# Add parent paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "stage1_hes"))
sys.path.insert(0, str(project_root / "processing"))

import torch
from models.hes_model import HESModel
from training.trainer import HESTrainer
from training.config import config
from data.dataloader import HESDataLoader


def main(args):
    """Main training function"""
    
    print("=" * 80)
    print("HES MODEL - STAGE 1 TRAINING")
    print("=" * 80)
    
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    # Override config with CLI arguments
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.epochs:
        config.NUM_EPOCHS = args.epochs
    if args.lr:
        config.LEARNING_RATE = args.lr
    if args.dropout:
        config.DROPOUT = args.dropout
    
    print(f"\nConfiguration:")
    print(f"  - Batch size: {config.BATCH_SIZE}")
    print(f"  - Epochs: {config.NUM_EPOCHS}")
    print(f"  - Learning rate: {config.LEARNING_RATE}")
    print(f"  - Embedding dim: {config.EMBEDDING_DIM}")
    print(f"  - MPN layers: {config.NUM_MPN_LAYERS}")
    print(f"  - Dropout: {config.DROPOUT}")
    print(f"  - Device: {config.DEVICE}")
    
    # ========================================================================
    # DATA LOADING
    # ========================================================================
    print(f"\n[1/3] Loading data from {config.HES_DATA_DIR}...")
    
    try:
        dataloader = HESDataLoader(
            dataset_root=str(config.HES_DATA_DIR),
            batch_size=config.BATCH_SIZE,
            train_ratio=config.TRAIN_RATIO,
            val_ratio=config.VAL_RATIO,
            test_ratio=config.TEST_RATIO,
            num_workers=0,  # Set to 0 on macOS to avoid issues
            shuffle_train=True,
            seed=config.SEED,
        )
        
        train_loader = dataloader.train_loader
        val_loader = dataloader.val_loader
        test_loader = dataloader.test_loader
        
        print(f"✓ Data loaded successfully")
        
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        print("\nFalling back to synthetic data for testing...")
        
        # Create synthetic loaders for testing
        # This is useful for quick architecture validation
        train_loader = None
        val_loader = None
        test_loader = None
    
    # ========================================================================
    # MODEL INITIALIZATION
    # ========================================================================
    print(f"\n[2/3] Initializing HES model...")
    
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
    
    # ========================================================================
    # TRAINING
    # ========================================================================
    print(f"\n[3/3] Starting training loop...")
    
    if train_loader is not None:
        trainer = HESTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            config=config,
        )
        
        # Run training with timing
        start_time = datetime.now()
        start_timestamp = time.time()
        print(f"\n" + "=" * 80)
        print(f"TRAINING START TIME: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"=" * 80)
        
        trainer.train()
        
        end_time = datetime.now()
        end_timestamp = time.time()
        duration_seconds = end_timestamp - start_timestamp
        duration_minutes = duration_seconds / 60
        duration_hours = duration_minutes / 60
        
        print(f"\n" + "=" * 80)
        print(f"TRAINING END TIME: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"TOTAL TRAINING TIME: {duration_hours:.2f}h ({duration_minutes:.1f}m / {duration_seconds:.0f}s)")
        print(f"=" * 80)
        
        # Save final artifacts
        print(f"\n[FINAL] Saving artifacts for Stage 2...")
        trainer.save_checkpoint(config.NUM_EPOCHS, is_best=False)
        
        print(f"\n" + "=" * 80)
        print("TRAINING COMPLETE ✓")
        print(f"=" * 80)
        print(f"\nCheckpoints and artifacts saved to:")
        print(f"  {config.CHECKPOINTS_DIR}/")
        print(f"\nKey files for Stage 2:")
        print(f"  - best_model.pt (trained weights)")
        print(f"  - config.json (architecture config)")
        print(f"  - scaler.pkl (StandardScaler for property normalization)")
        print(f"  - vocab_info.json (vocabulary sizes)")
        
    else:
        print("✗ Could not initialize training (data loading failed)")
        print("✓ To fix: Ensure HES dataset exists at:")
        print(f"  {config.HES_DATA_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HES Model for Stage 1")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=None, help="Dropout rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/mps/cpu, auto fallback if not available)")
    
    args = parser.parse_args()
    
    main(args)
