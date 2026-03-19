"""
Training Loop for HES Model

Handles:
- Training and validation loops
- Checkpointing with model artifacts
- Early stopping
- Logging and metrics
"""

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import json
from pathlib import Path
from typing import Dict, Tuple, Optional
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import sys
from tqdm import tqdm

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "stage1_hes"))

from models.hes_model import HESModel
from models.losses import HESLoss, normalize_properties
from training.config import config


class HESTrainer:
    """
    Trainer class for HES model with full training loop, validation, and checkpointing.
    """
    
    def __init__(
        self,
        model: HESModel,
        train_loader,
        val_loader,
        test_loader=None,
        config=None,
    ):
        """
        Args:
            model: HESModel instance
            train_loader: PyG DataLoader for training
            val_loader: PyG DataLoader for validation
            test_loader: PyG DataLoader for testing (optional)
            config: Configuration object with hyperparameters
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config or globals()['config']
        
        # Device - priority: CUDA > CPU > MPS
        # MPS has overhead for small batches, so only use if CUDA unavailable
        if torch.cuda.is_available():
            device_str = "cuda"
        else:
            device_str = "cpu"  # CPU faster than MPS for small/medium workloads
        self.device = torch.device(device_str)
        self.model = self.model.to(self.device)
        
        # Optimizer
        if self.config.OPTIMIZER == "Adam":
            self.optimizer = Adam(
                self.model.parameters(),
                lr=self.config.LEARNING_RATE,
                weight_decay=self.config.WEIGHT_DECAY,
            )
        elif self.config.OPTIMIZER == "AdamW":
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=self.config.LEARNING_RATE,
                weight_decay=self.config.WEIGHT_DECAY,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.OPTIMIZER}")
        
        # Scheduler
        if self.config.SCHEDULER == "CosineAnnealingLR":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.NUM_EPOCHS,
                eta_min=1e-6,
            )
        else:
            self.scheduler = None
        
        # Loss function
        self.loss_fn = HESLoss(
            lambda_local_g=self.config.LAMBDA_LOCAL_G,
            lambda_global=self.config.LAMBDA_GLOBAL,
            lambda_sc_g=self.config.LAMBDA_SC_G,
            lambda_local_sc=self.config.LAMBDA_LOCAL_SC,
            gamma_prop=self.config.GAMMA_PROP,
            contrastive_temperature=self.config.CONTRASTIVE_TEMPERATURE,
            property_epsilon=self.config.PROPERTY_THRESHOLD,
        )
        
        # Track metrics
        self.train_metrics = {
            "epoch": [],
            "loss": [],
            "l_local_g": [],
            "l_global": [],
            "l_sc_g": [],
            "l_local_sc": [],
            "l_prop": [],
        }
        self.val_metrics = {
            "epoch": [],
            "loss": [],
            "prop_mse": [],
        }
        
        # Checkpointing
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.patience_counter = 0
        
        # Property scaler (for Stage 2)
        self.property_scaler = StandardScaler()
        self.scaler_fitted = False
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Epoch number
        
        Returns:
            Dictionary of metrics for this epoch
        """
        self.model.train()
        
        total_loss = 0
        losses_accum = {
            "l_local_g": 0,
            "l_global": 0,
            "l_sc_g": 0,
            "l_local_sc": 0,
            "l_prop": 0,
        }
        
        num_batches = len(self.train_loader)
        
        with tqdm(self.train_loader, desc=f"Epoch {epoch} [TRAIN]", disable=not self.config.VERBOSE) as pbar:
            for batch_idx, batch in enumerate(pbar):
                # Move to device
                batch = batch.to(self.device)
                
                batch_g = batch.x_g_batch if hasattr(batch, 'x_g_batch') else batch.batch
                batch_sc = batch.x_sc_batch if hasattr(batch, 'x_sc_batch') else batch.batch

                # Forward pass
                outputs = self.model(
                    x_g=batch.x_g,
                    edge_index_g=batch.edge_index_g,
                    edge_attr_g=None,
                    x_sc=batch.x_sc,
                    edge_index_sc=batch.edge_index_sc,
                    edge_attr_sc=None,
                    motif_indices=batch.motif_indices,
                    shape_indices=batch.shape_indices,
                    batch_g=batch_g,
                    batch_sc=batch_sc,
                )
                
                # Normalize properties using global StandardScaler
                props_normalized = torch.from_numpy(
                    self.property_scaler.transform(batch.y.cpu().numpy())
                ).to(batch.y.device).float()
                
                # Compute loss
                losses = self.loss_fn(
                    emb_g_mole=outputs["emb_g_mole"],
                    emb_g_frag=outputs["emb_g_frag"],
                    emb_sc_mole=outputs["emb_sc_mole"],
                    emb_sc_shape=outputs["emb_sc_shape"],
                    emb_motif=outputs["emb_motif"],
                    emb_shape=outputs["emb_shape"],
                    prop_pred=outputs["prop_pred"],
                    prop_target=batch.y,
                    batch_g=batch_g,
                    batch_sc=batch_sc,
                    properties=props_normalized,
                )
                
                loss = losses["l_hes"]
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.GRADIENT_CLIP,
                )
                
                self.optimizer.step()
                
                # Accumulate metrics
                total_loss += loss.item()
                for loss_name in ["l_local_g", "l_global", "l_sc_g", "l_local_sc", "l_prop"]:
                    losses_accum[loss_name] += losses[loss_name].item()
                
                # Update progress bar
                if batch_idx % self.config.LOG_INTERVAL == 0:
                    avg_loss = total_loss / (batch_idx + 1)
                    pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
        
        # Average metrics over epoch
        metrics = {
            "loss": total_loss / num_batches,
        }
        for loss_name in losses_accum:
            metrics[loss_name] = losses_accum[loss_name] / num_batches
        
        return metrics
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validate on val set.
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        total_loss = 0
        all_prop_preds = []
        all_prop_targets = []
        
        with tqdm(self.val_loader, desc="[VALIDATION]", disable=not self.config.VERBOSE) as pbar:
            for batch in pbar:
                batch = batch.to(self.device)
                
                batch_g = batch.x_g_batch if hasattr(batch, 'x_g_batch') else batch.batch
                batch_sc = batch.x_sc_batch if hasattr(batch, 'x_sc_batch') else batch.batch
                
                # Forward pass
                outputs = self.model(
                    x_g=batch.x_g,
                    edge_index_g=batch.edge_index_g,
                    edge_attr_g=None,
                    x_sc=batch.x_sc,
                    edge_index_sc=batch.edge_index_sc,
                    edge_attr_sc=None,
                    motif_indices=batch.motif_indices,
                    shape_indices=batch.shape_indices,
                    batch_g=batch_g,
                    batch_sc=batch_sc,
                )
                
                # Normalize properties using global StandardScaler
                props_normalized = torch.from_numpy(
                    self.property_scaler.transform(batch.y.cpu().numpy())
                ).to(batch.y.device).float()
                
                # Compute loss
                losses = self.loss_fn(
                    emb_g_mole=outputs["emb_g_mole"],
                    emb_g_frag=outputs["emb_g_frag"],
                    emb_sc_mole=outputs["emb_sc_mole"],
                    emb_sc_shape=outputs["emb_sc_shape"],
                    emb_motif=outputs["emb_motif"],
                    emb_shape=outputs["emb_shape"],
                    prop_pred=outputs["prop_pred"],
                    prop_target=batch.y,
                    batch_g=batch_g,
                    batch_sc=batch_sc,
                    properties=props_normalized,
                )
                
                loss = losses["l_hes"]
                total_loss += loss.item()
                
                # Accumulate for property metrics
                all_prop_preds.append(outputs["prop_pred"].cpu().numpy())
                all_prop_targets.append(batch.y.cpu().numpy())
                
                pbar.set_postfix({"loss": f"{total_loss / (len(pbar) + 1):.4f}"})
        
        # Compute property MSE
        all_prop_preds = np.concatenate(all_prop_preds, axis=0)
        all_prop_targets = np.concatenate(all_prop_targets, axis=0)
        prop_mse = np.mean((all_prop_preds - all_prop_targets) ** 2)
        
        metrics = {
            "loss": total_loss / len(self.val_loader),
            "prop_mse": prop_mse,
        }
        
        return metrics
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save model checkpoint with artifacts for Stage 2.
        
        Args:
            epoch: Current epoch
            is_best: Whether this is the best checkpoint
        """
        checkpoint_dir = self.config.CHECKPOINTS_DIR
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        model_path = checkpoint_dir / f"model_epoch{epoch}.pt"
        torch.save(self.model.state_dict(), model_path)
        
        # Save the best model separately
        if is_best:
            best_model_path = checkpoint_dir / "best_model.pt"
            torch.save(self.model.state_dict(), best_model_path)
            print(f"✓ Best model saved to {best_model_path}")
        
        # Save config
        config_path = checkpoint_dir / "config.json"
        self.config.save_json(config_path)
        
        # Save property scaler (for Stage 2 RL)
        if self.scaler_fitted:
            scaler_path = checkpoint_dir / "scaler.pkl"
            with open(scaler_path, "wb") as f:
                pickle.dump(self.property_scaler, f)
            print(f"✓ Scaler saved to {scaler_path}")
        
        # Save vocabulary info
        vocab_info = {
            "num_motif_ids": self.config.NUM_MOTIF_IDS,
            "num_shape_ids": self.config.NUM_SHAPE_IDS,
            "embedding_dim": self.config.EMBEDDING_DIM,
            "properties": self.config.PROPERTIES,
        }
        vocab_path = checkpoint_dir / "vocab_info.json"
        with open(vocab_path, "w") as f:
            json.dump(vocab_info, f, indent=2)
        print(f"✓ Vocab info saved to {vocab_path}")
    
    def fit_property_scaler(self):
        """
        Fit StandardScaler on training data properties.
        Must be called before training to prepare scaler for Stage 2.
        """
        print("[Trainer] Fitting property scaler on training data...")
        all_props = []
        
        try:
            with torch.no_grad():
                for idx, batch in enumerate(self.train_loader):
                    if hasattr(batch, 'y'):
                        all_props.append(batch.y.numpy() if isinstance(batch.y, torch.Tensor) else batch.y)
            
            if all_props:
                all_props = np.concatenate(all_props, axis=0)
                self.property_scaler.fit(all_props)
                self.scaler_fitted = True
                print(f"✓ Property scaler fitted")
            else:
                print(f"[WARN] Could not fit scaler, no properties available")
                self.scaler_fitted = False
        except Exception as e:
            print(f"[WARN] Error fitting property scaler: {e}")
            print(f"       Continuing without scaler fitting...")
            self.scaler_fitted = False
    
    def train(self):
        """
        Full training loop with validation and checkpointing.
        """
        print("=" * 80)
        print("STARTING TRAINING FOR HES MODEL")
        print("=" * 80)
        print(f"Device: {self.device}")
        print(f"Model params: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Total epochs: {self.config.NUM_EPOCHS}")
        print(f"Batch size: {self.config.BATCH_SIZE}")
        print("=" * 80)
        
        # Fit property scaler first
        self.fit_property_scaler()
        
        # Training loop
        for epoch in range(1, self.config.NUM_EPOCHS + 1):
            # Train
            train_metrics = self.train_epoch(epoch)
            self.train_metrics["epoch"].append(epoch)
            for key, value in train_metrics.items():
                self.train_metrics[key].append(value)
            
            # Validate
            if epoch % self.config.EVAL_INTERVAL == 0:
                val_metrics = self.validate()
                self.val_metrics["epoch"].append(epoch)
                for key, value in val_metrics.items():
                    self.val_metrics[key].append(value)
                
                print(f"\n[Epoch {epoch}]")
                print(f"  Train Loss: {train_metrics['loss']:.4f}")
                print(f"  Val Loss: {val_metrics['loss']:.4f}")
                print(f"  Val Prop MSE: {val_metrics['prop_mse']:.4f}")
                
                # Early stopping
                if val_metrics["loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["loss"]
                    self.best_epoch = epoch
                    self.patience_counter = 0
                    self.save_checkpoint(epoch, is_best=True)
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.config.PATIENCE:
                        print(f"\n✓ Early stopping at epoch {epoch} (no improvement for {self.config.PATIENCE} epochs)")
                        break
            else:
                print(f"[Epoch {epoch}] Train Loss: {train_metrics['loss']:.4f}")
            
            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Regular checkpoint
            if epoch % self.config.CHECKPOINT_INTERVAL == 0:
                self.save_checkpoint(epoch, is_best=False)
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print(f"Best validation loss: {self.best_val_loss:.4f} at epoch {self.best_epoch}")
        print("=" * 80)
