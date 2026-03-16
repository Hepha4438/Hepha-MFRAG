"""
Evaluation utilities for HES model.

Computes metrics:
- Property prediction MSE per dimension
- Embedding space statistics (norms, cosine similarities)
- Property correlation analysis
"""

import torch
import numpy as np
from typing import Dict, Tuple
import matplotlib.pyplot as plt
from pathlib import Path


class HESEvaluator:
    """Evaluate HES model on test set"""
    
    def __init__(self, model, test_loader, config=None):
        self.model = model
        self.test_loader = test_loader
        self.config = config
        self.device = next(model.parameters()).device
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate model on test set.
        
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        
        all_embeddings = {
            "emb_g_mole": [],
            "emb_g_frag": [],
            "emb_sc_mole": [],
            "emb_sc_shape": [],
        }
        all_prop_preds = []
        all_prop_targets = []
        
        for batch in self.test_loader:
            batch = batch.to(self.device)
            
            outputs = self.model(
                x_g=batch.x_g,
                edge_index_g=batch.edge_index_g,
                edge_attr_g=None,
                x_sc=batch.x_sc,
                edge_index_sc=batch.edge_index_sc,
                edge_attr_sc=None,
                motif_indices=batch.motif_indices,
                shape_indices=batch.shape_indices,
                batch_g=batch.batch_g,
                batch_sc=batch.batch_sc,
            )
            
            # Collect embeddings
            for key in all_embeddings:
                all_embeddings[key].append(outputs[key].cpu().numpy())
            
            all_prop_preds.append(outputs["prop_pred"].cpu().numpy())
            all_prop_targets.append(batch.y.cpu().numpy())
        
        # Concatenate
        for key in all_embeddings:
            all_embeddings[key] = np.concatenate(all_embeddings[key], axis=0)
        
        all_prop_preds = np.concatenate(all_prop_preds, axis=0)
        all_prop_targets = np.concatenate(all_prop_targets, axis=0)
        
        # Compute metrics
        metrics = {
            "prop_mse_total": np.mean((all_prop_preds - all_prop_targets) ** 2),
            "prop_mae_total": np.mean(np.abs(all_prop_preds - all_prop_targets)),
            "num_samples": all_prop_preds.shape[0],
        }
        
        # Per-property MSE
        for i, prop_name in enumerate(self.config.PROPERTIES):
            mse = np.mean((all_prop_preds[:, i] - all_prop_targets[:, i]) ** 2)
            metrics[f"mse_{prop_name}"] = mse
        
        # Embedding norms
        for key, emb in all_embeddings.items():
            norms = np.linalg.norm(emb, axis=1)
            metrics[f"{key}_norm_mean"] = np.mean(norms)
            metrics[f"{key}_norm_std"] = np.std(norms)
        
        return metrics, all_embeddings, all_prop_preds, all_prop_targets
    
    def visualize_embeddings(self, embeddings: Dict[str, np.ndarray], output_dir: Path):
        """
        Visualize embeddings with t-SNE (requires sklearn and umap).
        
        Args:
            embeddings: Dictionary of embedding arrays
            output_dir: Directory to save visualizations
        """
        try:
            from sklearn.manifold import TSNE
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            for key, emb in embeddings.items():
                print(f"Computing t-SNE for {key}...")
                tsne = TSNE(n_components=2, random_state=42, perplexity=30)
                emb_2d = tsne.fit_transform(emb)
                
                plt.figure(figsize=(10, 8))
                plt.scatter(emb_2d[:, 0], emb_2d[:, 1], alpha=0.5, s=20)
                plt.title(f"t-SNE: {key}")
                plt.savefig(output_dir / f"{key}_tsne.png", dpi=150, bbox_inches="tight")
                plt.close()
                
                print(f"✓ Saved {output_dir / f'{key}_tsne.png'}")
        
        except ImportError:
            print("sklearn not available for t-SNE visualization")


def compute_embedding_similarity(emb1: np.ndarray, emb2: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between two embedding matrices.
    
    Args:
        emb1: [N, d]
        emb2: [N, d]
    
    Returns:
        Cosine similarities [N]
    """
    # Normalize
    emb1_norm = emb1 / (np.linalg.norm(emb1, axis=1, keepdims=True) + 1e-8)
    emb2_norm = emb2 / (np.linalg.norm(emb2, axis=1, keepdims=True) + 1e-8)
    
    # Cosine similarity
    sim = np.sum(emb1_norm * emb2_norm, axis=1)
    
    return sim
