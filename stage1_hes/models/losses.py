"""
Loss Functions for HES Model Training

Components:
1. L_local,G: Alignment loss between atomic fragments and motif vocabulary
2. L_global: Global contrastive loss between atomic and scaffold molecular embeddings
3. L_Sc,G: Alignment loss between scaffold shapes and shape vocabulary
4. L_local,Sc: Local contrastive loss between scaffold and atomic fragment embeddings
5. L_prop: Property prediction loss with MSE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss.
    
    Implements the supervised contrastive loss from:
    "Exploring Simple Siamese Representation Learning via Alignmentand Uniformity on the Hypersphere"
    
    For each sample, we create positive pairs based on property similarity.
    Samples with similar properties (within threshold ε) are considered positive pairs.
    """
    
    def __init__(self, temperature: float = 0.1, epsilon: float = 0.05):
        """
        Args:
            temperature: τ in loss function, controls concentration of distribution
            epsilon: Property similarity threshold for defining positive pairs
        """
        super().__init__()
        self.temperature = temperature
        self.epsilon = epsilon
    
    def forward(
        self,
        embeddings: torch.Tensor,  # [N, d]
        properties: torch.Tensor,  # [N, num_props] or [N, 1] for single property
        mask: Optional[torch.Tensor] = None,  # [N, N] optional manual mask
    ) -> torch.Tensor:
        """
        Compute supervised contrastive loss.
        
        Args:
            embeddings: Batch of embeddings [N, d]
            properties: Properties for computing similarity [N, num_props]
            mask: Optional manual positive pair mask [N, N]
        
        Returns:
            Scalar loss value
        """
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute pairwise similarity matrix
        similarity = torch.mm(embeddings, embeddings.t())  # [N, N]
        similarity = similarity / self.temperature
        
        # Create positive pair mask based on property similarity
        if mask is None:
            if properties.dim() == 1:
                properties = properties.unsqueeze(1)  # [N, 1]
            
            # Compute pairwise property distances
            prop_dists = torch.cdist(properties, properties, p=2)  # [N, N]
            prop_dists = prop_dists.sqrt() + 1e-8
            
            # Create mask: samples with similar properties
            mask = (prop_dists < self.epsilon).float()  # [N, N]
        
        # Remove self-similarity from positive pairs
        mask.fill_diagonal_(0)
        
        # For numerical stability
        logits = similarity
        
        # Compute log-sum-exp trick for stability
        log_probs = F.log_softmax(logits, dim=1)  # [N, N]
        
        # Loss: negative log-likelihood of positive pairs
        # For each sample, average log prob over its positive pairs
        positive_mask = mask > 0.5
        num_samples = embeddings.shape[0]
        
        # Use a list to accumulate losses, then sum
        losses = []
        
        for i in range(num_samples):
            if positive_mask[i].sum() > 0:
                # This sample has positive pairs
                pos_logprobs = log_probs[i, positive_mask[i]]
                losses.append(-pos_logprobs.mean())
        
        # Average over samples that have positive pairs
        if len(losses) > 0:
            loss = torch.stack(losses).mean()
        else:
            # No positive pairs for any sample
            loss = torch.tensor(0.0, device=embeddings.device, dtype=embeddings.dtype, requires_grad=True)
        
        return loss


class AlignmentLoss(nn.Module):
    """
    Alignment Loss for embedding space alignment.
    
    Minimizes L2 distance between pairs of embeddings.
    Used to align fragment embeddings with vocabulary embeddings.
    """
    
    def __init__(self, reduction: str = "mean"):
        """
        Args:
            reduction: "mean", "sum", or "none"
        """
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self,
        emb1: torch.Tensor,  # [N, d]
        emb2: torch.Tensor,  # [N, d]
        weight: Optional[torch.Tensor] = None,  # [N] optional per-sample weight
    ) -> torch.Tensor:
        """
        Compute L2 alignment loss.
        
        Args:
            emb1: First set of embeddings [N, d]
            emb2: Second set of embeddings [N, d]
            weight: Optional per-sample weights [N]
        
        Returns:
            Scalar loss
        """
        # L2 distance
        loss = torch.norm(emb1 - emb2, p=2, dim=1)  # [N]
        
        if weight is not None:
            loss = loss * weight
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class HESLoss(nn.Module):
    """
    Combined HES Loss Function.
    
    L_HES = λ₁*L_local,G + λ₂*L_global + λ₃*L_Sc,G + λ₄*L_local,Sc + γ*L_prop
    
    Where:
    - L_local,G: Alignment between Emb_G_frag and Emb_motif
    - L_global: Contrastive loss between Emb_G_mole and Emb_Sc_mole
    - L_Sc,G: Alignment between Emb_Sc_shape and Emb_shape (vocabulary)
    - L_local,Sc: Contrastive loss between Emb_Sc_mole and Emb_G_mole (global level)
    - L_prop: Property prediction MSE loss
    """
    
    def __init__(
        self,
        lambda_local_g: float = 1.0,
        lambda_global: float = 0.8,
        lambda_sc_g: float = 1.0,
        lambda_local_sc: float = 0.8,
        gamma_prop: float = 0.5,
        contrastive_temperature: float = 0.1,
        property_epsilon: float = 0.05,
    ):
        """
        Args:
            lambda_local_g: Weight for L_local,G
            lambda_global: Weight for L_global
            lambda_sc_g: Weight for L_Sc,G
            lambda_local_sc: Weight for L_local,Sc
            gamma_prop: Weight for L_prop
            contrastive_temperature: Temperature for contrastive loss
            property_epsilon: Property similarity threshold
        """
        super().__init__()
        
        self.lambda_local_g = lambda_local_g
        self.lambda_global = lambda_global
        self.lambda_sc_g = lambda_sc_g
        self.lambda_local_sc = lambda_local_sc
        self.gamma_prop = gamma_prop
        
        self.alignment_loss = AlignmentLoss(reduction="mean")
        self.contrastive_loss = SupervisedContrastiveLoss(
            temperature=contrastive_temperature,
            epsilon=property_epsilon,
        )
        self.property_loss = nn.MSELoss()
    
    def forward(
        self,
        emb_g_mole: torch.Tensor,  # [B, d]
        emb_g_frag: torch.Tensor,  # [N_atoms, d]
        emb_sc_mole: torch.Tensor,  # [B, d]
        emb_sc_shape: torch.Tensor,  # [N_motifs, d]
        emb_motif: torch.Tensor,  # [N_motifs, d]
        emb_shape: torch.Tensor,  # [N_motifs, d]
        prop_pred: torch.Tensor,  # [B, num_props]
        prop_target: torch.Tensor,  # [B, num_props]
        batch_g: torch.Tensor,  # [N_atoms] batch assignment for atoms
        batch_sc: torch.Tensor,  # [N_motifs] batch assignment for motifs
        properties: Optional[torch.Tensor] = None,  # [B, num_props] normalized properties for contrastive loss
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all loss components.
        
        Args:
            emb_g_mole: Molecular embeddings from atomic graph [B, d]
            emb_g_frag: Fragment embeddings from atomic graph [N_atoms, d]
            emb_sc_mole: Molecular embeddings from scaffold graph [B, d]
            emb_sc_shape: Shape embeddings from scaffold graph [N_motifs, d]
            emb_motif: Motif vocabulary embeddings [N_motifs, d]
            emb_shape: Shape vocabulary embeddings [N_motifs, d]
            prop_pred: Predicted properties [B, num_props]
            prop_target: Target properties [B, num_props]
            batch_g: Batch assignment for atoms [N_atoms]
            batch_sc: Batch assignment for motifs [N_motifs]
            properties: Properties for computing contrastive loss [B, num_props] (optional)
        
        Returns:
            Dictionary with loss components and total loss
        """
        losses = {}
        
        # ====================================================================
        # L_local,G: Alignment between Emb_G_frag and Emb_motif
        # ====================================================================
        # Group fragment embeddings by batch
        l_local_g = self._compute_fragment_alignment(
            emb_g_frag, emb_motif, batch_g, batch_sc
        )
        losses["l_local_g"] = l_local_g
        
        # ====================================================================
        # L_global: Contrastive loss between Emb_G_mole and Emb_Sc_mole
        # ====================================================================
        # Combine embeddings and properties for contrastive loss
        combined_mole_emb = torch.cat([emb_g_mole, emb_sc_mole], dim=0)  # [2*B, d]
        
        if properties is not None:
            # Use properties for positive pair definition
            combined_props = torch.cat([properties, properties], dim=0)  # [2*B, num_props]
            l_global = self.contrastive_loss(combined_mole_emb, combined_props)
        else:
            # Fallback: use property predictions
            combined_props = torch.cat([prop_pred, prop_pred], dim=0)
            l_global = self.contrastive_loss(combined_mole_emb, combined_props)
        
        losses["l_global"] = l_global
        
        # ====================================================================
        # L_Sc,G: Alignment between Emb_Sc_shape and Emb_shape
        # ====================================================================
        l_sc_g = self.alignment_loss(emb_sc_shape, emb_shape)
        losses["l_sc_g"] = l_sc_g
        
        # ====================================================================
        # L_local,Sc: Contrastive loss between Emb_Sc_mole and Emb_G_mole
        # ====================================================================
        # Create positive pairs: molecules with similar properties
        if properties is not None:
            combined_all_emb = torch.cat([emb_g_mole, emb_sc_mole], dim=0)  # [2*B, d]
            combined_all_props = torch.cat([properties, properties], dim=0)  # [2*B, num_props]
            l_local_sc = self.contrastive_loss(combined_all_emb, combined_all_props)
        else:
            combined_all_emb = torch.cat([emb_g_mole, emb_sc_mole], dim=0)
            combined_all_props = torch.cat([prop_pred, prop_pred], dim=0)
            l_local_sc = self.contrastive_loss(combined_all_emb, combined_all_props)
        
        losses["l_local_sc"] = l_local_sc
        
        # ====================================================================
        # L_prop: Property prediction loss
        # ====================================================================
        l_prop = self.property_loss(prop_pred, prop_target)
        losses["l_prop"] = l_prop
        
        # ====================================================================
        # L_HES: Weighted combination
        # ====================================================================
        l_hes = (
            self.lambda_local_g * losses["l_local_g"] +
            self.lambda_global * losses["l_global"] +
            self.lambda_sc_g * losses["l_sc_g"] +
            self.lambda_local_sc * losses["l_local_sc"] +
            self.gamma_prop * losses["l_prop"]
        )
        losses["l_hes"] = l_hes
        
        return losses
    
    def _compute_fragment_alignment(
        self,
        emb_g_frag: torch.Tensor,  # [N_atoms, d]
        emb_motif: torch.Tensor,  # [N_motifs, d]
        batch_g: torch.Tensor,  # [N_atoms]
        batch_sc: torch.Tensor,  # [N_motifs]
    ) -> torch.Tensor:
        """
        Compute alignment loss between fragment and motif embeddings.
        
        Since fragments are atoms and motifs are higher-level, we aggregate
        fragment embeddings within each motif's structural context.
        
        Simplified approach: align mean(emb_g_frag per batch) with mean(emb_motif per batch)
        """
        # Aggregate embeddings per batch
        batch_size = batch_g.max().item() + 1
        
        loss = 0
        for b in range(batch_size):
            # Get embeddings for this batch
            frag_mask = batch_g == b
            motif_mask = batch_sc == b
            
            if frag_mask.sum() > 0 and motif_mask.sum() > 0:
                # Compute mean embeddings
                emb_g_frag_mean = emb_g_frag[frag_mask].mean(dim=0, keepdim=True)
                emb_motif_mean = emb_motif[motif_mask].mean(dim=0, keepdim=True)
                
                # Alignment loss
                loss += torch.norm(emb_g_frag_mean - emb_motif_mean, p=2)
        
        loss = loss / batch_size if batch_size > 0 else torch.tensor(0.0, device=emb_g_frag.device)
        
        return loss


# ============================================================================
# LOSS COMPUTATION UTILITIES
# ============================================================================


def normalize_properties(properties: torch.Tensor) -> torch.Tensor:
    """
    Normalize properties to zero mean and unit variance.
    
    Args:
        properties: [B, num_props]
    
    Returns:
        Normalized properties [B, num_props]
    """
    mean = properties.mean(dim=0, keepdim=True)
    std = properties.std(dim=0, keepdim=True) + 1e-8
    return (properties - mean) / std


def compute_property_similarity(prop1: torch.Tensor, prop2: torch.Tensor, threshold: float = 0.05) -> torch.Tensor:
    """
    Compute pairwise property similarity.
    
    Args:
        prop1: [N1, num_props]
        prop2: [N2, num_props]
        threshold: Distance threshold for considering as "similar"
    
    Returns:
        Similarity matrix [N1, N2] with 1 for similar, 0 for dissimilar
    """
    dist = torch.cdist(prop1, prop2, p=2)
    similarity = (dist < threshold).float()
    return similarity
