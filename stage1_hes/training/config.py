"""
Configuration for Stage 1 HES Model Training
"""

import json
from pathlib import Path
from typing import Dict, Any

# ============================================================================
# PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent
STAGE1_ROOT = PROJECT_ROOT / "stage1_hes"
CHECKPOINTS_DIR = STAGE1_ROOT / "checkpoints"
HES_DATA_DIR = PROJECT_ROOT / "processing" / "output"  # Where graphs and properties.csv are
PROPERTIES_CSV = PROJECT_ROOT / "processing" / "output" / "properties.csv"
MOTIF_VOCAB = PROJECT_ROOT / "processing" / "output" / "motif_vocab.pkl"
SHAPE_VOCAB = PROJECT_ROOT / "processing" / "output" / "shape_vocab.pkl"

# Create checkpoint directory if it doesn't exist
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# ARCHITECTURE HYPERPARAMETERS
# ============================================================================
EMBEDDING_DIM = 256
NUM_MPN_LAYERS = 3
MPN_HIDDEN_DIM = 256
DROPOUT = 0.1

# Vocabulary sizes
NUM_MOTIF_IDS = 7370  # Includes ID 0 as padding
NUM_SHAPE_IDS = 346   # Includes ID 0 as padding

# Molecular descriptors for node features
# SimpleAtomFeaturizer from MAGNet returns approximately 13-15 features per atom
ATOM_FEATURE_DIM = 15  # Adjust based on actual SimpleAtomFeaturizer output

# ============================================================================
# NULL SCAFFOLD HANDLING
# ============================================================================
# For molecules without rings (null scaffold), we create a virtual null node
# Node feature = [atom_feature_vec, null_flag]
# null_flag = 1.0 if scaffold is null, 0.0 if scaffold exists
NULL_NODE_FLAG_DIM = 1
SCAFFOLD_NODE_FEATURE_DIM = ATOM_FEATURE_DIM + NULL_NODE_FLAG_DIM

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================
BATCH_SIZE = 128
NUM_EPOCHS = 100
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-5
GRADIENT_CLIP = 1.0

# Data split ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Optimizer settings
OPTIMIZER = "Adam"  # or "AdamW"
SCHEDULER = "CosineAnnealingLR"  # or None for no scheduler
WARMUP_EPOCHS = 5

# ============================================================================
# LOSS FUNCTION WEIGHTS
# ============================================================================
# L_HES = λ₁*L_local,G + λ₂*L_global + λ₃*L_Sc,G + λ₄*L_local,Sc + γ*L_prop
LAMBDA_LOCAL_G = 1.0
LAMBDA_GLOBAL = 0.8
LAMBDA_SC_G = 1.0
LAMBDA_LOCAL_SC = 0.8
GAMMA_PROP = 0.5

# Supervised contrastive loss parameters
CONTRASTIVE_TEMPERATURE = 0.1       # τ in loss function
PROPERTY_THRESHOLD = 0.05           # ε for property similarity threshold

# ============================================================================
# PROPERTY NORMALIZATION
# ============================================================================
# Properties in order: [logP, qed, SAS, docking_parp1, docking_fa7, docking_5ht1b, docking_braf, docking_jak2]
PROPERTIES = ["logP", "qed", "SAS", "docking_parp1", "docking_fa7", "docking_5ht1b", "docking_braf", "docking_jak2"]
NUM_PROPERTIES = len(PROPERTIES)

# ============================================================================
# CHECKPOINTING & EVALUATION
# ============================================================================
CHECKPOINT_INTERVAL = 5  # Save checkpoint every N epochs
EVAL_INTERVAL = 1        # Evaluate every N epochs
PATIENCE = 20            # Early stopping patience (epochs without improvement)
SEED = 42

# ============================================================================
# DEVICE & LOGGING
# ============================================================================
DEVICE = "cuda"  # Auto fallback: CUDA > MPS > CPU (see trainer.py)
LOG_INTERVAL = 100  # Log every N batches
VERBOSE = True


class HESConfig:
    """Configuration container for HES model training"""
    
    def __init__(self, **kwargs):
        """Initialize config with custom overrides"""
        # Set all module-level variables as instance attributes
        for key, value in globals().items():
            if key.isupper() and not key.startswith("_"):
                setattr(self, key, value)
        
        # Apply custom overrides
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith("_") and not callable(v) and k.isupper()
        }
    
    def save_json(self, path: Path):
        """Save config to JSON file"""
        config_dict = {
            "embedding_dim": self.EMBEDDING_DIM,
            "num_mpn_layers": self.NUM_MPN_LAYERS,
            "mpn_hidden_dim": self.MPN_HIDDEN_DIM,
            "dropout": self.DROPOUT,
            "num_motif_ids": self.NUM_MOTIF_IDS,
            "num_shape_ids": self.NUM_SHAPE_IDS,
            "atom_feature_dim": self.ATOM_FEATURE_DIM,
            "scaffold_node_feature_dim": self.SCAFFOLD_NODE_FEATURE_DIM,
            "batch_size": self.BATCH_SIZE,
            "learning_rate": self.LEARNING_RATE,
            "weight_decay": self.WEIGHT_DECAY,
            "loss_weights": {
                "lambda_local_g": self.LAMBDA_LOCAL_G,
                "lambda_global": self.LAMBDA_GLOBAL,
                "lambda_sc_g": self.LAMBDA_SC_G,
                "lambda_local_sc": self.LAMBDA_LOCAL_SC,
                "gamma_prop": self.GAMMA_PROP,
            },
            "properties": self.PROPERTIES,
        }
        with open(path, "w") as f:
            json.dump(config_dict, f, indent=2)


# Default configuration instance
config = HESConfig()
