"""
Stage 2 RL Training Configuration
Soft Actor-Critic for Molecular Generation with HES Alignment
"""
import torch
from pathlib import Path

# ===== Paths =====
PROJECT_ROOT = Path(__file__).parent.parent.parent
STAGE1_CHECKPOINT = PROJECT_ROOT / "stage1_hes/checkpoints/best_model.pt"
STAGE1_SCALER = PROJECT_ROOT / "stage1_hes/checkpoints/scaler.pkl"
DATA_OUTPUT_DIR = PROJECT_ROOT / "processing/output"
GRAPHS_DIR = DATA_OUTPUT_DIR / "graphs"
PROPERTIES_CSV = DATA_OUTPUT_DIR / "properties.csv"
MOTIF_VOCAB_PATH = DATA_OUTPUT_DIR / "vocabularies/motif_vocab.pkl"
SHAPE_VOCAB_PATH = DATA_OUTPUT_DIR / "vocabularies/shape_vocab.pkl"

# Stage 2 output directory
STAGE2_OUTPUT = PROJECT_ROOT / "stage2_rl"
STAGE2_OUTPUT.mkdir(parents=True, exist_ok=True)

# ===== Data Configuration =====
BATCH_SIZE = 32
NUM_WORKERS = 4
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# Data split (from preprocessing)
TRAIN_INDICES = list(range(0, 199564))          # First 199,564
VAL_INDICES = list(range(199564, 224509))       # Next 24,945
TEST_INDICES = list(range(224509, 249455))      # Last 24,946

# ===== Molecular Properties =====
NUM_PROPERTIES = 8  # logP, QED, SA, + 5 docking scores
PROPERTY_NAMES = ["logP", "QED", "SA", "prop1", "prop2", "prop3", "prop4", "prop5"]

# ===== Atom Configuration =====
# Supported atom types: C, N, O, S, P, F, Cl, Br, I (9 atom types)
# Atomic numbers: 6, 7, 8, 16, 15, 9, 17, 35, 53
ATOM_TYPES = {
    'C': 6,   'N': 7,   'O': 8,   'S': 16,  'P': 15,
    'F': 9,   'Cl': 17, 'Br': 35, 'I': 53
}
ATOM_TYPES_REVERSE = {v: k for k, v in ATOM_TYPES.items()}
NUM_ATOM_TYPES = len(ATOM_TYPES)  # 9

# Bond types: 1=single, 2=double, 3=triple, 4=aromatic
BOND_TYPES = {1: 'single', 2: 'double', 3: 'triple', 4: 'aromatic'}
NUM_BOND_TYPES = 4

# ===== Vocabulary Configuration =====
NUM_MOTIFS = 7370      # From preprocessing
NUM_SHAPES = 346       # From preprocessing

# ===== HES Model Configuration =====
HES_FEATURE_DIM = 256  # Output dim of encoder_g and encoder_sc
HES_ENCODING_DIM = 512  # Concatenated encoding: encoder_g(x_g) + encoder_sc(x_sc) = 512
HES_OUTPUT_IS_NORMALIZED = False  # True if HES prop_pred is already StandardScaler-normalized
TARGET_PROPERTIES_ARE_NORMALIZED = True  # False if Y* is provided in raw space and must be transformed by SS

# ===== Environment Configuration =====
MAX_ATOMS_PER_MOLECULE = 50          # Max atoms in generated molecule
MAX_MOTIFS_PER_EPISODE = 8           # Max motifs to add per episode
ACTION_VECTOR_SIZE = 3               # a1 (attachment), a2 (shape), a3 (orientation)

# ===== SAC Hyperparameters =====
# Actor-Critic architecture
ACTOR_HIDDEN_DIM = 256
CRITIC_HIDDEN_DIM = 256
ACTOR_LEARNING_RATE = 3e-4
CRITIC_LEARNING_RATE = 3e-4
ALPHA_LEARNING_RATE = 3e-4

# SAC training
TARGET_UPDATE_INTERVAL = 1          # Update target every N steps
TAU = 0.005                         # Soft update coefficient (polyak averaging)
GAMMA = 0.99                        # Discount factor
ALPHA = 0.2                         # Temperature for entropy regularization (learnable)

# Replay buffer
REPLAY_BUFFER_SIZE = 100000
BATCH_SIZE_SAC = 128
LEARN_STARTS = 1000   # Start SAC training after this many env steps

# ===== Training Configuration =====
NUM_EPISODES = 10000
MAX_STEPS_PER_EPISODE = 20      # Usually ends earlier due to STOP action
NUM_UPDATES_PER_STEP = 1        # SAC updates per environment step

# Evaluation
EVAL_INTERVAL = 250              # Evaluate every N episodes
NUM_EVAL_EPISODES = 20            # Number of episodes for evaluation
SAVE_INTERVAL = 500              # Save checkpoint every N episodes

# ===== Reward Configuration =====
# Potential-based reward shaping
REWARD_SCALE = 1.0

# Property reward weights
W_QED = 0.3         # QED contribution to property reward
W_SA = 0.3          # SA (Synthetic Accessibility) contribution
W_DOCKING = 0.4     # Average docking scores

# Phase A2 / Terminal combined explicit rewards
R_QED_WEIGHT = 0.3   # Explicit target weight for QED calculation
R_SA_WEIGHT = 0.3    # Explicit target weight for SA calculation

# Termination rewards
TERMINAL_VALID_MOLECULE_BONUS = 1.0
TERMINAL_INVALID_MOLECULE_PENALTY = -1.0

# ===== Logging =====
WANDB_PROJECT = "Hepha-MFRAG"
WANDB_RUN_NAME = "stage2_rl_sac"
LOG_INTERVAL = 25
CHECKPOINT_DIR = STAGE2_OUTPUT / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# ===== Random Seed =====
SEED = 42
