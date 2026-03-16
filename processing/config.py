from pathlib import Path
from dataclasses import dataclass
import multiprocessing

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSING_DIR = Path(__file__).parent
OUTPUT_DIR = PROCESSING_DIR / "output"
GRAPHS_DIR = OUTPUT_DIR / "graphs"
VOCAB_DIR = OUTPUT_DIR / "vocabularies"

# Create output directories
for d in [GRAPHS_DIR, VOCAB_DIR]:
    d.mkdir(parents=True, exist_ok=True)

@dataclass
class ProcessingConfig:
    """Configuration for preprocessing pipeline"""
    
    # Input/Output paths
    input_smiles: Path = DATA_DIR / "smiles/zinc250k/zinc250k.smi"
    input_csv: Path = DATA_DIR / "smiles/zinc250k/zinc250k.csv"
    output_properties_csv: Path = OUTPUT_DIR / "properties.csv"
    motif_vocab_path: Path = VOCAB_DIR / "motif_vocab.pkl"
    shape_vocab_path: Path = VOCAB_DIR / "shape_vocab.pkl"
    graphs_dir: Path = GRAPHS_DIR
    
    # Processing options
    num_processes: int = 0  # Will be auto-detected if 0
    batch_size: int = 128
    skip_existing_graphs: bool = True
    output_dir: Path = OUTPUT_DIR  # For backward compatibility
    
    # Vocabulary config
    max_motif_molecules: int = None  # Max molecules to process for motif vocab (None = all)
    max_shape_molecules: int = None  # Max molecules to process for shape vocab (None = all)
    
    # MAGNet config
    max_mol_size: int = 50
    max_shape_size: int = 15
    max_num_shapes: int = 100
    
    # Docking config
    docking_proteins: list = None  # parp1, fa7, 5ht1b, braf, jak2
    use_autodock_vina: bool = False  # Set True if Vina installed
    
    def __post_init__(self):
        if self.docking_proteins is None:
            self.docking_proteins = ["parp1", "fa7", "5ht1b", "braf", "jak2"]
        if self.num_processes == 0:
            self.num_processes = multiprocessing.cpu_count()

# Default config
DEFAULT_CONFIG = ProcessingConfig()
