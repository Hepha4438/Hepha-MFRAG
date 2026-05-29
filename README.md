# Hepha-MFRAG
*(Harmonized Embedding Space + Motif-based Fragment RL)*

**Hepha-MFRAG** is a groundbreaking Artificial Intelligence (AI) system in the field of **De Novo Drug Design**. By combining the power of *Graph Neural Networks (GNNs)* to embed molecular topologies and *Reinforcement Learning (RL)* for multi-objective optimization, the system autonomously generates novel molecules (ligands) that satisfy three core objectives: 
1) High binding affinity to a target protein pocket (PARP1, JAK2, BRAF, 5HT1B, FA7).
2) Optimized QED (Quantitative Estimate of Drug-likeness).
3) Optimized SA (Synthetic Accessibility).

---

## Repository Structure

The project is highly modularized into distinct subsystems and pipeline stages:

```text
Hepha-MFRAG/
├── data/                       # Static data (Protein pocket coordinates, Datasets)
│   ├── docking_grids/          # 3D spatial coordinates of target proteins (5ht1b, braf, fa7, jak2, parp1)
│   └── smiles/                 # SMILES datasets (including starting_scaffolds.smi)
├── processing/                 # Stage 0: Preprocessing, decomposition, and Vocab creation
│   ├── 01_compute_properties.py
│   ├── 02_build_motif_vocab.py
│   ├── 03_build_shape_vocab.py
│   ├── 04_build_graphs.py
│   ├── 05_build_shape_to_motifs.py
│   └── run_pipeline.py         # Automates the entire preprocessing pipeline
├── stage1_hes/                 # Stage 1: Representation Learning (Harmonized Embedding Space)
│   ├── checkpoints/            # Pre-trained weights for the HES model and Property Scaler (scaler.pkl)
│   ├── training/               # Training loop for the dual GNN encoders (Encoder_G and Encoder_Sc)
│   └── models/                 # Neural network architectures for HES
├── stage2_rl/                  # Stage 2: RL Optimization via Soft Actor-Critic (SAC)
│   ├── environment/            # MDP definition (molecule_env.py) managing Curriculum Scaffolds & Motif Merging
│   ├── models/                 # Actor/Critic architecture and Autoregressive Masking mechanism
│   ├── training/               # Multi-objective Reward computation (docking, QED, SA) (rewards.py)
│   ├── train.py                # Standard training loop script
│   └── evaluate.py             # Inference and benchmarking after training
└── root scripts                # Auxiliary root scripts (download_zinc_kaggle.py, train_surrogates.py, final_evaluation.py)
```

---

## Installation & Usage

**Environment Setup:** The system requires a `conda` environment with essential chemistry (`rdkit`) and AI libraries (`pytorch`, `torch-geometric`, `gym`) installed. The recommended environment name is `baselines`.

### 1. Data Preprocessing & Vocabulary Construction (Stage 0)

This process analyzes the ZINC250k molecular dataset: extracting Scaffolds/Motifs, building the topology vocabulary, and fitting the Property Scaler to standardize RL reward curves.

```bash
# Download the datasets (if not already downloaded)
python download_zinc250k.py
python download_zinc_kaggle.py

# Run the complete preprocessing pipeline
python processing/run_pipeline.py
```

*To individually update the shape-to-motifs mapping dictionary, run:*
`python processing/05_build_shape_to_motifs.py`

### 2. Reinforcement Learning Training (Stage 2)

The core Agent training loop. The `MoleculeEnv` is equipped with a **Curriculum Learning** algorithm (gradually increasing allowed starting scaffolds) coupled with an **Autoregressive SAC** (Soft Actor-Critic with valency-aware masking).

```bash
# Start Agent Training with a specified warmup schedule
python stage2_rl/train.py
```

### 3. Evaluation 

Generate novel molecules fine-tuned for a specific Target Protein and log the benchmarks to `evaluation_results.csv`.

```bash
# Inference and explicit evaluation against the PARP1 target
python stage2_rl/evaluate.py --target parp1 --num-samples 1000

# Global evaluation across the entire pipeline
python final_evaluation.py
```

---

## Key Technical Features

*   **Autoregressive Action Selection**: The RL Agent constructs molecules via a rigorous 3-step hierarchical logic: Select Attachment Point $\rightarrow$ Select Motif Shape $\rightarrow$ Select Motif Attachment Node to seamlessly merge graphs.
*   **Valency Masking Validation**: RDKit valency limits are natively integrated into the categorical Logit Masking (Masking invalid actions with $-1e^9$). The agent mathematically cannot generate chemically invalid bonds.
*   **Curriculum Scaffold Expansion**: Initiates training with simple topologies (e.g., $5$ basic benzene-like scaffolds) and progressively unlocks over $200$ advanced scaffolds as the agent stabilizes its Reward slope.
*   **Weighted Multi-objective Reward**: An integrated reward topology that uniformly balances StandardScaler values of $\Delta Docking$ (via Surrogates), $\Delta QED$, and $\Delta SA$.
