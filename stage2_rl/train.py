#!/usr/bin/env python
"""
Stage 2 Training: Molecular Generation with SAC and HES Alignment

Main training loop that:
1. Loads Stage 1 HES model and preprocessed data
2. Initializes SAC agent and environment
3. Runs training episodes
4. Saves checkpoints and logs metrics
"""
import sys
from pathlib import Path
import argparse
import numpy as np
import torch
import pickle
import time
from tqdm import tqdm
from collections import defaultdict
import warnings
from rdkit import RDLogger

# Disable warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
RDLogger.DisableLog('rdApp.*')

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "stage1_hes"))
sys.path.insert(0, str(PROJECT_ROOT / "processing"))

from stage2_rl.training.config import *
from stage2_rl.environment.molecule_env import MoleculeEnv
from stage2_rl.models.sac_agent import SACAgent
from stage2_rl.training.rewards import RewardComputer, ReplayBuffer
from stage1_hes.models.hes_model import HESModel


def load_stage1_components():
    """Load trained HES model and preprocessors from Stage 1."""
    print("\n" + "="*80)
    print("LOADING STAGE 1 COMPONENTS")
    print("="*80)
    
    # Load property scaler
    print("\n[1/3] Loading property scaler...")
    with open(STAGE1_SCALER, 'rb') as f:
        property_scaler = pickle.load(f)
    print(f"  ✓ Loaded scaler from {STAGE1_SCALER}")
    
    # Load vocabularies
    print("\n[2/3] Loading vocabularies...")
    with open(MOTIF_VOCAB_PATH, 'rb') as f:
        motif_vocab = pickle.load(f)
    print(f"  ✓ Loaded {len(motif_vocab)} motifs")
    
    with open(SHAPE_VOCAB_PATH, 'rb') as f:
        shape_vocab = pickle.load(f)
    print(f"  ✓ Loaded {len(shape_vocab)} shapes")
    
    # Load HES model
    print("\n[3/3] Loading HES model...")
    if STAGE1_CHECKPOINT.exists():
        print(f"  ✓ Checkpoint exists at {STAGE1_CHECKPOINT}")
        hes_model = HESModel(
            atom_feature_dim=15,
            scaffold_node_feature_dim=16,
            embedding_dim=256,
            num_mpn_layers=3,
            hidden_dim=256,
            num_motif_ids=NUM_MOTIFS,
            num_shape_ids=NUM_SHAPES,
            num_properties=NUM_PROPERTIES,
            dropout=0.1,
        )
        state_dict = torch.load(STAGE1_CHECKPOINT, map_location=DEVICE)
        hes_model.load_state_dict(state_dict, strict=False)
        hes_model = hes_model.to(DEVICE)
        hes_model.eval()
        print("  ✓ HES model loaded and set to eval mode")
    else:
        print(f"  [WARNING] Checkpoint not found at {STAGE1_CHECKPOINT}")
        print(f"           Stage 1 training should be run first!")
        hes_model = None
    
    print("\n" + "-"*80)
    return hes_model, motif_vocab, shape_vocab, property_scaler


def initialize_training(hes_model, motif_vocab, shape_vocab, property_scaler):
    """Initialize environment, agent, and reward computer."""
    print("\n" + "="*80)
    print("INITIALIZING STAGE 2 TRAINING")
    print("="*80)
    
    # Initialize reward computer first
    print("\n[1/4] Initializing reward computer...")
    
    # Define ideal raw chemical properties (Alphabetical order from Stage 1):
    # [SAS, docking_5ht1b, docking_braf, docking_fa7, docking_jak2, docking_parp1, logP, QED]
    target_raw = [2.0, -12.0, -12.0, -12.0, -12.0, -12.0, 2.5, 0.9]
    
    # CHỈNH LẠI DUNG SAI (SIGMA): Làm bẹt đường cong cho các chỉ số Docking
    tolerances = [
        1.0,    # SAS (Z=-1.26, sigma=1.0 là đủ nhạy)
        10.0,   # docking_5ht1b (Z=-14, cần sigma=10 để R > 0)
        10.0,   # docking_braf
        10.0,   # docking_fa7
        10.0,   # docking_jak2
        10.0,   # docking_parp1
        1.0,    # logP (Z=0.01, sigma=1.0 là đủ)
        1.0     # qed (Z=1.21, sigma=1.0 là đủ)
    ]
    
    reward_computer = RewardComputer(
        property_scaler_path=STAGE1_SCALER,
        hes_model=hes_model,
        target_properties=target_raw,              # <--- CRITICAL FIX: Use raw target values
        property_sigma=tolerances,                 # Use custom tolerances
        hes_output_is_normalized=HES_OUTPUT_IS_NORMALIZED,
        target_properties_are_normalized=False,    # <--- CRITICAL FIX: Force RewardComputer to scale the raw targets
    )
    print(f"  ✓ Reward computer initialized")
    print(f"    Target properties: {reward_computer.target_properties}")
    print(f"    Property tolerances: {reward_computer.property_sigma}")
    
    # Initialize environment
    print("\n[2/4] Initializing environment...")
    env = MoleculeEnv(
        hes_model=hes_model,
        motif_vocab=motif_vocab,
        shape_vocab=shape_vocab,
        property_scaler=property_scaler,
        reward_computer=reward_computer,
    )
    print(f"  ✓ Environment initialized")
    print(f"    - Observation space: {env.observation_space.shape}")
    print(f"    - Action space: Dict with {len(env.action_space.spaces)} actions")
    
    # Initialize SAC agent
    print("\n[3/4] Initializing SAC agent...")
    agent = SACAgent(device=DEVICE)
    print(f"  ✓ Agent initialized on device: {DEVICE}")
    print(f"    - Actor: {sum(p.numel() for p in agent.actor.parameters()):,} params")
    print(f"    - Critic1: {sum(p.numel() for p in agent.critic1.parameters()):,} params")
    print(f"    - Critic2: {sum(p.numel() for p in agent.critic2.parameters()):,} params")
    
    # Initialize replay buffer
    print("\n[4/4] Initializing replay buffer...")
    replay_buffer = ReplayBuffer(capacity=REPLAY_BUFFER_SIZE)
    print(f"  ✓ Replay buffer capacity: {REPLAY_BUFFER_SIZE:,}")
    
    print("\n" + "-"*80)
    return env, agent, replay_buffer, reward_computer


def run_episode(env, agent, reward_computer, replay_buffer, episode_num, training=True, batch_size_sac=BATCH_SIZE_SAC):
    """
    Run a single training episode.
    
    Args:
        env: MoleculeEnv instance
        agent: SACAgent instance
        reward_computer: RewardComputer instance
        replay_buffer: ReplayBuffer instance
        episode_num: episode number
        training: whether in training mode
        batch_size_sac: replay buffer sample size for SAC updates
    
    Returns:
        episode_info: dict with episode statistics
    """
    state = env.reset()
    episode_reward = 0.0
    episode_length = 0
    done = False
    
    episode_info = {
        'episode': episode_num,
        'total_reward': 0.0,
        'length': 0,
        'valid': True,
        'errors': [],
    }
    
    while not done and episode_length < MAX_STEPS_PER_EPISODE:
        # Get valid action mask
        action_mask = env.get_action_mask()
        
        # Select and execute action
        action = agent.select_action(state, training=training, action_mask=action_mask)
        next_state, reward, done, info = env.step(action)
        
        episode_reward += reward
        episode_length += 1
        
        # Store in replay buffer
        if training:
            replay_buffer.push(state, action, reward, next_state, done)
        
        # SAC updates
        if training and len(replay_buffer) >= LEARN_STARTS:
            for _ in range(NUM_UPDATES_PER_STEP):
                batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = replay_buffer.sample(batch_size_sac)
                
                # Convert to tensors
                batch_dict = {
                    'states': torch.FloatTensor(batch_states),
                    'actions': {
                        'a1': torch.LongTensor([a['a1'] for a in batch_actions]),
                        'a2': torch.LongTensor([a['a2'] for a in batch_actions]),
                        'a3': torch.LongTensor([a['a3'] for a in batch_actions]),
                        'a2_atom': torch.LongTensor(np.array([a['a2_atom'] for a in batch_actions])),
                        'a2_bond': torch.LongTensor(np.array([a['a2_bond'] for a in batch_actions])),
                    },
                    'rewards': torch.FloatTensor(batch_rewards),
                    'next_states': torch.FloatTensor(batch_next_states),
                    'dones': torch.FloatTensor(batch_dones),
                }
                
                losses = agent.update(batch_dict)
        
        state = next_state
    
    episode_info['total_reward'] = episode_reward
    episode_info['length'] = episode_length
    episode_info['final_smiles'] = env.molecule_smiles
    
    return episode_info


def train(num_episodes=NUM_EPISODES, resume_from=None, batch_size_sac=BATCH_SIZE_SAC):
    """
    Main training loop.
    
    Args:
        num_episodes: number of episodes to train
        resume_from: optional checkpoint path to resume from
        batch_size_sac: replay buffer sample size for SAC updates
    """
    # Set random seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    # Load Stage 1 components
    hes_model, motif_vocab, shape_vocab, property_scaler = load_stage1_components()
    
    # Initialize Stage 2 components
    env, agent, replay_buffer, reward_computer = initialize_training(
        hes_model, motif_vocab, shape_vocab, property_scaler
    )
    
    # Resume if checkpoint provided
    if resume_from and Path(resume_from).exists():
        print(f"\n[*] Resuming from checkpoint: {resume_from}")
        agent.load(Path(resume_from))
    
    # Training loop
    print("\n" + "="*80)
    print("STARTING STAGE 2 TRAINING")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  - Episodes: {num_episodes}")
    print(f"  - Batch size: {batch_size_sac}")
    print(f"  - Learning starts: {LEARN_STARTS} steps")
    print(f"  - Device: {DEVICE}")
    print(f"  - Max steps per episode: {MAX_STEPS_PER_EPISODE}")
    print("\n" + "-"*80 + "\n")
    
    episode_metrics = defaultdict(list)
    start_time = time.time()
    best_eval_reward = -np.inf
    
    try:
        for episode in range(num_episodes):
            # Run episode
            episode_info = run_episode(
                env,
                agent,
                reward_computer,
                replay_buffer,
                episode,
                training=True,
                batch_size_sac=batch_size_sac,
            )
            
            # Log metrics
            episode_metrics['reward'].append(episode_info['total_reward'])
            episode_metrics['length'].append(episode_info['length'])
            
            # Print progress
            if (episode + 1) % LOG_INTERVAL == 0:
                avg_reward = np.mean(episode_metrics['reward'][-LOG_INTERVAL:])
                avg_length = np.mean(episode_metrics['length'][-LOG_INTERVAL:])
                elapsed_time = time.time() - start_time
                
                print(f"[Episode {episode+1:5d}/{num_episodes}] "
                      f"Avg Reward: {avg_reward:7.3f} | "
                      f"Avg Length: {avg_length:6.2f} | "
                      f"Buffer Size: {len(replay_buffer):6d} | "
                      f"Time: {elapsed_time:7.1f}s")
            
            # Save checkpoint
            if (episode + 1) % SAVE_INTERVAL == 0:
                checkpoint_path = CHECKPOINT_DIR / f"agent_ep{episode+1:06d}.pt"
                agent.save(checkpoint_path)
                print(f"[✓] Saved checkpoint at episode {episode+1}")
            
            # Evaluation
            if (episode + 1) % EVAL_INTERVAL == 0:
                print(f"\n[*] Running evaluation at episode {episode+1}...")
                eval_rewards = []
                for _ in range(NUM_EVAL_EPISODES):
                    eval_info = run_episode(
                        env,
                        agent,
                        reward_computer,
                        replay_buffer,
                        episode,
                        training=False,
                        batch_size_sac=batch_size_sac,
                    )
                    eval_rewards.append(eval_info['total_reward'])
                
                eval_avg = np.mean(eval_rewards)
                eval_std = np.std(eval_rewards)
                print(f"    Eval Avg Reward: {eval_avg:.3f} ± {eval_std:.3f}")
                
                if eval_avg > best_eval_reward:
                    best_eval_reward = eval_avg
                    checkpoint_path = CHECKPOINT_DIR / "agent_best.pt"
                    agent.save(checkpoint_path)
                    print(f"    [🌟] New best model saved with evaluation reward: {best_eval_reward:.3f}\n")
                else:
                    print()
    
    except KeyboardInterrupt:
        print("\n\n[!] Training interrupted by user")
    
    except Exception as e:
        print(f"\n\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Save final checkpoint
        final_checkpoint = CHECKPOINT_DIR / f"agent_final.pt"
        agent.save(final_checkpoint)
        
        # Summary
        print("\n" + "="*80)
        print("TRAINING SUMMARY")
        print("="*80)
        total_time = time.time() - start_time
        final_avg_reward = np.mean(episode_metrics['reward'][-100:]) if episode_metrics['reward'] else 0
        
        print(f"\nFinal Statistics:")
        print(f"  - Total episodes: {len(episode_metrics['reward'])}")
        print(f"  - Total time: {total_time/3600:.2f} hours")
        print(f"  - Final avg reward (last 100 eps): {final_avg_reward:.3f}")
        print(f"  - Replay buffer final size: {len(replay_buffer):,}")
        print(f"\nCheckpoints saved to: {CHECKPOINT_DIR}")
        print("="*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2 RL Training")
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES, help="Number of episodes")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--batch-size-sac", type=int, default=BATCH_SIZE_SAC, help="SAC replay sample batch size")
    
    args = parser.parse_args()
    
    train(
        num_episodes=args.episodes,
        resume_from=args.resume,
        batch_size_sac=args.batch_size_sac,
    )
