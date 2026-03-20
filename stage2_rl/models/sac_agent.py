"""
SAC (Soft Actor-Critic) Agent Networks
Implements actor and critic networks for hierarchical molecular generation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any, Optional
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "stage1_hes"))

from stage2_rl.training.config import *


class ActorNetwork(nn.Module):
    """
    Actor network that outputs action probabilities for hierarchical action space.
    
    Input: HES encoding (256-dim)
    Outputs: 
    - a1 logits: (batch, MAX_ATOMS_PER_MOLECULE + 1)  # last index is STOP
    - a2 logits: (batch, NUM_SHAPES)
    - a3 logits: (batch, 4)
    """
    
    def __init__(self, input_dim=HES_ENCODING_DIM, hidden_dim=ACTOR_HIDDEN_DIM):
        super().__init__()
        
        # Shared layers
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Action heads (one for each action dimension)
        self.a1_head = nn.Linear(hidden_dim, MAX_ATOMS_PER_MOLECULE + 1)
        self.a2_head = nn.Linear(hidden_dim, NUM_SHAPES)
        self.a3_head = nn.Linear(hidden_dim, 4)
        self.a2_atom_head = nn.Linear(hidden_dim, MAX_ATOMS_PER_MOLECULE * NUM_ATOM_TYPES)
        self.a2_bond_head = nn.Linear(hidden_dim, MAX_ATOMS_PER_MOLECULE * MAX_ATOMS_PER_MOLECULE * NUM_BOND_TYPES)
        
    def forward(self, state: torch.Tensor, action_mask: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            state: (batch_size, HES_ENCODING_DIM)
            action_mask: dict with boolean masks for 'a1', 'a2', 'a3' (1 for valid, 0 for invalid)
        
        Returns:
            dict with keys 'a1', 'a2', 'a3' containing logits
        """
        features = self.net(state)
        
        batch_size = features.size(0)
        
        logits = {
            'a1': self.a1_head(features),
            'a2': self.a2_head(features),
            'a3': self.a3_head(features),
            'a2_atom': self.a2_atom_head(features).view(batch_size, MAX_ATOMS_PER_MOLECULE, NUM_ATOM_TYPES),
            'a2_bond': self.a2_bond_head(features).view(batch_size, MAX_ATOMS_PER_MOLECULE, MAX_ATOMS_PER_MOLECULE, NUM_BOND_TYPES),
        }
        
        if action_mask is not None:
            # Apply large negative value to invalid actions to mask them in softmax
            for key in action_mask:
                if key in logits:
                    # Mask invalid actions (where mask == 0) with a very large negative number
                    bool_mask = action_mask[key].bool()
                    logits[key] = logits[key].masked_fill(~bool_mask, -1e9)
                    
        return logits
    
    def sample_action(self, state: torch.Tensor, temperature=1.0, action_mask: Optional[Dict[str, torch.Tensor]] = None):
        """
        Sample actions from actor network using softmax sampling.
        
        Args:
            state: (batch_size, HES_ENCODING_DIM)
            temperature: temperature for softmax (higher = more stochastic)
            action_mask: dict with boolean masks for 'a1', 'a2', 'a3'
        
        Returns:
            action_dict: dict with sampled integer actions
            log_probs: log probability of sampled actions
        """
        logits_dict = self.forward(state, action_mask)
        
        action_dict = {}
        log_probs = torch.zeros(state.size(0), device=state.device)
        
        for action_key in ['a1', 'a2', 'a3', 'a2_atom', 'a2_bond']:
            logits = logits_dict[action_key]
            
            # Apply temperature scaling and softmax
            probs = F.softmax(logits, dim=-1)
            
            # Sample action from distribution
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            
            action_dict[action_key] = action
            
            log_prob = dist.log_prob(action)
            if action_key == 'a2_atom':
                log_prob = log_prob.sum(dim=1)
            elif action_key == 'a2_bond':
                log_prob = log_prob.sum(dim=[1, 2])
            
            log_probs += log_prob
        
        return action_dict, log_probs

    def sample_action_differentiable(self, state: torch.Tensor, temperature=1.0, action_mask=None):
        logits_dict = self.forward(state, action_mask)
        action_onehots = {}
        log_probs = torch.zeros(state.size(0), device=state.device)
        
        for action_key in ['a1', 'a2', 'a3', 'a2_atom', 'a2_bond']:
            logits = logits_dict[action_key]
            # Differentiable one-hot sampling
            onehot = F.gumbel_softmax(logits, tau=temperature, hard=True)
            action_onehots[action_key] = onehot
            
            # Calculate log probabilities
            action = onehot.argmax(dim=-1)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            log_prob = dist.log_prob(action)
            
            if action_key == 'a2_atom':
                log_prob = log_prob.sum(dim=1)
            elif action_key == 'a2_bond':
                log_prob = log_prob.sum(dim=[1, 2])
            log_probs += log_prob
            
        return action_onehots, log_probs


class CriticNetwork(nn.Module):
    """
    Critic network (Q-function) that estimates state value.
    
    Input: HES encoding + action dict
    Output: Q-value (scalar)
    """
    
    def __init__(self, input_dim=HES_ENCODING_DIM, hidden_dim=CRITIC_HIDDEN_DIM):
        super().__init__()
        
        # Calculate action encoding size
        action_encoding_size = ((MAX_ATOMS_PER_MOLECULE + 1) + NUM_SHAPES + 4) + \
                               (MAX_ATOMS_PER_MOLECULE * NUM_ATOM_TYPES) + \
                               (MAX_ATOMS_PER_MOLECULE * MAX_ATOMS_PER_MOLECULE * NUM_BOND_TYPES)
        
        # Network: concat state and action encoding
        self.net = nn.Sequential(
            nn.Linear(input_dim + action_encoding_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # Output single Q-value
        )
    
    def forward(self, state: torch.Tensor, action_onehots: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            state: (batch_size, HES_ENCODING_DIM)
            action_onehots: dict with 'a1', 'a2', 'a3', 'a2_atom', 'a2_bond' as float tensors (one-hot or soft one-hot)
        
        Returns:
            q_values: (batch_size, 1)
        """
        batch_size = state.size(0)
        
        a1_onehot = action_onehots['a1']
        a2_onehot = action_onehots['a2']
        a3_onehot = action_onehots['a3']
        a2_atom_onehot = action_onehots['a2_atom'].view(batch_size, -1)
        a2_bond_onehot = action_onehots['a2_bond'].view(batch_size, -1)
        
        # Concatenate action encodings
        action_encoding = torch.cat([a1_onehot, a2_onehot, a3_onehot, a2_atom_onehot, a2_bond_onehot], dim=-1)
        
        # Concatenate state and action encoding
        state_action = torch.cat([state, action_encoding], dim=-1)
        
        # Compute Q-value
        q_value = self.net(state_action)
        
        return q_value


class SACAgent:
    """
    Soft Actor-Critic agent for molecular generation.
    """
    
    def __init__(self, device=DEVICE):
        """
        Initialize SAC agent with actor and critic networks.
        
        Args:
            device: torch device (cuda or cpu)
        """
        self.device = device
        
        # Initialize networks
        self.actor = ActorNetwork(input_dim=HES_ENCODING_DIM, hidden_dim=ACTOR_HIDDEN_DIM).to(device)
        self.critic1 = CriticNetwork(input_dim=HES_ENCODING_DIM, hidden_dim=CRITIC_HIDDEN_DIM).to(device)
        self.critic2 = CriticNetwork(input_dim=HES_ENCODING_DIM, hidden_dim=CRITIC_HIDDEN_DIM).to(device)
        
        # Target critic networks (for stability)
        self.target_critic1 = CriticNetwork(input_dim=HES_ENCODING_DIM, hidden_dim=CRITIC_HIDDEN_DIM).to(device)
        self.target_critic2 = CriticNetwork(input_dim=HES_ENCODING_DIM, hidden_dim=CRITIC_HIDDEN_DIM).to(device)
        
        # Copy initial weights to target networks
        self._soft_update_target_networks(tau=1.0)
        
        # Learnable entropy coefficient
        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=device)
        self.target_entropy = -np.prod([MAX_ATOMS_PER_MOLECULE + 1, NUM_SHAPES, 4])
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=ACTOR_LEARNING_RATE)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=CRITIC_LEARNING_RATE)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=CRITIC_LEARNING_RATE)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=ALPHA_LEARNING_RATE)
    
    def select_action(self, state: np.ndarray, training=True, action_mask: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
        """
        Select action given state.
        
        Args:
            state: observation (HES encoding)
            training: if True, sample stochastically; if False, use greedy action
            action_mask: dict with boolean masks for 'a1', 'a2', 'a3'
        
        Returns:
            action_dict: dict with 'a1', 'a2', 'a3' as integers and 'a2_atom', 'a2_bond' as arrays
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if training:
                action_dict, _ = self.actor.sample_action(state_tensor, temperature=1.0, action_mask=action_mask)
            else:
                logits_dict = self.actor(state_tensor, action_mask=action_mask)
                action_dict = {k: v.argmax(dim=-1) for k, v in logits_dict.items()}
        
        # Convert to numpy dict
        out_dict = {}
        for k, v in action_dict.items():
            # For multi-dimensional action items (a2_atom, a2_bond), keep them as arrays
            if v.numel() > 1:
                out_dict[k] = v[0].cpu().numpy()
            else:
                out_dict[k] = v.cpu().item() if hasattr(v, 'item') else v
                
        return out_dict
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update actor and critic networks from a batch of experience.
        
        Args:
            batch: Dict with keys 'states', 'actions', 'rewards', 'next_states', 'dones'
        
        Returns:
            losses: Dict with loss values for logging
        """
        states = batch['states'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        next_states = batch['next_states'].to(self.device)
        dones = batch['dones'].to(self.device)
        
        # Convert action dict to tensors
        actions = {k: v.to(self.device) for k, v in batch['actions'].items()}
        
        # Convert replay buffer actions to one-hots
        actions_onehot = {
            'a1': F.one_hot(actions['a1'], MAX_ATOMS_PER_MOLECULE + 1).float(),
            'a2': F.one_hot(actions['a2'], NUM_SHAPES).float(),
            'a3': F.one_hot(actions['a3'], 4).float(),
            'a2_atom': F.one_hot(actions['a2_atom'], NUM_ATOM_TYPES).float(),
            'a2_bond': F.one_hot(actions['a2_bond'], NUM_BOND_TYPES).float()
        }
        
        # --- CRITIC UPDATE ---
        with torch.no_grad():
            # Get next actions from actor
            next_action_dict, next_log_probs = self.actor.sample_action(next_states)
            next_actions_onehot = {
                'a1': F.one_hot(next_action_dict['a1'], MAX_ATOMS_PER_MOLECULE + 1).float(),
                'a2': F.one_hot(next_action_dict['a2'], NUM_SHAPES).float(),
                'a3': F.one_hot(next_action_dict['a3'], 4).float(),
                'a2_atom': F.one_hot(next_action_dict['a2_atom'], NUM_ATOM_TYPES).float(),
                'a2_bond': F.one_hot(next_action_dict['a2_bond'], NUM_BOND_TYPES).float()
            }
            
            # Compute target Q-values using target networks
            target_q1 = self.target_critic1(next_states, next_actions_onehot)
            target_q2 = self.target_critic2(next_states, next_actions_onehot)
            target_q = torch.min(target_q1, target_q2)
            
            # Subtract entropy term
            alpha = torch.exp(self.log_alpha)
            target_q = target_q - alpha * next_log_probs.detach().unsqueeze(-1)
            
            # Compute target with reward and discount
            target_q_value = rewards.unsqueeze(-1) + (1 - dones.unsqueeze(-1)) * GAMMA * target_q
        
        # Compute critic losses
        q1_pred = self.critic1(states, actions_onehot)
        q2_pred = self.critic2(states, actions_onehot)
        
        critic1_loss = F.mse_loss(q1_pred, target_q_value)
        critic2_loss = F.mse_loss(q2_pred, target_q_value)
        
        # Update critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # --- ACTOR UPDATE ---
        # USE DIFFERENTIABLE SAMPLING SO GRADIENTS FLOW BACK
        action_onehots_diff, log_probs = self.actor.sample_action_differentiable(states)
        
        # Compute Q-values from updated critics
        q1 = self.critic1(states, action_onehots_diff)
        q2 = self.critic2(states, action_onehots_diff)
        q = torch.min(q1, q2)
        
        # Actor loss: maximize Q - entropy
        alpha = torch.exp(self.log_alpha)
        actor_loss = (alpha * log_probs.unsqueeze(-1) - q).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # ===== Update Entropy Coefficient =====
        alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # ===== Update Target Networks =====
        self._soft_update_target_networks(tau=TAU)
        
        return {
            'critic1_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': alpha.item(),
        }
    
    def _soft_update_target_networks(self, tau=TAU):
        """Soft update target networks: target = tau * current + (1-tau) * target"""
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
    
    def save(self, checkpoint_path: Path):
        """Save agent networks and optimizers."""
        checkpoint = {
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'target_critic1': self.target_critic1.state_dict(),
            'target_critic2': self.target_critic2.state_dict(),
            'log_alpha': self.log_alpha.data,
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic1_optimizer': self.critic1_optimizer.state_dict(),
            'critic2_optimizer': self.critic2_optimizer.state_dict(),
            'alpha_optimizer': self.alpha_optimizer.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"[✓] Saved agent checkpoint to {checkpoint_path}")
    
    def load(self, checkpoint_path: Path):
        """Load agent networks and optimizers."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.target_critic1.load_state_dict(checkpoint['target_critic1'])
        self.target_critic2.load_state_dict(checkpoint['target_critic2'])
        self.log_alpha.data = checkpoint['log_alpha']
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer'])
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
        print(f"[✓] Loaded agent checkpoint from {checkpoint_path}")
