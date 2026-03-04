"""
Soft Actor-Critic (SAC) agent for continuous control.
- Stochastic Gaussian actor with tanh squashing
- Twin critics (clipped double-Q) to reduce overestimation
- Automatic entropy temperature (alpha) tuning

Reference: Haarnoja et al., "Soft Actor-Critic Algorithms and Applications", 2019
"""

import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# ======================================================================
#  Neural Network Architectures
# ======================================================================

LOG_STD_MIN = -20
LOG_STD_MAX = 2


class GaussianActor(nn.Module):
    """
    Stochastic policy: state → (mean, log_std) → tanh-squashed action.
    Uses the reparameterization trick for differentiable sampling.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        h = self.trunk(state)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state):
        """
        Sample an action via reparameterization trick + tanh squashing.
        Returns (action, log_prob) where action ∈ [-1, 1].
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mean, std)

        # Reparameterization: z = mean + std * eps
        z = dist.rsample()
        action = torch.tanh(z)

        # Log-prob with tanh correction (Appendix C of SAC paper)
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    def deterministic_action(self, state):
        """Return the mean action (no noise) for evaluation."""
        mean, _ = self.forward(state)
        return torch.tanh(mean)


class QNetwork(nn.Module):
    """Maps (state, action) → Q-value."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))


# ======================================================================
#  SAC Agent
# ======================================================================

class SACAgent:
    """
    Soft Actor-Critic with automatic entropy tuning.

    Key differences from DDPG:
    - Stochastic policy (exploration is built-in, no external noise)
    - Twin Q-networks to combat overestimation
    - Entropy bonus in the objective → better exploration
    - Learned temperature α adjusts exploration automatically
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        init_alpha: float = 0.2,
        device=None,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim

        # ── Actor ─────────────────────────────────────────────────────
        self.actor = GaussianActor(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        # ── Twin Critics ──────────────────────────────────────────────
        self.critic1 = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic2 = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=critic_lr,
        )

        # ── Automatic Entropy Tuning ──────────────────────────────────
        self.target_entropy = -float(action_dim)  # heuristic: -dim(A)
        self.log_alpha = torch.tensor(
            np.log(init_alpha), dtype=torch.float32, device=self.device, requires_grad=True
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    # ------------------------------------------------------------------
    #  Action selection
    # ------------------------------------------------------------------
    def select_action(self, state, deterministic=False):
        """Select action given a single state (numpy → numpy)."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if deterministic:
                action = self.actor.deterministic_action(state_t)
            else:
                action, _ = self.actor.sample(state_t)
        return action.cpu().numpy()[0]

    # ------------------------------------------------------------------
    #  Training update
    # ------------------------------------------------------------------
    def update(self, replay_buffer, batch_size=256):
        """
        One gradient step on actor, twin critics, and alpha.
        Returns dict of losses for logging.
        """
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # ── Critic loss (clipped double-Q) ────────────────────────────
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_target = self.critic1_target(next_states, next_actions)
            q2_target = self.critic2_target(next_states, next_actions)
            q_target = torch.min(q1_target, q2_target) - self.alpha * next_log_probs
            y = rewards + (1 - dones) * self.gamma * q_target

        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ── Actor loss ────────────────────────────────────────────────
        new_actions, log_probs = self.actor.sample(states)
        q1_new = self.critic1(states, new_actions)
        q2_new = self.critic2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha.detach() * log_probs - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ── Alpha (entropy temperature) loss ──────────────────────────
        alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # ── Soft-update target networks ───────────────────────────────
        self._soft_update(self.critic1, self.critic1_target)
        self._soft_update(self.critic2, self.critic2_target)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha.item(),
        }

    def _soft_update(self, source, target):
        for sp, tp in zip(source.parameters(), target.parameters()):
            tp.data.copy_(self.tau * sp.data + (1.0 - self.tau) * tp.data)

    # ------------------------------------------------------------------
    #  Save / Load
    # ------------------------------------------------------------------
    def save(self, directory="models"):
        os.makedirs(directory, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(directory, "actor.pth"))
        torch.save(self.critic1.state_dict(), os.path.join(directory, "critic1.pth"))
        torch.save(self.critic2.state_dict(), os.path.join(directory, "critic2.pth"))
        torch.save(self.log_alpha, os.path.join(directory, "log_alpha.pth"))

    def load(self, directory="models"):
        self.actor.load_state_dict(
            torch.load(os.path.join(directory, "actor.pth"), map_location=self.device)
        )
        self.critic1.load_state_dict(
            torch.load(os.path.join(directory, "critic1.pth"), map_location=self.device)
        )
        self.critic2.load_state_dict(
            torch.load(os.path.join(directory, "critic2.pth"), map_location=self.device)
        )
        self.log_alpha = torch.load(
            os.path.join(directory, "log_alpha.pth"), map_location=self.device
        )
        # Rebuild targets
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)
