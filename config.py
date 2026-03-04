"""
Centralized configuration for SAC training on the robotic arm task.
All hyperparameters in one place for easy tuning and reproducibility.
"""

from dataclasses import dataclass, field


@dataclass
class TrainingConfig:
    """All hyperparameters for the SAC robotic arm project."""

    # ── Training ──────────────────────────────────────────────────────
    num_episodes: int = 3000
    max_steps: int = 300
    batch_size: int = 256
    warmup_steps: int = 1000
    log_interval: int = 10
    updates_per_step: int = 1          # gradient updates per env step

    # ── SAC Agent ─────────────────────────────────────────────────────
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4             # entropy temperature learning rate
    gamma: float = 0.99                # discount factor
    tau: float = 0.005                 # soft-update coefficient
    hidden_dim: int = 256              # hidden layer width
    init_alpha: float = 0.2            # initial entropy coefficient
    buffer_capacity: int = 1_000_000

    # ── Environment ───────────────────────────────────────────────────
    max_torque: float = 50.0
    n_substeps: int = 4                # physics sub-steps per control step
    reach_threshold: float = 0.08      # metres — success if distance < this
    target_r_min: float = 0.3          # min spawn radius for target
    target_r_max: float = 0.6          # max spawn radius for target
    target_z_min: float = 0.15
    target_z_max: float = 0.7
    velocity_penalty: float = 0.1      # weight on joint-velocity penalty

    # ── Paths ─────────────────────────────────────────────────────────
    model_dir: str = "models"
    log_dir: str = "logs"


# Convenience singleton
DEFAULT_CONFIG = TrainingConfig()
