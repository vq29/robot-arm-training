"""
Training script for SAC on the robotic arm reaching task.
Logs to TensorBoard and saves training histories + best model.

Usage:
    python train.py                    # default 1000 episodes
    python train.py --episodes 200     # custom episode count
    python train.py --no-gui           # headless training (faster)
"""

import os
import argparse
import numpy as np

from config import TrainingConfig
from env import RobotArmEnv
from agent import SACAgent
from replay_buffer import ReplayBuffer

# TensorBoard (optional — graceful fallback if not installed)
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ImportError:
    HAS_TB = False


def train(cfg: TrainingConfig, gui: bool = True):
    """Run SAC training loop with TensorBoard logging."""

    env = RobotArmEnv(gui=gui, max_steps=cfg.max_steps, config=cfg)
    agent = SACAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        hidden_dim=cfg.hidden_dim,
        actor_lr=cfg.actor_lr,
        critic_lr=cfg.critic_lr,
        alpha_lr=cfg.alpha_lr,
        gamma=cfg.gamma,
        tau=cfg.tau,
        init_alpha=cfg.init_alpha,
    )
    buffer = ReplayBuffer(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        capacity=cfg.buffer_capacity,
    )

    # ── TensorBoard ───────────────────────────────────────────────────
    writer = None
    if HAS_TB:
        tb_dir = os.path.join(cfg.log_dir, "tensorboard")
        writer = SummaryWriter(log_dir=tb_dir)

    # ── Histories ─────────────────────────────────────────────────────
    reward_history = []
    distance_history = []
    success_history = []
    actor_loss_history = []
    critic_loss_history = []
    alpha_history = []
    best_reward = -float("inf")

    print("=" * 60)
    print("  SAC Training — Robotic Arm Reaching Task")
    print("=" * 60)
    print(f"  Episodes:      {cfg.num_episodes}")
    print(f"  Max steps/ep:  {cfg.max_steps}")
    print(f"  Sub-steps:     {cfg.n_substeps}")
    print(f"  Batch size:    {cfg.batch_size}")
    print(f"  Warmup steps:  {cfg.warmup_steps}")
    print(f"  Device:        {agent.device}")
    print(f"  TensorBoard:   {'enabled' if writer else 'disabled (install tensorboard)'}")
    print("=" * 60)

    total_steps = 0

    for episode in range(1, cfg.num_episodes + 1):
        state = env.reset()
        episode_reward = 0.0
        episode_distances = []
        episode_actor_losses = []
        episode_critic_losses = []
        episode_success = False

        for step in range(cfg.max_steps):
            # Select action
            if total_steps < cfg.warmup_steps:
                action = np.random.uniform(-1, 1, size=env.action_dim).astype(
                    np.float32
                )
            else:
                action = agent.select_action(state, deterministic=False)

            # Step environment
            next_state, reward, done, info = env.step(action)
            buffer.add(state, action, reward, next_state, done)

            episode_reward += reward
            episode_distances.append(info["distance"])
            if info["success"]:
                episode_success = True

            # Train agent
            if len(buffer) >= cfg.batch_size and total_steps >= cfg.warmup_steps:
                for _ in range(cfg.updates_per_step):
                    losses = agent.update(buffer, cfg.batch_size)
                    episode_actor_losses.append(losses["actor_loss"])
                    episode_critic_losses.append(losses["critic_loss"])

            state = next_state
            total_steps += 1

            if done:
                break

        # ── Per-episode bookkeeping ───────────────────────────────────
        avg_dist = np.mean(episode_distances)
        reward_history.append(episode_reward)
        distance_history.append(avg_dist)
        success_history.append(float(episode_success))

        avg_actor_loss = np.mean(episode_actor_losses) if episode_actor_losses else 0.0
        avg_critic_loss = (
            np.mean(episode_critic_losses) if episode_critic_losses else 0.0
        )
        current_alpha = agent.alpha.item()
        actor_loss_history.append(avg_actor_loss)
        critic_loss_history.append(avg_critic_loss)
        alpha_history.append(current_alpha)

        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(cfg.model_dir)

        # ── TensorBoard logging ───────────────────────────────────────
        if writer:
            writer.add_scalar("reward/episode", episode_reward, episode)
            writer.add_scalar("distance/avg", avg_dist, episode)
            writer.add_scalar(
                "success_rate/last_50",
                np.mean(success_history[-50:]),
                episode,
            )
            writer.add_scalar("loss/actor", avg_actor_loss, episode)
            writer.add_scalar("loss/critic", avg_critic_loss, episode)
            writer.add_scalar("alpha", current_alpha, episode)

        # ── Console logging ───────────────────────────────────────────
        if episode % cfg.log_interval == 0 or episode == 1:
            recent_rwd = np.mean(reward_history[-cfg.log_interval :])
            recent_dist = np.mean(distance_history[-cfg.log_interval :])
            recent_success = np.mean(success_history[-cfg.log_interval :]) * 100
            print(
                f"  Ep {episode:>4d}/{cfg.num_episodes}  |  "
                f"Reward: {episode_reward:>8.1f}  |  "
                f"Dist: {avg_dist:.4f}  |  "
                f"Success: {recent_success:>5.1f}%  |  "
                f"alpha: {current_alpha:.3f}"
            )

    # ── Save everything ───────────────────────────────────────────────
    os.makedirs(cfg.log_dir, exist_ok=True)
    np.save(os.path.join(cfg.log_dir, "reward_history.npy"), np.array(reward_history))
    np.save(
        os.path.join(cfg.log_dir, "distance_history.npy"), np.array(distance_history)
    )
    np.save(
        os.path.join(cfg.log_dir, "success_history.npy"), np.array(success_history)
    )
    np.save(
        os.path.join(cfg.log_dir, "actor_loss_history.npy"),
        np.array(actor_loss_history),
    )
    np.save(
        os.path.join(cfg.log_dir, "critic_loss_history.npy"),
        np.array(critic_loss_history),
    )
    np.save(os.path.join(cfg.log_dir, "alpha_history.npy"), np.array(alpha_history))

    # Save final model
    agent.save(cfg.model_dir)

    if writer:
        writer.close()

    print("=" * 60)
    print("  Training complete!")
    print(f"  Best episode reward:   {best_reward:.1f}")
    print(f"  Final avg distance:    {np.mean(distance_history[-20:]):.4f}")
    print(f"  Final success rate:    {np.mean(success_history[-50:]) * 100:.1f}%")
    print(f"  Models saved to:       {cfg.model_dir}/")
    print(f"  Logs saved to:         {cfg.log_dir}/")
    if HAS_TB:
        print(f"  TensorBoard:           tensorboard --logdir={cfg.log_dir}/tensorboard")
    print("=" * 60)

    env.close()
    return reward_history, distance_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAC on robotic arm")
    parser.add_argument(
        "--episodes", type=int, default=None, help="Number of training episodes"
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument(
        "--warmup", type=int, default=None, help="Random exploration steps"
    )
    parser.add_argument(
        "--no-gui", action="store_true", help="Disable MuJoCo GUI (faster training)"
    )
    args = parser.parse_args()

    cfg = TrainingConfig()
    if args.episodes is not None:
        cfg.num_episodes = args.episodes
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.warmup is not None:
        cfg.warmup_steps = args.warmup

    train(cfg, gui=not args.no_gui)
