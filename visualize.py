"""
Professional training dashboard — generates a 6-panel figure from logs.

Panels:
  1. Episode Reward        2. Distance to Target
  3. Success Rate (%)      4. Actor Loss
  5. Critic Loss           6. Entropy Coefficient (α)

Usage:
    python visualize.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt


# ── Styling ───────────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "figure.facecolor": "#1e1e2e",
        "axes.facecolor": "#2a2a3d",
        "axes.edgecolor": "#444466",
        "axes.labelcolor": "#ccccdd",
        "text.color": "#ccccdd",
        "xtick.color": "#999999",
        "ytick.color": "#999999",
        "grid.color": "#333355",
        "grid.alpha": 0.5,
        "font.family": "sans-serif",
        "font.size": 10,
    }
)

# Color palette
RAW_ALPHA = 0.25
COLORS = {
    "reward_raw": "#5b7fbf",
    "reward_smooth": "#e63946",
    "dist_raw": "#2a9d8f",
    "dist_smooth": "#e76f51",
    "success": "#a8dadc",
    "actor_loss": "#f4a261",
    "critic_loss": "#457b9d",
    "alpha": "#e9c46a",
}


def smooth(data, window=20):
    """Simple moving-average smoothing."""
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="valid")


def _plot_metric(ax, data, title, ylabel, raw_color, smooth_color, window=20):
    """Plot raw + smoothed overlay."""
    episodes = np.arange(1, len(data) + 1)
    ax.plot(episodes, data, alpha=RAW_ALPHA, color=raw_color, linewidth=0.8, label="Raw")

    if len(data) >= window:
        sm = smooth(data, window)
        sm_x = np.arange(window, window + len(sm))
        ax.plot(sm_x, sm, color=smooth_color, linewidth=2.0, label=f"Smoothed ({window}-ep)")

    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold", fontsize=11)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True)


def generate_dashboard(log_dir="logs", save_path="training_dashboard.png"):
    """Create 6-panel training dashboard."""

    # Load data
    def _load(name):
        path = os.path.join(log_dir, name)
        return np.load(path) if os.path.exists(path) else None

    rewards = _load("reward_history.npy")
    distances = _load("distance_history.npy")
    successes = _load("success_history.npy")
    actor_losses = _load("actor_loss_history.npy")
    critic_losses = _load("critic_loss_history.npy")
    alphas = _load("alpha_history.npy")

    if rewards is None:
        print("  ERROR: Training logs not found in logs/")
        print("  Run 'python train.py' first.")
        return

    n_eps = len(rewards)
    print(f"  Loaded {n_eps} episodes of training data")

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))

    # Panel 1 — Reward
    _plot_metric(
        axes[0, 0], rewards,
        "Episode Reward", "Reward",
        COLORS["reward_raw"], COLORS["reward_smooth"],
    )

    # Panel 2 — Distance
    if distances is not None:
        _plot_metric(
            axes[0, 1], distances,
            "Avg Distance to Target", "Distance (m)",
            COLORS["dist_raw"], COLORS["dist_smooth"],
        )
    else:
        axes[0, 1].text(0.5, 0.5, "No data", ha="center", va="center", transform=axes[0, 1].transAxes)

    # Panel 3 — Success rate
    if successes is not None:
        window = 50
        eps = np.arange(1, len(successes) + 1)
        running_rate = np.array(
            [np.mean(successes[max(0, i - window): i + 1]) * 100 for i in range(len(successes))]
        )
        axes[0, 2].plot(eps, running_rate, color=COLORS["success"], linewidth=2.0)
        axes[0, 2].set_xlabel("Episode")
        axes[0, 2].set_ylabel("Success Rate (%)")
        axes[0, 2].set_title("Success Rate (rolling 50-ep)", fontweight="bold", fontsize=11)
        axes[0, 2].set_ylim(-5, 105)
        axes[0, 2].grid(True)
    else:
        axes[0, 2].text(0.5, 0.5, "No data", ha="center", va="center", transform=axes[0, 2].transAxes)

    # Panel 4 — Actor loss
    if actor_losses is not None:
        _plot_metric(
            axes[1, 0], actor_losses,
            "Actor Loss", "Loss",
            COLORS["actor_loss"], COLORS["actor_loss"],
        )
    else:
        axes[1, 0].text(0.5, 0.5, "No data", ha="center", va="center", transform=axes[1, 0].transAxes)

    # Panel 5 — Critic loss
    if critic_losses is not None:
        _plot_metric(
            axes[1, 1], critic_losses,
            "Critic Loss", "Loss",
            COLORS["critic_loss"], COLORS["critic_loss"],
        )
    else:
        axes[1, 1].text(0.5, 0.5, "No data", ha="center", va="center", transform=axes[1, 1].transAxes)

    # Panel 6 — Alpha
    if alphas is not None:
        eps = np.arange(1, len(alphas) + 1)
        axes[1, 2].plot(eps, alphas, color=COLORS["alpha"], linewidth=2.0)
        axes[1, 2].set_xlabel("Episode")
        axes[1, 2].set_ylabel("α")
        axes[1, 2].set_title("Entropy Coefficient (α)", fontweight="bold", fontsize=11)
        axes[1, 2].grid(True)
    else:
        axes[1, 2].text(0.5, 0.5, "No data", ha="center", va="center", transform=axes[1, 2].transAxes)

    fig.suptitle(
        "SAC Training Dashboard — Robotic Arm Reaching Task",
        fontsize=14, fontweight="bold", color="#ffffff", y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
    print(f"  ✓ Saved: {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    print("=" * 50)
    print("  Generating Training Dashboard")
    print("=" * 50)

    generate_dashboard()

    print("=" * 50)
    print("  Done! Open training_dashboard.png to view.")
    print("=" * 50)
