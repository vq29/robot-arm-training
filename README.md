# 🤖 SAC Robotic Arm — Reinforcement Learning Target Reaching

A **Soft Actor-Critic (SAC)** agent trained in **PyTorch** to control a 3-joint robotic arm in **MuJoCo**, learning to reach randomly placed targets in 3D space.

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)
![MuJoCo](https://img.shields.io/badge/MuJoCo-3.0+-4A90D9)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🎯 Project Overview

This project implements a **continuous control** RL pipeline where a simulated robotic arm learns to reach a randomly spawned target. The agent uses **Soft Actor-Critic**, a state-of-the-art off-policy algorithm that maximizes both expected reward and policy entropy for robust exploration.

### Why SAC?

| Feature | DDPG | **SAC** ✓ |
|---|---|---|
| Policy type | Deterministic | **Stochastic** (Gaussian) |
| Exploration | External noise (OU) | **Built-in** (entropy bonus) |
| Overestimation | Single critic | **Twin critics** (clipped double-Q) |
| Temperature | Manual tuning | **Automatic** (learned α) |
| Stability | Sensitive to hyperparams | **More robust** |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│                  Training Loop                  │
│                  (train.py)                      │
│                                                 │
│  ┌──────────┐    action    ┌──────────────────┐ │
│  │          │─────────────►│                  │ │
│  │   SAC    │              │   MuJoCo Env     │ │
│  │  Agent   │◄─────────────│   (3-joint arm)  │ │
│  │          │  obs, reward │                  │ │
│  └──────────┘              └──────────────────┘ │
│       │                                         │
│       ▼                                         │
│  ┌──────────┐                                   │
│  │  Replay  │                                   │
│  │  Buffer  │                                   │
│  └──────────┘                                   │
└─────────────────────────────────────────────────┘

SAC Agent internals:
┌───────────────────────────────────────┐
│  Gaussian Actor (π)                   │
│  state → (μ, σ) → tanh(sample)       │
├───────────────────────────────────────┤
│  Twin Critics (Q₁, Q₂)               │
│  (state, action) → Q-value           │
├───────────────────────────────────────┤
│  Auto Entropy (α)                     │
│  learned log(α) → balances explore    │
└───────────────────────────────────────┘
```

---

## 📁 Project Structure

```
robot_project/
├── config.py            # Centralized hyperparameters (@dataclass)
├── agent.py             # SAC: GaussianActor, twin QNetworks, auto-α
├── env.py               # MuJoCo environment with shaped reward
├── train.py             # Training loop + TensorBoard logging
├── replay_buffer.py     # Ring buffer with uniform sampling
├── visualize.py         # 6-panel training dashboard
├── demo_trained.py      # Run the trained policy (deterministic)
├── demo_untrained.py    # Random-action baseline
├── robot_arm.xml        # MuJoCo robot model (MJCF)
├── robot_arm.urdf       # URDF description (reference)
├── requirements.txt     # Python dependencies
├── models/              # Saved model weights
│   ├── actor.pth
│   ├── critic1.pth
│   ├── critic2.pth
│   └── log_alpha.pth
└── logs/                # Training histories + TensorBoard
    ├── reward_history.npy
    ├── distance_history.npy
    ├── success_history.npy
    ├── actor_loss_history.npy
    ├── critic_loss_history.npy
    ├── alpha_history.npy
    └── tensorboard/
```

---

## 🚀 Setup & Usage

### 1. Create virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the agent

```bash
# Default: 1000 episodes with GUI
python train.py

# Custom settings
python train.py --episodes 500 --batch-size 128

# Headless (faster, no rendering)
python train.py --no-gui --episodes 2000
```

### 4. Monitor training (TensorBoard)

```bash
tensorboard --logdir=logs/tensorboard
```

Open http://localhost:6006 to see real-time reward, distance, success rate, and loss curves.

### 5. Visualize results

```bash
python visualize.py
# → Outputs: training_dashboard.png (6-panel figure)
```

### 6. Demo

```bash
# Trained agent (deterministic policy)
python demo_trained.py --targets 5

# Untrained baseline (random actions)
python demo_untrained.py
```

---

## 🧠 Algorithm Details

### Soft Actor-Critic (SAC)

SAC optimizes a **maximum entropy** objective:

```
J(π) = Σ E[ r(s,a) + α · H(π(·|s)) ]
```

- **Actor**: Gaussian policy outputs (μ, σ), samples via reparameterization trick, squashes with tanh
- **Twin Critics**: Two independent Q-networks; take the minimum to reduce overestimation
- **Entropy Temperature (α)**: Automatically learned to balance exploration vs exploitation

### Reward Shaping

```
reward = -distance - 0.1 × ||velocities|| + 10.0 × (reached target)
```

- **Distance penalty**: Drives the end-effector toward the target
- **Velocity penalty**: Encourages smooth, efficient motions
- **Reach bonus**: +10 reward spike when within 5 cm of target (triggers early episode end)

---

## 📊 Results

After training, run `python visualize.py` to generate the training dashboard. Key metrics:

- **Episode Reward** — should increase over training
- **Distance to Target** — should decrease toward 0
- **Success Rate** — percentage of episodes reaching within 5 cm
- **Entropy (α)** — shows the agent reducing randomness as it learns

---

## 🔧 Hyperparameters

All hyperparameters are centralized in `config.py`:

| Parameter | Value | Description |
|---|---|---|
| `actor_lr` | 3e-4 | Actor learning rate |
| `critic_lr` | 3e-4 | Critic learning rate |
| `alpha_lr` | 3e-4 | Entropy temperature LR |
| `gamma` | 0.99 | Discount factor |
| `tau` | 0.005 | Soft-update coefficient |
| `hidden_dim` | 256 | Network hidden layer size |
| `batch_size` | 256 | Training batch size |
| `buffer_capacity` | 1M | Replay buffer size |

---

## 💡 Key Learnings

- **SAC vs DDPG**: SAC's stochastic policy and entropy bonus provide much more stable training than DDPG's deterministic policy + OU noise
- **Reward shaping matters**: Adding velocity penalty and reach bonus significantly accelerated convergence
- **Automatic entropy tuning**: Eliminates the need to manually tune the exploration-exploitation trade-off
- **Twin critics**: The clipped double-Q trick effectively prevents Q-value overestimation

---

## 📚 References

- Haarnoja et al., [Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor](https://arxiv.org/abs/1801.01290) (2018)
- Haarnoja et al., [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905) (2019)
- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- [PyTorch Documentation](https://pytorch.org/docs/)

---

## 📄 License

This project is open source under the [MIT License](LICENSE).
