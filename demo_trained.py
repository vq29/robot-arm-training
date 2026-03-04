"""
Demo: Trained robotic arm reaching targets using learned SAC policy.
Loads the saved model and demonstrates smooth target-reaching behaviour.

Usage:
    python demo_trained.py
    python demo_trained.py --targets 5    # cycle through 5 targets
"""

import os
import time
import argparse
import numpy as np
from env import RobotArmEnv
from agent import SACAgent


def run_trained_demo(num_targets=3, steps_per_target=500):
    """Load trained SAC model and demonstrate reaching behaviour."""
    print("=" * 50)
    print("  Demo: TRAINED Arm (SAC Policy)")
    print("=" * 50)

    model_dir = "models"
    if not os.path.exists(os.path.join(model_dir, "actor.pth")):
        print("  ERROR: No trained model found in models/")
        print("  Run 'python train.py' first to train the agent.")
        return

    env = RobotArmEnv(gui=True, max_steps=steps_per_target)
    agent = SACAgent(state_dim=env.state_dim, action_dim=env.action_dim)
    agent.load(model_dir)
    print(f"  Loaded SAC model from {model_dir}/")
    print(f"  Showing {num_targets} target reach(es).")
    print("  Close the MuJoCo window to exit.\n")

    successes = 0

    for t in range(1, num_targets + 1):
        state = env.reset()
        print(
            f"  Target {t}/{num_targets} — pos: [{env.target_pos[0]:.2f}, "
            f"{env.target_pos[1]:.2f}, {env.target_pos[2]:.2f}]"
        )

        start_time = time.time()
        target_reached = False

        for step in range(steps_per_target):
            action = agent.select_action(state, deterministic=True)
            state, reward, done, info = env.step(action)

            if info["success"]:
                target_reached = True

            # Slow down to ~real-time
            elapsed = time.time() - start_time
            expected = (step + 1) / 240.0
            if expected > elapsed:
                time.sleep(expected - elapsed)

            if done:
                break

        final_dist = info["distance"]
        status = "✓ REACHED" if target_reached else "✗ missed"
        print(f"    → {status}  (final distance: {final_dist:.4f})")

        if target_reached:
            successes += 1

        time.sleep(0.5)

    print(f"\n  Demo complete! {successes}/{num_targets} targets reached.")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo trained robotic arm (SAC)")
    parser.add_argument(
        "--targets", type=int, default=3, help="Number of targets to reach"
    )
    args = parser.parse_args()

    run_trained_demo(num_targets=args.targets)
