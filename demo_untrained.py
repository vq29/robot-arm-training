"""
Demo: Untrained robotic arm with random torques.
Opens MuJoCo GUI showing the arm flailing randomly with a visible target sphere.

Usage:
    python demo_untrained.py
"""

import time
import numpy as np
from env import RobotArmEnv


def run_untrained_demo(duration_seconds=8):
    """Show untrained arm applying random torques."""
    print("=" * 50)
    print("  Demo: UNTRAINED Arm (Random Torques)")
    print("=" * 50)
    print("  Watch the arm flail around randomly.")
    print("  The red sphere is the target position.")
    print("  Close the MuJoCo window to exit.\n")

    env = RobotArmEnv(gui=True, max_steps=10000)
    state = env.reset()

    steps = int(duration_seconds * 240)  # 240 Hz sim
    start_time = time.time()

    for i in range(steps):
        # Random action every 10 sim steps (for smoother motion)
        if i % 10 == 0:
            action = np.random.uniform(-1, 1, size=3).astype(np.float32)

        state, reward, done, info = env.step(action)

        # Slow down to ~real-time
        elapsed = time.time() - start_time
        expected = (i + 1) / 240.0
        if expected > elapsed:
            time.sleep(expected - elapsed)

        if done:
            state = env.reset()

    print(f"\n  Demo finished after {duration_seconds} seconds.")
    env.close()


if __name__ == "__main__":
    run_untrained_demo()
