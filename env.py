"""
MuJoCo environment for a 3-joint robotic arm reaching a target.
Gym-like interface: reset(), step(action), render(), close().

Reward shaping:
  - Distance penalty: -distance_to_target
  - Velocity penalty: -0.1 * ||joint_velocities||
  - Reach bonus: +10 if distance < threshold (triggers early done)
"""

import os
import math
import numpy as np
import mujoco
import mujoco.viewer
import time

from config import TrainingConfig


class RobotArmEnv:
    """
    State  (12-dim): [joint_angles (3), joint_velocities (3),
                      target_pos (3), vec_to_target (3)]
    Action  (3-dim): continuous torques in [-1, 1], scaled to joint limits.
    Reward: shaped reward encouraging fast, smooth reaching.
    """

    def __init__(self, gui=False, max_steps=200, config: TrainingConfig = None):
        self.cfg = config or TrainingConfig()
        self.gui = gui
        self.max_steps = max_steps
        self.step_count = 0
        self.prev_dist = None  # for approach reward

        # Load MuJoCo model
        xml_path = os.path.join(os.path.dirname(__file__), "robot_arm.xml")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Physics sub-stepping
        self.n_substeps = self.cfg.n_substeps

        # IDs for easier access
        self.ee_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site"
        )
        self.target_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target"
        )

        # Joint setup
        self.joint_indices = [0, 1, 2]
        self.num_joints = 3

        # Action scaling
        self.max_torque = self.cfg.max_torque

        # State / action dimensions
        self.state_dim = 12
        self.action_dim = 3

        # Viewer
        self.viewer = None
        if self.gui:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        self.target_pos = np.zeros(3)

    # ------------------------------------------------------------------
    #  Reset
    # ------------------------------------------------------------------
    def reset(self):
        """Reset the environment and return initial observation."""
        mujoco.mj_resetData(self.model, self.data)

        # Randomise starting joint positions slightly
        for i in range(self.num_joints):
            self.data.qpos[i] = np.random.uniform(-0.3, 0.3)
            self.data.qvel[i] = 0.0

        # Spawn target at random reachable position
        self._spawn_target()

        # Forward dynamics to update kinematics
        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0
        self.prev_dist = self._get_distance()

        if self.viewer:
            self.viewer.sync()

        return self._get_obs()

    # ------------------------------------------------------------------
    #  Step
    # ------------------------------------------------------------------
    def step(self, action):
        """
        Apply torques, simulate, return (obs, reward, done, info).
        action: np.array of shape (3,) in [-1, 1]
        """
        action = np.clip(action, -1.0, 1.0)
        torques = action * self.max_torque

        # Apply controls and sub-step physics
        self.data.ctrl[:] = torques
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1

        obs = self._get_obs()
        dist = self._get_distance()
        ee_pos = self._get_ee_pos()

        # ── Shaped reward ─────────────────────────────────────────────
        # 1. Distance penalty (main signal)
        reward = -dist

        # 2. Approach reward (bonus for getting closer vs previous step)
        if self.prev_dist is not None:
            approach = self.prev_dist - dist  # positive when closing in
            reward += 5.0 * approach
        self.prev_dist = dist

        # 3. Proximity bonus (continuous exponential shaping near target)
        reward += math.exp(-5.0 * dist)

        # 4. Velocity penalty (encourages smooth motion)
        vel_magnitude = np.linalg.norm(self.data.qvel[: self.num_joints])
        reward -= self.cfg.velocity_penalty * vel_magnitude

        # 5. Reach bonus (big reward when close enough)
        success = dist < self.cfg.reach_threshold
        if success:
            reward += 10.0

        # ── Done condition ────────────────────────────────────────────
        done = self.step_count >= self.max_steps or success

        info = {
            "distance": dist,
            "success": success,
            "ee_pos": ee_pos.tolist(),
        }

        if self.viewer:
            self.viewer.sync()

        return obs, reward, done, info

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------
    def _get_obs(self):
        """Return 12-dim observation."""
        angles = self.data.qpos[: self.num_joints].copy()
        velocities = self.data.qvel[: self.num_joints].copy()

        ee_pos = self._get_ee_pos()
        vec_to_target = self.target_pos - ee_pos

        obs = np.concatenate(
            [
                angles / np.pi,          # normalize to ~[-1, 1]
                velocities * 0.1,         # scale down
                self.target_pos,          # target in world frame
                vec_to_target,            # relative direction
            ]
        )
        return obs.astype(np.float32)

    def _get_distance(self):
        """Euclidean distance from end-effector to target."""
        ee_pos = self.data.site_xpos[self.ee_site_id]
        return float(np.linalg.norm(ee_pos - self.target_pos))

    def _get_ee_pos(self):
        """Return end-effector position."""
        return self.data.site_xpos[self.ee_site_id].copy()

    def _spawn_target(self):
        """Place target at random reachable position via Mocap."""
        r = np.random.uniform(self.cfg.target_r_min, self.cfg.target_r_max)
        theta = np.random.uniform(0, 2 * math.pi)
        z = np.random.uniform(self.cfg.target_z_min, self.cfg.target_z_max)
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        self.target_pos = np.array([x, y, z], dtype=np.float32)

        # Update Mocap body position (only 1 mocap body → index 0)
        self.data.mocap_pos[0] = self.target_pos

    def close(self):
        """Close viewer."""
        if self.viewer:
            self.viewer.close()


# ------------------------------------------------------------------
#  Quick test
# ------------------------------------------------------------------
if __name__ == "__main__":
    env = RobotArmEnv(gui=True)
    obs = env.reset()
    print(f"Initial obs shape: {obs.shape}")
    print(f"Initial obs: {obs}")

    print("Running simulation loop... Close the viewer window to stop.")
    while env.viewer.is_running():
        action = np.random.uniform(-0.5, 0.5, 3)
        obs2, reward, done, info = env.step(action)
        time.sleep(0.01)
        if done:
            env.reset()

    print(f"Final Info: {info}")
    env.close()
    print("Environment OK")
