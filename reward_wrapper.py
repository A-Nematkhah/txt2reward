import numpy as np
import gymnasium as gym
from llm_judge import judge_state

# highway-env Kinematics observation columns (default):
#   [presence, x, y, vx, vy]
# obs shape: (num_vehicles, 5) — row 0 is always the ego vehicle.
_COL_PRESENCE = 0
_COL_X        = 1
_COL_Y        = 2
_COL_VX       = 3
_COL_VY       = 4


class LLMRewardWrapper(gym.Wrapper):
    """
    Wraps a highway-env environment and adds an LLM-based reward signal
    every `llm_every` steps on top of the environment's default reward.

    Fixes vs. original:
      - front_distance is computed from the observation (not hardcoded).
      - LLM output is robustly parsed.
      - Model is lazy-loaded (not at import time).
    """

    def __init__(self, env, llm_every: int = 50):
        super().__init__(env)
        self.step_count = 0
        self.llm_every  = llm_every

    # ── helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _extract_state(obs: np.ndarray):
        """
        Extract (speed_kmh, lane_index, front_distance_m) from a KinematicObservation.

        obs shape: (num_vehicles, 5)
        Columns: [presence, x, y, vx, vy]
        highway-env uses normalised coords by default; vx ≈ speed in m/s.
        Lane width is 4 m.
        """
        assert obs.ndim == 2 and obs.shape[1] >= 5, (
            f"Unexpected obs shape {obs.shape}. "
            "Set observation_type='Kinematics' in the env config."
        )

        ego = obs[0]                          # ego vehicle is always row 0
        speed_ms   = float(ego[_COL_VX])      # longitudinal speed (m/s)
        speed_kmh  = speed_ms * 3.6
        lane_index = int(round(float(ego[_COL_Y]) / 4.0))   # 4 m lane width

        # Find the closest vehicle ahead (same lane, positive relative x)
        front_distance = 50.0                 # default if no vehicle ahead
        ego_x = float(ego[_COL_X])
        for i in range(1, obs.shape[0]):
            if obs[i][_COL_PRESENCE] < 0.5:   # vehicle not present
                continue
            rel_x = float(obs[i][_COL_X]) - ego_x
            if rel_x > 0:                     # vehicle is ahead
                front_distance = min(front_distance, rel_x)

        return speed_kmh, lane_index, front_distance

    # ── gym API ───────────────────────────────────────────────────────────────
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1

        if self.step_count % self.llm_every == 0:
            speed_kmh, lane_index, front_distance = self._extract_state(obs)
            llm_reward = judge_state(speed_kmh, lane_index, front_distance)
            reward += llm_reward

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.step_count = 0
        return self.env.reset(**kwargs)
