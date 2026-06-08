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
    Optimized version:
    - Fully vectorized state extraction
    - No Python loops per step
    - LLM called sparsely and safely
    """

    def __init__(self, env, llm_every: int = 200):
        super().__init__(env)
        self.step_count = 0
        self.llm_every = llm_every
        self.cached_llm_reward = 0.0

    def _extract_state(self, obs: np.ndarray):
        """
        Vectorized extraction of:
        - speed_kmh
        - lane_index
        - front_distance
        """

        ego = obs[0]

        # speed
        speed_kmh = float(ego[3]) * 3.6

        # lane index (approx)
        lane_index = int(np.round(float(ego[2]) / 4.0))

        ego_x = float(ego[1])

        # vectorized vehicles (excluding ego)
        others = obs[1:]
        present = others[:, 0] > 0.5

        others = others[present]

        if len(others) == 0:
            front_distance = 50.0
        else:
            rel_x = others[:, 1] - ego_x
            front_ahead = rel_x[rel_x > 0]

            front_distance = float(np.min(front_ahead)) if len(front_ahead) > 0 else 50.0

        return speed_kmh, lane_index, front_distance

    def _maybe_update_llm(self, speed, lane, dist):
        """
        LLM is NOT in the critical loop anymore.
        It updates sparsely and caches result.
        """

        if self.step_count % self.llm_every != 0:
            return self.cached_llm_reward

        # expensive call isolated
        self.cached_llm_reward = float(
            judge_state(speed, lane, dist)
        )

        return self.cached_llm_reward

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1

        # cheap vectorized state extraction
        speed, lane, dist = self._extract_state(obs)

        # base reward + cached LLM reward
        llm_r = self._maybe_update_llm(speed, lane, dist)

        reward = reward + llm_r

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.step_count = 0
        self.cached_llm_reward = 0.0
        return self.env.reset(**kwargs)
        self.step_count = 0
        return self.env.reset(**kwargs)
