import gymnasium as gym
from llm_judge import judge_state

class LLMRewardWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.step_count = 0

    def step(self, action):

        obs, reward, terminated, truncated, info = self.env.step(action)

        self.step_count += 1

        ego = obs[0]

        speed = float(ego[2])
        lane = int(ego[1])

        front_distance = 50.0

        if self.step_count % 50 == 0:
            reward += judge_state(
                speed,
                lane,
                front_distance
            )

        return (
            obs,
            reward,
            terminated,
            truncated,
            info
        )
