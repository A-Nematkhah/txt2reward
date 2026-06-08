import gymnasium as gym
from llm_judge import judge_state

class LLMRewardWrapper(gym.Wrapper):
    def __init__(self, env, llm_interval=50):
        super().__init__(env)
        self.step_count  = 0
        self.llm_interval = llm_interval  # چند استپ یک‌بار LLM صدا بزنیم

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1

        # ── parse ego vehicle ─────────────────────────────────────────────────
        ego            = obs[0]
        speed          = float(ego[2])
        lane           = int(ego[1])
        # فاصله واقعی از جلوترین ماشین (اگه موجود باشه)
        front_distance = _get_front_distance(obs)

        # ── LLM reward هر llm_interval استپ ──────────────────────────────────
        if self.step_count % self.llm_interval == 0:
            llm_bonus = judge_state(speed, lane, front_distance)
            reward   += llm_bonus

        return obs, reward, terminated, truncated, info


def _get_front_distance(obs):
    """فاصله به نزدیک‌ترین ماشین جلویی رو از observation برگردون"""
    ego_x = float(obs[0][0])
    best  = 50.0  # مقدار پیش‌فرض

    for i in range(1, len(obs)):
        veh_x = float(obs[i][0])
        dx    = veh_x - ego_x
        if 0 < dx < best:
            best = dx

    return best
