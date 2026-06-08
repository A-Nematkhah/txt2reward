import gymnasium as gym
import highway_env  # noqa: F401 — registers highway-v0
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from reward_wrapper import LLMRewardWrapper

# ── Environment config ────────────────────────────────────────────────────────
ENV_CONFIG = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 10,
        "features": ["presence", "x", "y", "vx", "vy"],
        "normalize": False,   # raw values so _extract_state works correctly
    },
    "action": {"type": "DiscreteMetaAction"},
    "lanes_count": 4,
    "vehicles_count": 20,
    "duration": 40,          # seconds per episode
    "reward_speed_range": [20, 30],
}


def make_env():
    env = gym.make("highway-v0", config=ENV_CONFIG)
    env = LLMRewardWrapper(env, llm_every=50)
    return env


# ── Training ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    env = make_env()

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device="cuda",          # switch to "cpu" if no GPU
        n_steps=512,
        batch_size=64,
        learning_rate=3e-4,
        tensorboard_log="./tb_logs/",
    )

    model.learn(total_timesteps=100_000)
    model.save("ppo_highway_qwen_reward")
    print("Model saved to ppo_highway_qwen_reward.zip")

    env.close()
