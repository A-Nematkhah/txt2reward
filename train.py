import time
import gymnasium as gym
import highway_env  # noqa: F401
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from reward_wrapper import LLMRewardWrapper

# ── Environment config ────────────────────────────────────────────────────────
ENV_CONFIG = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 10,
        "features": ["presence", "x", "y", "vx", "vy"],
        "normalize": False,
    },
    "action": {"type": "DiscreteMetaAction"},
    "lanes_count": 4,
    "vehicles_count": 20,
    "duration": 40,
    "reward_speed_range": [20, 30],
}


def make_env():
    env = gym.make("highway-v0", config=ENV_CONFIG)
    env = LLMRewardWrapper(env, llm_every=200)
    return env


# ── Keep-alive callback ───────────────────────────────────────────────────────
class KeepAliveCallback(BaseCallback):
    """Prints a progress line every `print_every` steps so Colab stays alive."""

    def __init__(self, print_every: int = 512):
        super().__init__()
        self.print_every = print_every
        self._start = time.time()

    def _on_step(self) -> bool:
        if self.num_timesteps % self.print_every == 0:
            elapsed = time.time() - self._start
            fps = self.num_timesteps / max(elapsed, 1)
            print(
                f"[step {self.num_timesteps:>7}]  "
                f"elapsed: {elapsed/60:.1f} min  "
                f"fps: {fps:.0f}",
                flush=True,
            )
        return True  # returning False would stop training


# ── Training ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    env = make_env()

    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,          # silenced — KeepAliveCallback handles output
        device="cpu",
        n_steps=512,        # kept small so first print comes within ~30 sec
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
        tensorboard_log="./tb_logs/",
    )

    model.learn(
        total_timesteps=100_000,
        callback=KeepAliveCallback(print_every=512),
    )

    model.save("ppo_highway_qwen_reward")
    print("Model saved → ppo_highway_qwen_reward.zip")

    env.close()
