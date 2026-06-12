"""
evaluate.py
───────────
ارزیابی مدل PPO آموزش‌دیده روی highway-v0 با LLM reward.

نحوه استفاده:
  python evaluate.py --model ppo_highway_qwen_reward.zip
  python evaluate.py --model ppo_highway_qwen_reward.zip --episodes 20 --render
  python evaluate.py --model ppo_highway_qwen_reward.zip --no-llm --episodes 10

خروجی‌ها:
  - آمار episode (reward، طول، crash)
  - میانگین و انحراف معیار کلی
  - ذخیره نتایج در evaluate_results.json  (با --save)
"""

import os
import json
import argparse
import numpy as np
import gymnasium as gym
import highway_env  # noqa: F401 — registers highway-v0
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from reward_wrapper import LLMRewardWrapper

# همان config که در train.py استفاده شده
ENV_CONFIG = {
    "vehicles_count":       15,
    "simulation_frequency":  5,
    "policy_frequency":      1,
    "duration":             40,
    "lanes_count":           4,
    "observation": {
        "type":           "Kinematics",
        "vehicles_count":  5,
        "features":       ["presence", "x", "y", "vx", "vy"],
        "normalize":      True,
        "absolute":       False,
    },
    "action": {
        "type": "DiscreteMetaAction",
    },
    "reward_speed_range": [20, 30],
    "collision_reward":   -1.0,
    "high_speed_reward":   0.4,
    "right_lane_reward":   0.1,
    "lane_change_reward":  0.0,
}


def make_eval_env(use_llm: bool, llm_interval: int, render_mode: str | None):
    """یک محیط ارزیابی می‌سازد."""
    config = dict(ENV_CONFIG)
    if render_mode:
        config["render_mode"] = render_mode

    env = gym.make("highway-v0", config=config)
    if use_llm:
        env = LLMRewardWrapper(env, llm_interval=llm_interval)
    env = Monitor(env)
    return env


def run_episode(model, env, deterministic: bool = True) -> dict:
    """یک episode کامل را اجرا می‌کند و نتایج را برمی‌گرداند."""
    obs, info = env.reset()
    total_reward = 0.0
    steps        = 0
    crashed      = False

    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps        += 1

        # highway-env برخورد را در info["crashed"] ثبت می‌کند
        if info.get("crashed", False):
            crashed = True

        if terminated or truncated:
            break

    return {
        "total_reward": float(total_reward),
        "steps":        steps,
        "crashed":      crashed,
    }


def evaluate(
    model_path:   str,
    n_episodes:   int  = 10,
    use_llm:      bool = True,
    llm_interval: int  = 50,
    render:       bool = False,
    deterministic:bool = True,
    save_path:    str | None = None,
) -> dict:
    """
    مدل را روی n_episodes اپیزود ارزیابی می‌کند.

    Returns
    -------
    dict با کلیدهای:
        episodes       : لیست نتایج هر episode
        mean_reward    : میانگین کل reward
        std_reward     : انحراف معیار reward
        mean_steps     : میانگین طول episode
        crash_rate     : نسبت اپیزودهای با crash
    """
    # بارگذاری مدل
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"[evaluate] Model not found: '{model_path}'\n"
            "  ابتدا train.py را اجرا کنید تا مدل ذخیره شود."
        )

    print(f"[evaluate] Loading model: {model_path}")

    render_mode = "human" if render else None
    env = make_eval_env(use_llm, llm_interval, render_mode)

    # SB3 نیاز به VecEnv دارد، حتی برای ارزیابی
    vec_env = DummyVecEnv([lambda: env])
    model   = PPO.load(model_path, env=vec_env, device="cpu")

    print(
        f"[evaluate] Running {n_episodes} episodes | "
        f"LLM={'ON' if use_llm else 'OFF'} | "
        f"deterministic={deterministic}\n"
    )

    results = []
    for ep in range(1, n_episodes + 1):
        ep_result = run_episode(model, env, deterministic=deterministic)
        results.append(ep_result)
        print(
            f"  Episode {ep:3d}/{n_episodes} | "
            f"reward={ep_result['total_reward']:+7.3f} | "
            f"steps={ep_result['steps']:3d} | "
            f"crashed={'YES' if ep_result['crashed'] else ' no'}"
        )

    env.close()

    # آمار کلی
    rewards    = [r["total_reward"] for r in results]
    steps_list = [r["steps"]        for r in results]
    crashes    = [r["crashed"]      for r in results]

    summary = {
        "model_path":   model_path,
        "n_episodes":   n_episodes,
        "use_llm":      use_llm,
        "deterministic":deterministic,
        "episodes":     results,
        "mean_reward":  float(np.mean(rewards)),
        "std_reward":   float(np.std(rewards)),
        "min_reward":   float(np.min(rewards)),
        "max_reward":   float(np.max(rewards)),
        "mean_steps":   float(np.mean(steps_list)),
        "crash_rate":   float(np.mean(crashes)),
    }

    # نمایش خلاصه
    print("\n" + "─" * 55)
    print(f"  Mean reward  : {summary['mean_reward']:+.3f}  ± {summary['std_reward']:.3f}")
    print(f"  Min / Max    : {summary['min_reward']:+.3f}  /  {summary['max_reward']:+.3f}")
    print(f"  Mean steps   : {summary['mean_steps']:.1f}")
    print(f"  Crash rate   : {summary['crash_rate']*100:.1f}%  "
          f"({sum(crashes)}/{n_episodes} episodes)")
    print("─" * 55)

    # ذخیره نتایج
    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n[evaluate] Results saved to '{save_path}'")

    return summary


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained PPO model on highway-v0"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ppo_highway_qwen_reward.zip",
        help="Path to the saved PPO model (.zip)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes (default: 10)",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM reward wrapper during evaluation (env reward only)",
    )
    parser.add_argument(
        "--llm-interval",
        type=int,
        default=50,
        help="Steps between LLM reward queries (default: 50)",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the environment visually (requires a display)",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy instead of deterministic",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        metavar="PATH",
        help="Save evaluation results to a JSON file (e.g. evaluate_results.json)",
    )
    args = parser.parse_args()

    evaluate(
        model_path    = args.model,
        n_episodes    = args.episodes,
        use_llm       = not args.no_llm,
        llm_interval  = args.llm_interval,
        render        = args.render,
        deterministic = not args.stochastic,
        save_path     = args.save,
    )
