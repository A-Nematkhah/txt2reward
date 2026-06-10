import numpy as np
import gymnasium as gym
from llm_judge import judge_state


class LLMRewardWrapper(gym.Wrapper):
    """
    Gym wrapper که هر `llm_interval` استپ یه LLM bonus به reward اضافه میکنه.

    تغییرات نسبت به نسخه قبل:
    - step_count موقع reset ریست میشه (قبلاً نمیشد)
    - observation parsing با normalize=True/False کار میکنه
    - front_distance با clip محافظت شده
    - LLM call async هست → training loop هیچوقت بلاک نمیشه
    """

    def __init__(self, env: gym.Env, llm_interval: int = 50):
        super().__init__(env)
        self.llm_interval = llm_interval
        self.step_count   = 0

    def reset(self, **kwargs):
        self.step_count = 0          # ← fix: قبلاً ریست نمیشد
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1

        if self.step_count % self.llm_interval == 0:
            speed, lane, front_dist = _parse_obs(obs)
            llm_bonus = judge_state(speed, lane, front_dist)
            reward   += llm_bonus

        return obs, reward, terminated, truncated, info


# ── observation parsing ───────────────────────────────────────────────────────

# highway-v0 با KinematicObservation پیش‌فرض:
# هر ردیف: [presence, x, y, vx, vy]  (normalize=True: همه بین [-1,1] یا [0,1])
# ego vehicle همیشه ردیف ۰ هست
_IDX_X  = 1   # موقعیت x (نرمالایز‌شده یا خام)
_IDX_Y  = 2   # موقعیت y / lane
_IDX_VX = 3   # سرعت طولی

# highway-v0 محور x رو نرمالایز میکنه به [0,1] با absolute=False (پیش‌فرض)
# سرعت نرمالایز‌شده رو باید به km/h برگردونیم
_SPEED_SCALE    = 40.0   # تقریباً max_speed پیش‌فرض (m/s→ normalize)
_KMH_FACTOR     = 3.6
_LANE_WIDTH     = 4.0    # متر — برای تبدیل y نرمالایز به lane index
_NUM_LANES      = 4


def _parse_obs(obs: np.ndarray) -> tuple[float, int, float]:
    """
    از Kinematics observation: (speed_kmh, lane_index, front_distance) برمیگردونه.
    هم normalize=True هم normalize=False رو handle میکنه.
    """
    ego = obs[0]

    # ── سرعت ─────────────────────────────────────────────────────────────────
    vx_raw = float(ego[_IDX_VX])
    # اگه normalize=True باشه vx_raw بین [-1,1]، وگرنه متر بر ثانیه‌ست
    # هر دو حالت رو handle میکنیم: اگه abs(vx) <= 1.5 احتمالاً نرمالایزه
    if abs(vx_raw) <= 1.5:
        speed_ms  = vx_raw * _SPEED_SCALE
    else:
        speed_ms  = vx_raw
    speed_kmh = max(0.0, speed_ms * _KMH_FACTOR)

    # ── lane index ────────────────────────────────────────────────────────────
    y_raw = float(ego[_IDX_Y])
    if abs(y_raw) <= 1.5:
        # نرمالایز: y در [0,1] → lane
        lane = int(np.clip(round(y_raw * (_NUM_LANES - 1)), 0, _NUM_LANES - 1))
    else:
        lane = int(np.clip(round(y_raw / _LANE_WIDTH), 0, _NUM_LANES - 1))

    # ── فاصله از جلوترین ماشین ───────────────────────────────────────────────
    front_dist = _get_front_distance(obs)

    return speed_kmh, lane, front_dist


def _get_front_distance(obs: np.ndarray) -> float:
    """نزدیک‌ترین ماشین جلویی رو پیدا میکنه."""
    ego_x = float(obs[0][_IDX_X])
    best  = 50.0   # fallback

    for i in range(1, len(obs)):
        if float(obs[i][0]) < 0.5:   # presence < 0.5 → ماشین وجود نداره
            continue
        dx = float(obs[i][_IDX_X]) - ego_x
        if 0.0 < dx < best:
            best = dx

    # اگه نرمالایز بود dx خیلی کوچیکه؛ به متر تبدیل کن
    # highway-env x رو روی ~[-1,1] نرمالایز میکنه با بازه ~100m
    if best < 2.0:
        best = best * 100.0   # approximate: 1 unit ≈ 100m در normalized mode

    return float(np.clip(best, 0.0, 200.0))
