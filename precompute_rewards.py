"""
precompute_rewards.py
─────────────────────
قبل از train.py اجرا کن.
تمام ترکیب‌های ممکن speed/lane/distance رو یک‌بار از Qwen می‌پرسه
و توی reward_cache.pkl ذخیره می‌کنه.
اینطوری حین training هیچ inference ای انجام نمیشه.
"""

import pickle
import itertools
from llm_judge import query_qwen, CACHE_FILE

# ── محدوده مقادیر ممکن در highway-env ────────────────────────────────────────
SPEED_RANGE    = range(0, 40, 5)    # 0, 5, 10, … 35
LANE_RANGE     = range(0, 4)        # 0, 1, 2, 3
DISTANCE_RANGE = range(0, 60, 10)   # 0, 10, 20, … 50

combos = list(itertools.product(SPEED_RANGE, LANE_RANGE, DISTANCE_RANGE))
total  = len(combos)
print(f"▶ Pre-computing {total} reward entries …\n")

cache = {}
for idx, (speed, lane, dist) in enumerate(combos):
    key    = (speed // 5, int(lane), dist // 10)
    reward = query_qwen(speed, lane, dist)
    cache[key] = reward

    if (idx + 1) % 10 == 0 or (idx + 1) == total:
        print(f"  [{idx+1}/{total}]  speed={speed}  lane={lane}  "
              f"dist={dist}  → reward={reward:.3f}")

with open(CACHE_FILE, "wb") as f:
    pickle.dump(cache, f)

print(f"\n✅ Done! {len(cache)} entries saved to '{CACHE_FILE}'")
print("   حالا می‌تونی train.py رو اجرا کنی.")
