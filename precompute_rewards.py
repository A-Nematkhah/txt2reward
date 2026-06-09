"""
precompute_rewards.py
─────────────────────
قبل از train.py اجرا کن.
192 ترکیب speed/lane/distance رو از Groq می‌پرسه و cache می‌کنه.
"""

import os
import pickle
import itertools
import time
from llm_judge import query_groq, CACHE_FILE

SPEED_RANGE    = range(0, 40, 5)
LANE_RANGE     = range(0, 4)
DISTANCE_RANGE = range(0, 60, 10)

combos = list(itertools.product(SPEED_RANGE, LANE_RANGE, DISTANCE_RANGE))
total  = len(combos)
print(f"▶ Pre-computing {total} reward entries از Groq …\n")

cache = {}
for idx, (speed, lane, dist) in enumerate(combos):
    key    = (speed // 5, int(lane), dist // 10)

    # اگه از قبل تو cache بود skip کن (برای resume کردن)
    if key in cache:
        continue

    reward = query_groq(speed, lane, dist)
    cache[key] = reward

    # throttle: Groq free tier ~30 req/min → هر req یه کم صبر کن
    time.sleep(0.5)

    if (idx + 1) % 10 == 0 or (idx + 1) == total:
        print(f"  [{idx+1}/{total}]  speed={speed}  lane={lane}  "
              f"dist={dist}  → reward={reward:.3f}")

    # هر ۵۰ تا ذخیره کن (در صورت قطع شدن)
    if (idx + 1) % 50 == 0:
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(cache, f)
        print(f"  💾 checkpoint ذخیره شد ({len(cache)} entries)")

with open(CACHE_FILE, "wb") as f:
    pickle.dump(cache, f)

print(f"\n✅ Done! {len(cache)} entries saved to '{CACHE_FILE}'")
print("   حالا می‌تونی train.py رو اجرا کنی.")
