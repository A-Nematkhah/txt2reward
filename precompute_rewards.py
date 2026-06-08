"""
Run this ONCE before training to build the full reward lookup table.
Queries Qwen for every (speed_bin, lane, distance_bin) combination
and saves the result to reward_cache.pkl.

Total unique states: ~10 speed bins × 4 lanes × 11 distance bins = 440 queries.
At ~2 sec/query on T4 → ~15 minutes total, then training is cache-only (fast).
"""

import os
import pickle
import itertools
import time

# Must set before importing llm_judge
os.environ.setdefault("MODEL_PATH", "/content/drive/MyDrive/models/qwen3-4b")

from llm_judge import query_qwen, discretize, CACHE_FILE

CACHE_FILE = "reward_cache.pkl"

# ── Define the full state space ───────────────────────────────────────────────
# speed_bin  : 0-9  → represents speeds 0-5, 5-10, ..., 45-50 km/h
# lane       : 0-3
# distance_bin: 0-10 → represents 0-10, 10-20, ..., 100+ m

speed_bins    = range(0, 10)   # bin * 5 = centre speed
lanes         = range(0, 4)
distance_bins = range(0, 11)

all_states = list(itertools.product(speed_bins, lanes, distance_bins))
total = len(all_states)
print(f"Total states to precompute: {total}")

# ── Load existing cache (resume if interrupted) ───────────────────────────────
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "rb") as f:
        cache = pickle.load(f)
    print(f"Resuming — {len(cache)} states already cached.")
else:
    cache = {}

# ── Query missing states ──────────────────────────────────────────────────────
start = time.time()
new_queries = 0

for i, (sb, lane, db) in enumerate(all_states):
    key = (sb, lane, db)
    if key in cache:
        continue

    # Convert bins back to representative values for the LLM prompt
    speed_kmh = sb * 5 + 2.5        # bin midpoint
    front_dist = min(db * 10 + 5, 95.0)  # bin midpoint, capped

    reward = query_qwen(speed_kmh, lane, front_dist)
    cache[key] = reward
    new_queries += 1

    elapsed = time.time() - start
    remaining = (total - i - 1) * (elapsed / new_queries) if new_queries else 0
    print(
        f"[{i+1:>3}/{total}]  "
        f"state=({sb},{lane},{db})  reward={reward:+.3f}  "
        f"eta: {remaining/60:.1f} min",
        flush=True,
    )

    # Save checkpoint every 20 new queries
    if new_queries % 20 == 0:
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(cache, f)
        print(f"  ✓ checkpoint saved ({len(cache)} entries)", flush=True)

# ── Final save ────────────────────────────────────────────────────────────────
with open(CACHE_FILE, "wb") as f:
    pickle.dump(cache, f)

print(f"\nDone. {len(cache)} states in cache → {CACHE_FILE}")
print("You can now run train.py — zero LLM calls during training.")
