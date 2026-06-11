import os
import re
import pickle
import time
import threading
import concurrent.futures
from groq import Groq
from prompts import SYSTEM_PROMPT

# ── API key ───────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    raise EnvironmentError(
        "\n[ERROR] GROQ_API_KEY is not set.\n"
        "In Colab, run this before importing:\n"
        '  import os; os.environ["GROQ_API_KEY"] = "gsk_xxxxxxxx"'
    )

client = Groq(api_key=GROQ_API_KEY)
MODEL  = "llama-3.3-70b-versatile"

# Groq free tier: 30 req/min = 1 req per 2s.
# Use 2.5s to stay safely below the limit.
_RATE_LIMIT_DELAY = 2.5   # seconds between requests in prefill_cache_sync

# ── Persistent cache (thread-safe) ────────────────────────────────────────────
CACHE_FILE  = "reward_cache.pkl"
_cache_lock = threading.Lock()


def _load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            data = pickle.load(f)
        print(f"[cache] Loaded {len(data)} entries from {CACHE_FILE}")
        return data
    return {}


# Module-level cache dict — shared by both training and precompute
reward_cache: dict = _load_cache()


def _save_cache() -> None:
    with _cache_lock:
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(reward_cache, f)


# ── Async executor (non-blocking Groq calls during training) ──────────────────
_executor     = concurrent.futures.ThreadPoolExecutor(max_workers=2)
_pending: dict = {}        # key → Future
_pending_lock  = threading.Lock()


# ── Helpers ───────────────────────────────────────────────────────────────────

def discretize(speed: float, lane: int, distance: float) -> tuple:
    """Map a continuous state to a discrete cache key."""
    return (int(speed // 5), int(lane), int(min(distance, 50) // 10))


def _call_groq_once(user_msg: str) -> str | None:
    """
    Send a single request to Groq.
    Returns the raw response text, or None on a non-rate-limit error.
    Raises a RateLimitError string on HTTP 429 so callers can retry.
    """
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens=5,
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        if "rate_limit" in str(e).lower() or "429" in str(e):
            raise RuntimeError("rate_limit") from e
        print(f"[groq] Non-rate-limit error: {e}")
        return None


def _parse_score(text: str | None) -> float:
    """Convert a 0-5 digit string to a reward in [-1, +1]."""
    if text is None:
        return 0.0
    match = re.search(r"[0-5]", text)
    score = int(match.group()) if match else 3
    return (score - 2.5) / 2.5


def _query_groq_async(speed: float, lane: int, distance: float) -> float:
    """
    Best-effort Groq query used by the async background worker during training.
    Retries up to 3 times on rate limit, then returns 0.0.
    A 0.0 fallback here is acceptable — it just means no LLM bonus this step.
    """
    user_msg = (
        f"Speed:{speed:.0f} Lane:{lane} FrontDistance:{distance:.0f}\n"
        "Reply with ONLY a single digit 0-5. No explanation."
    )
    for attempt in range(3):
        try:
            text = _call_groq_once(user_msg)
            return _parse_score(text)
        except RuntimeError:
            wait = 2 ** attempt     # 1s, 2s, 4s
            print(f"[groq] Rate limit — backing off {wait}s")
            time.sleep(wait)
    return 0.0


def _query_groq_blocking(speed: float, lane: int, distance: float) -> float:
    """
    Reliable Groq query used by prefill_cache_sync.
    Retries indefinitely on rate limit with increasing back-off.
    Never returns 0.0 due to a rate limit — it always waits for a real answer.
    """
    user_msg = (
        f"Speed:{speed:.0f} Lane:{lane} FrontDistance:{distance:.0f}\n"
        "Reply with ONLY a single digit 0-5. No explanation."
    )
    attempt = 0
    while True:
        try:
            text = _call_groq_once(user_msg)
            if text is not None:
                return _parse_score(text)
            # Non-rate-limit error → fall through to a short wait and retry
        except RuntimeError:
            # Rate limit: wait longer with each consecutive hit, cap at 60s
            wait = min(60, 5 * (2 ** attempt))
            print(f"[groq] Rate limit — waiting {wait}s before retry ...")
            time.sleep(wait)
            attempt += 1
            continue
        # Non-rate-limit error: short wait then retry
        time.sleep(2.0)


def _fetch_and_cache(key: tuple, speed: float, lane: int, distance: float) -> float:
    """Background worker: query Groq (best-effort) and write to the shared cache."""
    reward = _query_groq_async(speed, lane, distance)
    with _cache_lock:
        reward_cache[key] = reward
        if len(reward_cache) % 50 == 0:
            _save_cache()
    return reward


# ── Public API ────────────────────────────────────────────────────────────────

def judge_state(speed: float, lane: int, front_distance: float) -> float:
    """
    Return the LLM reward for the given driving state.

    - Cache hit  : returns immediately (zero latency).
    - Cache miss : fires a background Future and returns 0.0 for this step.
                   The result is cached when the Future completes, so the
                   next visit to the same state gets an instant cache hit.

    The training loop is never blocked by a network call.
    """
    key = discretize(speed, lane, front_distance)

    # Fast path: state already cached
    with _cache_lock:
        if key in reward_cache:
            return reward_cache[key]

    # Check whether a request is already in-flight for this key
    with _pending_lock:
        if key in _pending:
            fut = _pending[key]
            if fut.done():
                del _pending[key]
                with _cache_lock:
                    return reward_cache.get(key, 0.0)
            return 0.0  # still in-flight — skip bonus this step

        # Launch a new background request
        fut = _executor.submit(_fetch_and_cache, key, speed, lane, front_distance)
        _pending[key] = fut

    return 0.0


def prefill_cache_sync(verbose: bool = True) -> None:
    """
    Synchronously pre-fill the cache with all 192 possible discretised states.
    Run this once before training so that training is fully offline (no HTTP calls).

    Rate-limit strategy:
      - Waits _RATE_LIMIT_DELAY seconds between every request (proactive throttle).
      - On a 429 response, backs off with increasing delays and retries until
        a real answer is received. Never writes a fallback 0.0 to the cache.

    State space:
      speed    : 0, 5, 10, ..., 35  (8 bins)
      lane     : 0, 1, 2, 3         (4 values)
      distance : 0, 10, 20, ..., 50 (6 bins)
      total    : 8 × 4 × 6 = 192
    """
    import itertools

    speeds    = range(0, 40, 5)
    lanes     = range(0, 4)
    distances = range(0, 60, 10)

    combos = list(itertools.product(speeds, lanes, distances))
    needed = [
        (s, la, d) for s, la, d in combos
        if discretize(s, la, d) not in reward_cache
    ]

    if not needed:
        print("[cache] Already complete — no pre-fill needed.")
        return

    total = len(needed)
    eta   = total * _RATE_LIMIT_DELAY
    print(f"[cache] Pre-filling {total} / {len(combos)} states "
          f"(~{eta:.0f}s at {_RATE_LIMIT_DELAY}s/req) ...")

    for idx, (speed, lane, dist) in enumerate(needed, 1):
        key    = discretize(speed, lane, dist)
        # Proactive delay — keeps us under 30 req/min without hitting 429
        if idx > 1:
            time.sleep(_RATE_LIMIT_DELAY)

        # This call retries indefinitely on rate-limit; never returns a fake 0.0
        reward = _query_groq_blocking(speed, lane, dist)

        with _cache_lock:
            reward_cache[key] = reward

        if idx % 20 == 0 or idx == total:
            _save_cache()
            if verbose:
                print(f"  [{idx:3d}/{total}] "
                      f"speed={speed:2d}  lane={lane}  dist={dist:2d}  "
                      f"reward={reward:+.3f}")

    _save_cache()
    print(f"[cache] Done — {len(reward_cache)} entries saved to '{CACHE_FILE}'.")
