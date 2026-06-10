import os
import re
import pickle
import time
import threading
import concurrent.futures
from groq import Groq
from prompts import SYSTEM_PROMPT

# ── API Key ──────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    raise EnvironmentError(
        "\n❌ GROQ_API_KEY تنظیم نشده!\n"
        "در Colab قبل از اجرا این رو اجرا کن:\n"
        '  import os; os.environ["GROQ_API_KEY"] = "gsk_xxxxxxxx"'
    )

client = Groq(api_key=GROQ_API_KEY)
MODEL  = "llama-3.3-70b-versatile"

# ── Cache (thread-safe) ───────────────────────────────────────────────────────
CACHE_FILE  = "reward_cache.pkl"
_cache_lock = threading.Lock()

def _load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            data = pickle.load(f)
        print(f"✅ Cache لود شد: {len(data)} entry")
        return data
    return {}

# cache به صورت module-level لود میشه؛ precompute هم همین dict رو آپدیت میکنه
reward_cache: dict = _load_cache()

def _save_cache() -> None:
    with _cache_lock:
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(reward_cache, f)

# ── Async executor برای non-blocking Groq calls ───────────────────────────────
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
# pending futures: key → Future
_pending: dict = {}
_pending_lock = threading.Lock()

# ── Helpers ───────────────────────────────────────────────────────────────────
def discretize(speed: float, lane: int, distance: float) -> tuple:
    """state رو به یه key قابل cache تبدیل میکنه."""
    return (int(speed // 5), int(lane), int(min(distance, 50) // 10))

def _query_groq(speed: float, lane: int, distance: float, retries: int = 3) -> float:
    """یه Groq call میزنه و reward بین [-1,+1] برمیگردونه. blocking."""
    user_msg = (
        f"Speed:{speed:.0f} Lane:{lane} FrontDistance:{distance:.0f}\n"
        "Reply with ONLY a single digit 0-5. No explanation."
    )
    for attempt in range(retries):
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
            text  = resp.choices[0].message.content.strip()
            match = re.search(r"[0-5]", text)
            score = int(match.group()) if match else 3
            return (score - 2.5) / 2.5          # نرمالایز به [-1, +1]

        except Exception as e:
            err = str(e)
            if "rate_limit" in err.lower() or "429" in err:
                wait = 2 ** attempt
                print(f"  ⚠️  Rate limit — {wait}s صبر می‌کنم...")
                time.sleep(wait)
            else:
                print(f"  ❌ Groq error: {e}")
                break

    return 0.0  # fallback

def _fetch_and_cache(key: tuple, speed: float, lane: int, distance: float) -> float:
    """در thread جدا اجرا میشه: Groq رو صدا میزنه و cache رو آپدیت میکنه."""
    reward = _query_groq(speed, lane, distance)
    with _cache_lock:
        reward_cache[key] = reward
        if len(reward_cache) % 50 == 0:
            _save_cache()
    return reward

# ── Public API ────────────────────────────────────────────────────────────────
def judge_state(speed: float, lane: int, front_distance: float) -> float:
    """
    reward برای state داده‌شده برمیگردونه.

    - Cache hit  → فوری برمیگردونه (zero latency)
    - Cache miss → اگه future در راهه: 0.0 برمیگردونه و future رو ادامه میده
                   اگه future نیست: یه future جدید میسازه و 0.0 برمیگردونه
                   وقتی future تموم شد: cache پر میشه، دفعه بعد فوری برمیگرده

    training loop هیچوقت بلاک نمیشه.
    """
    key = discretize(speed, lane, front_distance)

    # ── cache hit ─────────────────────────────────────────────────────────────
    with _cache_lock:
        if key in reward_cache:
            return reward_cache[key]

    # ── چک کن future در راهه ─────────────────────────────────────────────────
    with _pending_lock:
        if key in _pending:
            fut = _pending[key]
            if fut.done():
                del _pending[key]
                # نتیجه الان توی cache هست
                with _cache_lock:
                    return reward_cache.get(key, 0.0)
            # هنوز در راهه → 0.0 برگردون
            return 0.0

        # ── future جدید بساز ─────────────────────────────────────────────────
        fut = _executor.submit(_fetch_and_cache, key, speed, lane, front_distance)
        _pending[key] = fut

    return 0.0  # این step بدون LLM bonus میگذره؛ دفعه بعد cache hit داره


def prefill_cache_sync(verbose: bool = True) -> None:
    """
    همه‌ی 192 state ممکن رو sync پر میکنه.
    قبل از train.py صدا بزن تا training بدون هیچ Groq callی ران بشه.
    """
    import itertools
    speeds    = range(0, 40, 5)   # 8 مقدار
    lanes     = range(0, 4)       # 4 مقدار
    distances = range(0, 60, 10)  # 6 مقدار  →  192 ترکیب

    combos = list(itertools.product(speeds, lanes, distances))
    needed = [(s, la, d) for s, la, d in combos
              if discretize(s, la, d) not in reward_cache]

    if not needed:
        print("✅ Cache کامله، نیازی به pre-fill نیست.")
        return

    print(f"▶ Pre-filling {len(needed)} / {len(combos)} state ...")
    for idx, (speed, lane, dist) in enumerate(needed, 1):
        key    = discretize(speed, lane, dist)
        reward = _query_groq(speed, lane, dist)
        with _cache_lock:
            reward_cache[key] = reward
        time.sleep(0.5)   # Groq free tier: ~30 req/min

        if idx % 20 == 0 or idx == len(needed):
            _save_cache()
            if verbose:
                print(f"  [{idx}/{len(needed)}] speed={speed} lane={lane} "
                      f"dist={dist} → {reward:.3f}")

    _save_cache()
    print(f"✅ Done — {len(reward_cache)} entries در cache.")
