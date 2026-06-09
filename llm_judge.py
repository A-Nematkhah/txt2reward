import os
import re
import pickle
import time
from groq import Groq
from prompts import SYSTEM_PROMPT

# ── API Key از environment variable یا مستقیم ────────────────────────────────
# در Colab: os.environ["GROQ_API_KEY"] = "gsk_..."
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    raise EnvironmentError(
        "\n❌ GROQ_API_KEY تنظیم نشده!\n"
        "در Colab قبل از اجرا این رو اجرا کن:\n"
        '  import os; os.environ["GROQ_API_KEY"] = "gsk_xxxxxxxx"'
    )

client = Groq(api_key=GROQ_API_KEY)
MODEL  = "llama-3.3-70b-versatile"   # سریع‌ترین مدل Groq با کیفیت بالا

# ── Cache ─────────────────────────────────────────────────────────────────────
CACHE_FILE = "reward_cache.pkl"
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "rb") as f:
        reward_cache = pickle.load(f)
    print(f"✅ Cache لود شد: {len(reward_cache)} entry")
else:
    reward_cache = {}

def discretize(speed, lane, distance):
    return (int(speed // 5), int(lane), int(distance // 10))

def query_groq(speed, lane, distance, retries=3):
    user_msg = (
        f"Speed:{speed:.0f} Lane:{lane} FrontDistance:{distance:.0f}\n"
        f"Reply with ONLY a single digit 0-5. No explanation."
    )

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                max_tokens=5,
                temperature=0.0,
            )
            text = response.choices[0].message.content.strip()

            match = re.search(r"[0-5]", text)
            score = int(match.group()) if match else 3
            return (score - 2.5) / 2.5

        except Exception as e:
            err = str(e)
            if "rate_limit" in err.lower() or "429" in err:
                wait = 2 ** attempt   # 1s, 2s, 4s
                print(f"  ⚠️  Rate limit — {wait}s صبر می‌کنم...")
                time.sleep(wait)
            else:
                print(f"  ❌ Groq error: {e}")
                break

    return 0.0   # fallback اگه همه retry‌ها شکست خوردن

def judge_state(speed, lane, front_distance):
    key = discretize(speed, lane, front_distance)

    if key in reward_cache:
        return reward_cache[key]

    reward = query_groq(speed, lane, front_distance)
    reward_cache[key] = reward

    if len(reward_cache) % 50 == 0:
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(reward_cache, f)

    return reward
