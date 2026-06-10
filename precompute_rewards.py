"""
precompute_rewards.py
─────────────────────
همه‌ی 192 ترکیب speed/lane/distance رو از Groq میپرسه و cache میکنه.
قبل از train.py اجرا کن تا training کاملاً offline باشه.

[FIX] نسخه قبل یه cache جدا (dict محلی) میساخت که به llm_judge وصل نبود.
      الان از prefill_cache_sync() در llm_judge استفاده میکنه که همان
      reward_cache module-level رو آپدیت و ذخیره میکنه.
"""

from llm_judge import prefill_cache_sync

if __name__ == "__main__":
    prefill_cache_sync(verbose=True)
