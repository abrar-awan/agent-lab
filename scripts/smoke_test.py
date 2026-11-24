import os
import sys
import time
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root
from providers import PROVIDERS  # noqa: E402

PROMPT = "Reply in one short sentence: which model are you?"


def main():
    for name, cfg in PROVIDERS.items():
        model = os.getenv(cfg["model_env"])
        api_key = "ollama" if cfg["key_env"] is None else os.getenv(cfg["key_env"])

        if not model or not api_key:
            missing = cfg["model_env"] if not model else cfg["key_env"]
            print(f"[{name}] SKIPPED: {missing} not set in .env")
            continue

        # max_retries=0 so errors show immediately instead of being retried silently.
        # timeout=120 because Ollama's first call has to load the model into memory.
        client = OpenAI(base_url=cfg["base_url"], api_key=api_key,
                        timeout=120, max_retries=0)

        start = time.perf_counter()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": PROMPT}],
            )
            reply = (resp.choices[0].message.content or "").strip()
            print(f"[{name}] OK {time.perf_counter() - start:.1f}s ({model}): {reply}")
        except Exception as e:
            print(f"[{name}] FAILED {time.perf_counter() - start:.1f}s ({model}): "
                  f"{type(e).__name__}: {str(e)[:200]}")


if __name__ == "__main__":
    main()