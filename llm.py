import os, random, time
import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import errors
from google.genai import types
load_dotenv()
client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"],
    http_options=types.HttpOptions(timeout=30000),  # 30 s (milliseconds)
)

MODELS = ["gemini-flash-latest", "gemini-flash-lite-latest"]
RETRYABLE = {429, 500, 502, 503, 504}

def generate(contents, config, attempts=4):
    last = None
    for model in MODELS:
        for i in range(attempts):
            try:
                return client.models.generate_content(
                    model=model, contents=contents, config=config
                )
            except (errors.APIError, httpx.TransportError) as e:
                if isinstance(e, errors.APIError):
                    if e.code not in RETRYABLE:
                        raise
                    label = e.code
                else:
                    label = type(e).__name__
                last = e
                wait = 2 ** i + random.random()
                print(f"[{model}] {label} — retrying in {wait:.1f}s")
                time.sleep(wait)
        print(f"[{model}] exhausted, falling back")
    raise last