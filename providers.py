"""Shared provider table. Values come from .env (loaded here, never printed)."""

import os

from dotenv import load_dotenv

load_dotenv()

# Order = fallback priority: free & fast first, paid last.
PROVIDERS = {
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_env": "GEMINI_API_KEY",
        "model_env": "GEMINI_MODEL",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "model_env": "GROQ_MODEL",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "model_env": "OPENROUTER_MODEL",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "key_env": None,  # local, no key needed
        "model_env": "OLLAMA_MODEL",
    },
    "mistral": {
        "base_url": "https://api.mistral.ai/v1",
        "key_env": "MISTRAL_API_KEY",
        "model_env": "MISTRAL_MODEL",
    },
    "cloudflare": {
        "base_url": f"https://api.cloudflare.com/client/v4/accounts/{os.getenv('CLOUDFLARE_ACCOUNT_ID')}/ai/v1",
        "key_env": "CLOUDFLARE_API_KEY",
        "model_env": "CLOUDFLARE_MODEL",
    },
}
