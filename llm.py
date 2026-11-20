"""Tiny client for any OpenAI-compatible chat-completions endpoint.

Configure with environment variables (or a local .env file, never committed):
    LLM_BASE_URL   e.g. the provider's .../v1 base URL (check the provider's docs)
    LLM_API_KEY    your key
    LLM_MODEL      model name from the provider's docs
Without all three, the app runs in extractive mode (no LLM).
"""
from __future__ import annotations

import os
import time


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 30.0, max_retries: int = 2):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

    @classmethod
    def from_env(cls):
        base, key, model = (os.getenv(n, "").strip() for n in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"))
        return cls(base, key, model) if base and key and model else None

    def complete(self, system: str, user: str, max_tokens: int = 400) -> str:
        import requests

        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        last = "unknown error"
        for attempt in range(self.max_retries + 1):
            retryable = True
            try:
                resp = requests.post(f"{self.base_url}/chat/completions", json=payload,
                                     headers=headers, timeout=self.timeout)
            except requests.RequestException as err:
                last = f"network error: {type(err).__name__}"
            else:
                if resp.status_code == 200:
                    try:
                        return resp.json()["choices"][0]["message"]["content"].strip()
                    except (KeyError, IndexError, TypeError, ValueError, AttributeError):
                        raise LLMError("unexpected response shape") from None
                last = f"HTTP {resp.status_code}"
                retryable = resp.status_code in (429, 500, 502, 503, 504)
            if not retryable or attempt == self.max_retries:
                break
            time.sleep(1.5 * (attempt + 1))
        raise LLMError(last)   # the message never contains the API key
