from __future__ import annotations

import json
import urllib.request
from typing import Optional

from evo_swarm.offline.llm.base import LocalLLM


class LlamaCppServerLLM(LocalLLM):
    """
    Minimal client for a local llama.cpp server.

    This avoids external deps (requests/httpx) so it works fully offline.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8080"):
        self.base_url = base_url.rstrip("/")

    def generate(self, *, system: str, prompt: str) -> str:
        # llama.cpp server supports OpenAI-compatible /v1/chat/completions in many builds.
        # If your build is different, adapt here.
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": "local",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        parsed = json.loads(body)
        return parsed["choices"][0]["message"]["content"]

