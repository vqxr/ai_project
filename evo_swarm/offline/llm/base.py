from __future__ import annotations

from abc import ABC, abstractmethod


class LocalLLM(ABC):
    """
    Pluggable offline LLM backend.

    Keep the interface tiny so you can swap between llama.cpp server, Ollama,
    local python bindings, etc.
    """

    @abstractmethod
    def generate(self, *, system: str, prompt: str) -> str:
        raise NotImplementedError


class EchoLLM(LocalLLM):
    """
    Safe default backend: no model, just echoes intent.
    Useful to validate the agent/runtime wiring while still being offline.
    """

    def generate(self, *, system: str, prompt: str) -> str:
        _ = system
        return (
            "LLM_BACKEND_NOT_CONFIGURED\n"
            "Provide a local LLM backend (e.g. llama.cpp server) to enable planning.\n\n"
            f"PROMPT:\n{prompt}\n"
        )

