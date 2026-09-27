"""LLM backends.

``ClaudeLLM`` calls Claude through the official Anthropic SDK. ``MockLLM`` is a
deterministic, offline stand-in so every pipeline (and the test-suite) runs
without an API key. ``get_llm()`` picks Claude when credentials are configured.
"""

from __future__ import annotations

import json
import os
import re

from .embeddings import tokenize

DEFAULT_MODEL = "claude-opus-5"
# Server-side refusal fallback: if Claude declines, the API re-runs the request on
# Anthropic's recommended fallback model inside the same call.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMRefusalError(RuntimeError):
    pass


class BaseLLM:
    is_mock = False
    model = ""

    def generate(self, prompt: str, system: str | list | None = None, max_tokens: int = 4096,
                 effort: str | None = None) -> str:
        raise NotImplementedError

    def generate_json(self, prompt: str, schema: dict, system: str | None = None,
                      effort: str | None = "low") -> dict:
        raise NotImplementedError


class ClaudeLLM(BaseLLM):
    def __init__(self, model: str | None = None):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model or os.environ.get("RAG_MODEL", DEFAULT_MODEL)

    def create(self, **params):
        """Thin wrapper over ``beta.messages.create`` that adds refusal fallbacks.

        Exposed so advanced pipelines (tool use, citations, web search) can build
        their own requests while sharing the same model and fallback config.
        """
        params.setdefault("model", self.model)
        params.setdefault("max_tokens", 4096)
        return self.client.beta.messages.create(betas=[FALLBACK_BETA], fallbacks="default", **params)

    @staticmethod
    def text_of(response) -> str:
        if response.stop_reason == "refusal":
            raise LLMRefusalError(f"Claude declined the request: {response.stop_details}")
        return "".join(b.text for b in response.content if b.type == "text").strip()

    def generate(self, prompt, system=None, max_tokens=4096, effort=None):
        params = {"max_tokens": max_tokens, "messages": [{"role": "user", "content": prompt}]}
        if system:
            params["system"] = system
        if effort:
            params["output_config"] = {"effort": effort}
        return self.text_of(self.create(**params))

    def generate_json(self, prompt, schema, system=None, effort="low"):
        output_config = {"format": {"type": "json_schema", "schema": schema}}
        if effort:
            output_config["effort"] = effort
        params = {"max_tokens": 4096, "messages": [{"role": "user", "content": prompt}],
                  "output_config": output_config}
        if system:
            params["system"] = system
        return json.loads(self.text_of(self.create(**params)))


class MockLLM(BaseLLM):
    """Offline LLM: answers extractively from the context in the prompt.

    It is *not* intelligent - it exists so pipelines are runnable and testable
    without network access. Structured outputs return schema-valid defaults.
    """

    is_mock = True
    model = "mock"

    @staticmethod
    def _section(prompt: str, name: str) -> str:
        match = re.search(rf"<{name}>\s*(.*?)\s*</{name}>", prompt, re.S)
        return match.group(1) if match else ""

    def _question(self, prompt: str) -> str:
        return self._section(prompt, "question") or prompt.strip().splitlines()[-1]

    def generate(self, prompt, system=None, max_tokens=4096, effort=None):
        question = self._question(prompt)
        context = self._section(prompt, "context") or self._section(prompt, "document")
        if not context:
            return f"[mock] {question}"
        q_terms = set(tokenize(question))
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", context) if len(s.strip()) > 20]
        sentences = [s for s in sentences if not s.startswith("[")]
        ranked = sorted(sentences, key=lambda s: -len(q_terms & set(tokenize(s))))
        return "[mock] " + " ".join(ranked[:2])

    def generate_json(self, prompt, schema, system=None, effort="low"):
        return self._fill(schema, self._question(prompt))

    def _fill(self, schema: dict, question: str):
        kind = schema.get("type")
        if "enum" in schema:
            return schema["enum"][0]
        if kind == "object":
            return {k: self._fill(v, question) for k, v in schema.get("properties", {}).items()}
        if kind == "array":
            item = schema.get("items", {})
            return [question] if item.get("type") == "string" else []
        if kind == "boolean":
            return True
        if kind in ("integer", "number"):
            return 1
        return question


def get_llm(force_mock: bool | None = None) -> BaseLLM:
    """Return ClaudeLLM when credentials exist, otherwise MockLLM.

    Set ``RAG_MOCK=1`` to force the mock even when a key is present.
    """
    if force_mock is None:
        force_mock = os.environ.get("RAG_MOCK") == "1"
    has_credentials = any(os.environ.get(v) for v in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"))
    if force_mock or not has_credentials:
        return MockLLM()
    return ClaudeLLM()
