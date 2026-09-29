"""Paid translation services: Claude, DeepL and OpenAI.

Each call sends the caption text (and, for context, the previous few lines)
to the provider. Keys come from tfv.keys; usage is counted in tfv.usage.
"""

from __future__ import annotations

import logging

from tfv import usage
from tfv.engine.mt.base import Context, Translator
from tfv.engine.mt.llm import LLMTranslator, chat_messages, clean_output

log = logging.getLogger(__name__)

CLAUDE_DEFAULT_MODEL = "claude-opus-5"
OPENAI_DEFAULT_MODEL = "gpt-6-luna"

# Models that take server-side refusal fallbacks (a declined line is retried on another model).
_CLAUDE_FALLBACK_MODELS = {"claude-opus-5", "claude-fable-5-1"}


class ClaudeTranslator(Translator):
    """Claude via the Anthropic SDK, with the previous lines as conversation context."""

    name = "claude"
    uses_context = True

    def __init__(self, api_key: str, model: str | None = None, timeout: float = 20.0, http_client=None):
        import anthropic

        # One retry at most: a caption that arrives late is worth less than the next one.
        self._client = anthropic.Anthropic(api_key=api_key, timeout=timeout, max_retries=1, http_client=http_client)
        self._model = model or CLAUDE_DEFAULT_MODEL
        self.description = f"Claude ({self._model})"

    def translate(self, text: str, source: str, target: str, context: Context = ()) -> str:
        messages = chat_messages(text, source, target, context)
        options: dict = {}
        if not self._model.startswith("claude-haiku"):
            # Short, latency-sensitive lines don't need deep thinking.
            options["output_config"] = {"effort": "low"}
        if self._model in _CLAUDE_FALLBACK_MODELS:
            options["betas"] = ["server-side-fallback-2026-07-01"]
            options["fallbacks"] = "default"
        response = self._client.beta.messages.create(
            model=self._model,
            max_tokens=4096,  # includes any thinking; lines themselves are short
            system=messages[0]["content"],
            messages=messages[1:],
            **options,
        )
        usage.record("claude", self._model, input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens)
        if response.stop_reason == "refusal":
            category = getattr(response.stop_details, "category", None) if response.stop_details else None
            raise RuntimeError(f"Claude declined to translate this line ({category or 'no category'})")
        out = "".join(block.text for block in response.content if block.type == "text")
        return clean_output(out, text)

    def close(self) -> None:
        self._client.close()


_DEEPL_SOURCE = {"zh": "ZH", "en": "EN", "ja": "JA"}
_DEEPL_TARGET = {"zh": "ZH-HANT", "en": "EN-US", "ja": "JA"}


class DeepLTranslator(Translator):
    """DeepL API. Free keys (ending in ":fx") use the free endpoint, 500k characters/month."""

    name = "deepl"
    uses_context = True

    def __init__(self, api_key: str, timeout: float = 15.0, transport=None):
        import httpx

        base = "https://api-free.deepl.com" if api_key.endswith(":fx") else "https://api.deepl.com"
        self._http = httpx.Client(
            base_url=base, timeout=timeout, headers={"Authorization": f"DeepL-Auth-Key {api_key}"}, transport=transport
        )
        self.description = "DeepL" + (" (free)" if api_key.endswith(":fx") else "")

    def translate(self, text: str, source: str, target: str, context: Context = ()) -> str:
        body = {"text": [text], "source_lang": _DEEPL_SOURCE[source], "target_lang": _DEEPL_TARGET[target]}
        if context:
            body["context"] = "\n".join(src for src, _ in context)  # not translated and not billed
        r = self._http.post("/v2/translate", json=body)
        if r.status_code == 456:
            raise RuntimeError("DeepL quota exceeded for this month")
        r.raise_for_status()
        usage.record("deepl", characters=len(text))
        return r.json()["translations"][0]["text"].strip()

    def close(self) -> None:
        self._http.close()


class OpenAITranslator(LLMTranslator):
    """OpenAI chat completions (api.openai.com)."""

    name = "openai"

    def __init__(self, api_key: str, model: str | None = None, timeout: float = 20.0, transport=None):
        model = model or OPENAI_DEFAULT_MODEL
        super().__init__(
            "https://api.openai.com", model=model, api_key=api_key, timeout=timeout, description=f"OpenAI ({model})", transport=transport
        )

    def _chat(self, messages: list[dict], max_tokens: int) -> str:
        body: dict = {"model": self._model, "messages": messages, "max_completion_tokens": 1024}
        # Reasoning models reject a custom temperature; ask them to skip reasoning instead.
        if self._model.startswith("gpt-6"):
            body["reasoning_effort"] = "none"
        elif self._model.startswith("gpt-5"):
            body["reasoning_effort"] = "minimal"
        elif self._model.startswith("gpt-4"):
            body["temperature"] = 0
        r = self._http.post("/v1/chat/completions", json=body)
        r.raise_for_status()
        data = r.json()
        tokens = data.get("usage") or {}
        usage.record("openai", self._model, input_tokens=tokens.get("prompt_tokens", 0), output_tokens=tokens.get("completion_tokens", 0))
        return data["choices"][0]["message"].get("content") or ""


def create_cloud_translator(choice, api_key: str | None) -> Translator:
    if not api_key:
        raise RuntimeError(f"no API key for {choice.backend}; add one with `python -m tfv keys set {choice.backend}`")
    if choice.backend == "claude":
        return ClaudeTranslator(api_key, choice.model)
    if choice.backend == "deepl":
        return DeepLTranslator(api_key)
    if choice.backend == "openai":
        return OpenAITranslator(api_key, choice.model)
    raise ValueError(f"unknown translation service {choice.backend!r}")
