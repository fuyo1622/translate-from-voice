"""Translation with an LLM behind an HTTP API.

Works with the managed llama-server and with any OpenAI-compatible server
(Ollama, LM Studio, OpenAI). Each model family gets the prompt it was trained
for; see the `prompt` field of the model catalog.
"""

from __future__ import annotations

import logging
import re

from tfv.engine.mt.base import Context, Translator
from tfv.languages import NAMES

log = logging.getLogger(__name__)

_STYLE_RULES = {
    "zh": "Use Traditional Chinese characters and Taiwanese wording (繁體中文、台灣用語).",
    "ja": "Use natural Japanese.",
    "en": "Use natural English.",
}

# TranslateGemma's template: language names and the codes it accepts.
_TG = {"zh": ("Chinese", "zh-TW"), "en": ("English", "en"), "ja": ("Japanese", "ja")}

# Hunyuan-MT's documented prompts name the target language in Chinese when Chinese is involved.
_HUNYUAN_ZH_NAMES = {"zh": "繁体中文", "en": "英语", "ja": "日语"}


def chat_messages(text: str, source: str, target: str, context: Context = ()) -> list[dict]:
    """System prompt + previous lines as earlier turns, so the model keeps names and tone consistent."""
    system = (
        f"You translate subtitles from {NAMES[source]} into {NAMES[target]}. "
        "Reply with the translation of the user's line only: no explanations, notes, quotes or romanization. "
        f"Keep names and numbers. {_STYLE_RULES[target]}"
    )
    messages = [{"role": "system", "content": system}]
    for src, tgt in context:
        messages += [{"role": "user", "content": src}, {"role": "assistant", "content": tgt}]
    messages.append({"role": "user", "content": text})
    return messages


def translategemma_prompt(text: str, source: str, target: str) -> str:
    """The exact prompt TranslateGemma's chat template builds (BOS is added by the server)."""
    (s, s_code), (t, t_code) = _TG[source], _TG[target]
    return (
        f"<start_of_turn>user\nYou are a professional {s} ({s_code}) to {t} ({t_code}) translator. "
        f"Your goal is to accurately convey the meaning and nuances of the original {s} text while adhering "
        f"to {t} grammar, vocabulary, and cultural sensitivities.\n"
        f"Produce only the {t} translation, without any additional explanations or commentary. "
        f"Please translate the following {s} text into {t}:\n\n\n{text.strip()}<end_of_turn>\n<start_of_turn>model\n"
    )


def hunyuan_messages(text: str, source: str, target: str) -> list[dict]:
    if "zh" in (source, target):
        content = f"把下面的文本翻译成{_HUNYUAN_ZH_NAMES[target]}，不要额外解释。\n\n{text}"
    else:
        content = f"Translate the following segment into {NAMES[target]}, without additional explanation.\n\n{text}"
    return [{"role": "user", "content": content}]


_QUOTES = {'"': '"', "'": "'", "「": "」", "『": "』", "“": "”"}


def clean_output(out: str, source_text: str) -> str:
    """Strip reasoning blocks, trailing notes and wrapping quotes the model added."""
    out = re.sub(r"<think>.*?</think>", "", out, flags=re.DOTALL).strip()
    out = out.split("\n\n")[0]  # anything after a blank line is commentary
    out = " ".join(line.strip() for line in out.splitlines() if line.strip())
    if len(out) >= 2 and _QUOTES.get(out[0]) == out[-1] and source_text[:1] not in _QUOTES:
        out = out[1:-1].strip()
    return out


class LLMTranslator(Translator):
    """Translates each caption with one request to an LLM server."""

    name = "llm"

    def __init__(
        self,
        base_url: str,
        *,
        model: str = "local",
        prompt: str = "chat",
        api_key: str | None = None,
        server=None,
        timeout: float = 30.0,
        description: str = "",
    ):
        import httpx

        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = httpx.Client(base_url=base_url.rstrip("/"), headers=headers, timeout=timeout)
        self._model = model
        self._prompt = prompt
        self._server = server  # a LlamaServer we own and must stop
        self.uses_context = prompt == "chat"
        self.description = description or f"{model} via {base_url}"

    def translate(self, text: str, source: str, target: str, context: Context = ()) -> str:
        max_tokens = min(512, 32 + 4 * len(text))
        if self._prompt == "translategemma":
            out = self._complete(translategemma_prompt(text, source, target), max_tokens, stop=["<end_of_turn>"])
        elif self._prompt == "hunyuan":
            out = self._chat(hunyuan_messages(text, source, target), max_tokens)
        else:
            out = self._chat(chat_messages(text, source, target, context), max_tokens)
        return clean_output(out, text)

    def _chat(self, messages: list[dict], max_tokens: int) -> str:
        body = {"model": self._model, "messages": messages, "temperature": 0, "max_tokens": max_tokens}
        if self._server is not None:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        r = self._http.post("/v1/chat/completions", json=body)
        r.raise_for_status()
        return r.json()["choices"][0]["message"].get("content") or ""

    def _complete(self, prompt: str, max_tokens: int, stop: list[str]) -> str:
        """llama-server's raw completion endpoint, for models with a fixed prompt format."""
        body = {"prompt": prompt, "n_predict": max_tokens, "temperature": 0, "stop": stop, "cache_prompt": True}
        r = self._http.post("/completion", json=body)
        r.raise_for_status()
        return r.json().get("content", "")

    def close(self) -> None:
        self._http.close()
        if self._server is not None:
            self._server.stop()
