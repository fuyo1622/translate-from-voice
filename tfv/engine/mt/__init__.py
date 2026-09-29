"""Machine translation backends."""

from __future__ import annotations

import logging
import os

from tfv import models
from tfv.engine.mt.base import Context, Translator

log = logging.getLogger(__name__)

BACKENDS = ("nllb", "llama", "openai-compatible")


def create_translator(
    backend: str,
    *,
    source: str,
    target: str,
    model: str | None = None,
    compute: str = "auto",
    url: str | None = None,
    api_key: str | None = None,
    threads: int | None = None,
) -> Translator:
    """Build and warm up a translator.

    - "nllb": NLLB-200 600M with CTranslate2 (CPU unless compute="cuda").
    - "llama": a catalog GGUF model (default qwen3.5-2b) in a managed llama-server;
      compute="cpu" keeps it off the GPU.
    - "openai-compatible": an existing server at `url` (Ollama, LM Studio, OpenAI),
      `model` is the server's model name; the key defaults to $TFV_LLM_API_KEY.
    """
    if backend == "nllb":
        from tfv.engine.mt.nllb import NLLBTranslator

        translator: Translator = NLLBTranslator(model or "nllb-600m", device="cuda" if compute == "cuda" else "cpu", threads=threads)
    elif backend == "llama":
        translator = _local_llm(model or "qwen3.5-2b", compute, threads)
    elif backend == "openai-compatible":
        from tfv.engine.mt.llm import LLMTranslator

        if not url or not model:
            raise ValueError("openai-compatible translation needs --mt-url and --mt-model")
        translator = LLMTranslator(url, model=model, api_key=api_key or os.environ.get("TFV_LLM_API_KEY"))
    else:
        raise ValueError(f"unknown translation backend {backend!r}; choose from {', '.join(BACKENDS)}")
    try:
        translator.warmup(source, target)
    except Exception:
        translator.close()
        raise
    return translator


def _local_llm(model_id: str, compute: str, threads: int | None) -> Translator:
    from tfv.engine.mt.llama_server import LlamaServer
    from tfv.engine.mt.llm import LLMTranslator

    info = models.get(model_id)
    if info.engine != "llama":
        raise ValueError(f"{model_id} is not an LLM; pick one of: " + ", ".join(m.id for m in models.CATALOG.values() if m.engine == "llama"))
    models.ensure(model_id)
    server = LlamaServer(info.main_file, device="none" if compute == "cpu" else "auto", threads=threads)
    server.start()
    where = server.gpu.name if server.gpu else "CPU"
    return LLMTranslator(
        server.url, prompt=info.prompt, api_key=server.api_key, server=server, description=f"{model_id} via llama.cpp on {where}"
    )


__all__ = ["BACKENDS", "Context", "Translator", "create_translator"]
