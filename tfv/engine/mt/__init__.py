"""Machine translation backends."""

from __future__ import annotations

import logging

from tfv import models
from tfv.engine.mt.base import Context, Translator

log = logging.getLogger(__name__)


def create_translator(
    choice,
    *,
    source: str,
    target: str,
    url: str | None = None,
    api_key: str | None = None,
    threads: int | None = None,
) -> Translator:
    """Build and warm up the translator described by a resources.Choice.

    - nllb: NLLB-200 with CTranslate2 on the CPU (or CUDA with @cuda).
    - llama: a catalog GGUF model in a managed llama-server; @cpu keeps it off the GPU.
    - openai-compatible:<model>: an existing server at `url` (Ollama, LM Studio, vLLM).
    - deepl, claude, openai: paid services.
    """
    if choice.backend == "nllb":
        from tfv.engine.mt.nllb import NLLBTranslator

        device = "cuda" if choice.device == "cuda" else "cpu"
        translator: Translator = NLLBTranslator(choice.model, device=device, threads=threads)
    elif choice.backend == "llama":
        translator = _local_llm(choice.model, choice.device, threads)
    elif choice.backend == "openai-compatible":
        from tfv.engine.mt.llm import LLMTranslator

        if not url or not choice.model:
            raise ValueError("an OpenAI-compatible server needs a URL and a model name (openai-compatible:<model>)")
        translator = LLMTranslator(url, model=choice.model, api_key=api_key)
    else:
        from tfv.engine.mt.cloud import create_cloud_translator

        translator = create_cloud_translator(choice, api_key=api_key)
    try:
        translator.warmup(source, target)
    except Exception:
        translator.close()
        raise
    return translator


def _local_llm(model_id: str, device: str, threads: int | None) -> Translator:
    from tfv.engine.mt.llama_server import LlamaServer
    from tfv.engine.mt.llm import LLMTranslator

    info = models.ensure(model_id)
    # TranslateGemma's template rejects ordinary chat messages, which llama-server tries at
    # startup; we send its exact prompt ourselves, so a plain built-in template is enough.
    extra = ["--no-jinja", "--chat-template", "gemma"] if info.prompt == "translategemma" else []
    server = LlamaServer(info.main_file, device="none" if device == "cpu" else "auto", threads=threads, extra_args=extra)
    server.start()
    where = server.gpu.name if server.gpu else "CPU"
    return LLMTranslator(
        server.url, prompt=info.prompt, api_key=server.api_key, server=server, description=f"{model_id} via llama.cpp on {where}"
    )


__all__ = ["Context", "Translator", "create_translator"]
