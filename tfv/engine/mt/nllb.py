from __future__ import annotations

import logging
import os
import time

from tfv import models
from tfv.engine.mt.base import Context, Translator

log = logging.getLogger(__name__)

# NLLB-200 language tags; "zho_Hant" produces Traditional characters.
_CODES = {"zh": "zho_Hant", "en": "eng_Latn", "ja": "jpn_Jpan"}


class NLLBTranslator(Translator):
    """NLLB-200 distilled 600M via CTranslate2 (int8). Small and fast on CPU; literal, no context."""

    name = "nllb"

    def __init__(self, model_id: str = "nllb-600m", device: str = "cpu", threads: int | None = None):
        import ctranslate2
        from tokenizers import Tokenizer

        info = models.ensure(model_id)
        threads = threads or max(1, min(4, (os.cpu_count() or 2) // 2))
        t0 = time.perf_counter()
        self._translator = ctranslate2.Translator(
            str(info.path), device=device, compute_type="int8" if device == "cpu" else "int8_float16", intra_threads=threads
        )
        self._tokenizer = Tokenizer.from_file(str(info.path / "tokenizer.json"))
        log.info("Loaded %s on %s in %.1fs", model_id, device, time.perf_counter() - t0)
        self.description = f"NLLB-200 600M on {device}"

    def translate(self, text: str, source: str, target: str, context: Context = ()) -> str:
        pieces = self._tokenizer.encode(text, add_special_tokens=False).tokens
        result = self._translator.translate_batch(
            [[_CODES[source], *pieces, "</s>"]],
            target_prefix=[[_CODES[target]]],
            beam_size=4,
            max_decoding_length=min(512, 3 * len(pieces) + 16),
        )
        output = result[0].hypotheses[0][1:]  # drop the target language tag
        ids = [self._tokenizer.token_to_id(t) for t in output]
        return self._tokenizer.decode([i for i in ids if i is not None], skip_special_tokens=True).strip()
