"""Paid backends against fake HTTP servers: request shape and response parsing, no keys needed."""

import json

import httpx
import numpy as np
import pytest

from tfv import usage
from tfv.engine.asr.cloud import DeepgramTranscriber, OpenAITranscriber
from tfv.engine.mt.cloud import ClaudeTranslator, DeepLTranslator, OpenAITranslator


@pytest.fixture(autouse=True)
def fresh_usage():
    usage.reset()
    yield
    usage.reset()


def recorder(response_json, status=200):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(status, json=response_json)

    return seen, httpx.MockTransport(handler)


def test_deepl_free_key_request():
    seen, transport = recorder({"translations": [{"text": "你好"}]})
    t = DeepLTranslator("abc:fx", transport=transport)
    assert t.translate("Hello", "en", "zh", context=[("Hi there", "嗨")]) == "你好"
    req = seen[0]
    assert str(req.url) == "https://api-free.deepl.com/v2/translate"
    assert req.headers["Authorization"] == "DeepL-Auth-Key abc:fx"
    assert json.loads(req.content) == {"text": ["Hello"], "source_lang": "EN", "target_lang": "ZH-HANT", "context": "Hi there"}
    assert usage.summary() == ["deepl: 1 requests, 5 characters, ~$0.000"]


def test_deepl_quota_error_is_clear():
    _, transport = recorder({"message": "Quota exceeded"}, status=456)
    with pytest.raises(RuntimeError, match="quota"):
        DeepLTranslator("pro-key", transport=transport).translate("Hello", "en", "ja")


def test_openai_translator_body_for_reasoning_models():
    seen, transport = recorder({"choices": [{"message": {"content": "こんにちは"}}], "usage": {"prompt_tokens": 50, "completion_tokens": 5}})
    t = OpenAITranslator("sk-test", transport=transport)
    assert t.translate("Hello", "en", "ja") == "こんにちは"
    body = json.loads(seen[0].content)
    assert body["model"] == "gpt-6-luna" and body["reasoning_effort"] == "none"
    assert "temperature" not in body and body["messages"][0]["role"] == "system"
    assert seen[0].headers["Authorization"] == "Bearer sk-test"


def test_openai_transcriber_sends_wav_and_language():
    seen, transport = recorder({"text": " 今日は "})
    asr = OpenAITranscriber("sk-test", "ja", transport=transport)
    result = asr.transcribe(np.zeros(16000, np.float32))
    assert result.text == "今日は" and result.language == "ja"
    body = seen[0].content
    assert b'name="model"' in body and b"gpt-4o-mini-transcribe" in body
    assert b'name="language"' in body and b"\r\n\r\nja\r\n" in body
    assert b"RIFF" in body  # a WAV file was attached
    assert usage.summary()[0].startswith("openai gpt-4o-mini-transcribe: 1 requests, 0.0 audio min")


def test_deepgram_request_and_parse():
    seen, transport = recorder({"results": {"channels": [{"alternatives": [{"transcript": "你好"}]}]}})
    asr = DeepgramTranscriber("dg-key", "zh", transport=transport)
    assert asr.transcribe(np.zeros(8000, np.float32)).text == "你好"
    req = seen[0]
    assert req.url.params["model"] == "nova-3" and req.url.params["language"] == "zh-TW"
    assert req.headers["Authorization"] == "Token dg-key" and req.headers["Content-Type"] == "audio/wav"


def claude_client(response_json):
    import anthropic
    import httpx2

    seen = []

    def handler(request):
        seen.append(request)
        return httpx2.Response(200, json=response_json)

    return seen, anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler))


def claude_message(text="你好。", stop_reason="end_turn", **extra):
    return {
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5",
        "content": [{"type": "text", "text": text}], "stop_reason": stop_reason, "stop_sequence": None,
        "usage": {"input_tokens": 80, "output_tokens": 6}, **extra,
    }  # fmt: skip


def test_claude_request_uses_low_effort_and_fallbacks():
    seen, http = claude_client(claude_message())
    t = ClaudeTranslator("sk-ant-test", http_client=http)
    assert t.translate("Hello.", "en", "zh", context=[("Hi.", "嗨。")]) == "你好。"
    req = seen[0]
    body = json.loads(req.content)
    assert body["model"] == "claude-opus-5"
    assert body["fallbacks"] == "default" and "server-side-fallback-2026-07-01" in req.headers["anthropic-beta"]
    assert body["output_config"] == {"effort": "low"}
    assert [m["role"] for m in body["messages"]] == ["user", "assistant", "user"]
    assert "Traditional Chinese" in body["system"]
    assert usage.summary() == ["claude claude-opus-5: 1 requests, 86 tokens, ~$0.001"]


def test_claude_haiku_request_has_no_effort_or_fallbacks():
    seen, http = claude_client(claude_message())
    ClaudeTranslator("k", model="claude-haiku-4-5", http_client=http).translate("Hello.", "en", "zh")
    body = json.loads(seen[0].content)
    assert "output_config" not in body and "fallbacks" not in body


def test_claude_refusal_raises():
    _, http = claude_client(claude_message(text="", stop_reason="refusal", stop_details={"type": "refusal", "category": "cyber", "explanation": None}))
    with pytest.raises(RuntimeError, match="declined"):
        ClaudeTranslator("k", http_client=http).translate("Hello.", "en", "zh")
