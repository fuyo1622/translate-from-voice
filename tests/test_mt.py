from tfv.engine.mt.llama_server import GpuDevice, best_device
from tfv.engine.mt.llm import chat_messages, clean_output, hunyuan_messages, translategemma_prompt


def test_chat_messages_include_context_as_turns():
    msgs = chat_messages("元気？", "ja", "zh", context=[("こんにちは", "你好")])
    assert msgs[0]["role"] == "system"
    assert "Japanese" in msgs[0]["content"] and "Traditional Chinese" in msgs[0]["content"]
    assert [m["role"] for m in msgs[1:]] == ["user", "assistant", "user"]
    assert msgs[-1]["content"] == "元気？"


def test_translategemma_prompt_matches_template():
    p = translategemma_prompt(" Hello ", "en", "zh")
    assert p.startswith("<start_of_turn>user\nYou are a professional English (en) to Chinese (zh-TW) translator.")
    assert p.endswith("into Chinese:\n\n\nHello<end_of_turn>\n<start_of_turn>model\n")


def test_hunyuan_prompt_is_chinese_when_chinese_is_involved():
    assert hunyuan_messages("Hi", "en", "zh")[0]["content"].startswith("把下面的文本翻译成繁体中文")
    assert hunyuan_messages("Hi", "en", "ja")[0]["content"].startswith("Translate the following segment into Japanese")


def test_clean_output():
    assert clean_output("<think>\n\n</think>\n\n你好。", "Hello.") == "你好。"
    assert clean_output("Hello.\n\nNote: this is a greeting.", "你好。") == "Hello."
    assert clean_output("「こんにちは」", "Hello") == "こんにちは"
    assert clean_output("「こんにちは」", "「Hello」") == "「こんにちは」"  # the source was quoted too
    assert clean_output("line one\nline two", "x") == "line one line two"


def test_best_device_skips_integrated_intel():
    devices = [
        GpuDevice("Vulkan0", "Intel(R) Iris(R) Xe Graphics", 8037, 7379),
        GpuDevice("Vulkan1", "NVIDIA GeForce RTX 3050 Laptop GPU", 3964, 3369),
    ]
    assert best_device(devices).id == "Vulkan1"
    assert best_device(devices[:1]) is None
    assert best_device([GpuDevice("Vulkan0", "Intel(R) Arc(TM) A770", 16000, 15000)]).id == "Vulkan0"


def test_best_device_skips_software_vulkan():
    # Linux with Mesa: lavapipe lists all of the RAM as free memory
    devices = [
        GpuDevice("Vulkan0", "AMD Radeon RX 7600 (RADV NAVI33)", 8176, 7900),
        GpuDevice("Vulkan1", "llvmpipe (LLVM 17.0.6, 256 bits)", 31950, 31950),
    ]
    assert best_device(devices).id == "Vulkan0"
    assert best_device(devices[1:]) is None
