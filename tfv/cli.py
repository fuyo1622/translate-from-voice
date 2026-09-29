"""Command line: `python -m tfv devices | listen | models`."""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from pathlib import Path

from tfv import languages

log = logging.getLogger("tfv")


def main(argv: list[str] | None = None) -> int:
    # Transcripts contain CJK text; don't let a legacy console code page crash printing.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose > 1 else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "httpcore", "huggingface_hub", "faster_whisper"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        log.error("%s", exc)
        return 1


def _parser() -> argparse.ArgumentParser:
    from tfv import keys

    p = argparse.ArgumentParser(prog="tfv", description="Live captions and translation for the audio this computer is playing.")
    p.add_argument("-v", "--verbose", action="count", default=0, help="show timings (-vv: debug logging)")
    sub = p.add_subparsers(required=True, metavar="COMMAND")

    d = sub.add_parser("devices", help="list output devices that can be captured")
    d.set_defaults(func=cmd_devices)

    hw = sub.add_parser("hardware", help="show this machine, the chosen profile and which models fit")
    hw.set_defaults(func=cmd_hardware)
    hw.add_argument("--profile", default="auto", help="show another profile instead of the automatic choice")

    k = sub.add_parser("keys", help="store API keys for paid services in the OS credential store")
    k.set_defaults(func=cmd_keys)
    k.add_argument("action", choices=("list", "set", "delete"))
    k.add_argument("service", nargs="?", choices=tuple(keys.SERVICES))

    listen = sub.add_parser("listen", help="caption (and translate) what is playing, or a file")
    listen.set_defaults(func=cmd_listen)
    lang = listen.add_argument_group("languages")
    lang.add_argument("--from", dest="source_lang", required=True, choices=languages.SUPPORTED, help="language being spoken")
    lang.add_argument("--to", dest="target_lang", choices=languages.SUPPORTED, help="translate into this language")
    src = listen.add_argument_group("audio source")
    src.add_argument("--audio-device", metavar="NAME", help="output device index or name substring (default: Windows default output)")
    src.add_argument("--file", type=Path, help="use an audio/video file instead of live audio")
    src.add_argument("--realtime", action="store_true", help="with --file: feed the file at playback speed")
    src.add_argument("--duration", type=float, metavar="SECONDS", help="stop after this many seconds")
    models_group = listen.add_argument_group(
        "models", "By default the hardware profile picks the models (see `tfv hardware`); these override it."
    )
    models_group.add_argument("--profile", default="auto", help="hardware profile to use (default: auto)")
    models_group.add_argument("--asr", default="auto", metavar="CHOICE", help="recognizer, e.g. whisper-small@cuda, sensevoice-small")
    models_group.add_argument(
        "--mt", default="auto", metavar="CHOICE",
        help="translator, e.g. qwen3.5-2b@gpu, nllb-600m, deepl, claude, openai-compatible:MODEL, or none",
    )  # fmt: skip
    models_group.add_argument("--mt-url", help="server URL for openai-compatible (e.g. http://localhost:11434 for Ollama)")
    models_group.add_argument("--no-adapt", action="store_true", help="never switch to lighter models while running")
    seg = listen.add_argument_group("segmentation")
    seg.add_argument("--vad-threshold", type=float, default=0.5, help="speech probability that starts an utterance")
    seg.add_argument("--min-silence-ms", type=int, default=500, help="pause that ends an utterance")
    seg.add_argument("--max-segment-s", type=float, default=12.0, help="split longer speech")
    other = listen.add_argument_group("other")
    other.add_argument("--threads", type=int, help="CPU threads for the models")
    other.add_argument("--log", type=Path, help="also append captions to this file")

    m = sub.add_parser("models", help="list, download or remove models")
    m.set_defaults(func=cmd_models)
    m.add_argument("action", nargs="?", choices=("list", "download", "remove"), default="list")
    m.add_argument("ids", nargs="*", metavar="MODEL", help="model ids for download/remove")
    return p


def cmd_devices(args) -> int:
    from tfv.capture.loopback import list_output_devices

    for dev in list_output_devices():
        mark = "*" if dev["default"] else " "
        print(f"{mark} [{dev['index']:>2}] {dev['name']}  ({dev['rate']} Hz, {dev['channels']} ch)")
    print("\n* = current default output. Use --audio-device with the index or part of the name.")
    return 0


def cmd_hardware(args) -> int:
    from tfv import hardware, keys, models, resources

    hw = hardware.probe()
    print(f"CPU:  {hw.cpu}, {hw.physical_cores} cores / {hw.logical_cores} threads")
    print(f"RAM:  {hw.ram_mb / 1024:.1f} GB ({hw.ram_free_mb / 1024:.1f} GB free)")
    for g in hw.gpus:
        free = f", {g.free_mb:,} MB free" if g.free_mb is not None else ""
        print(f"GPU:  {g.name} ({g.vram_mb:,} MB{free}){'  [integrated, not used]' if g.integrated else ''}")
    if any(g.vendor == "nvidia" for g in hw.gpus):
        print(f"CUDA: {'available' if hw.cuda else 'unavailable — ' + hw.cuda_note}")
    if hw.on_battery:
        print("Power: on battery")
    profile = resources.choose_profile(hw, args.profile)
    print(f"\nProfile: {profile.name} — {profile.summary}")
    print("Each stage uses the first option that fits; lower ones are fallbacks.")
    for title, rungs in (("Recognition", profile.asr), ("Translation", profile.mt)):
        print(f"\n{title}:")
        for choice in rungs:
            reason = resources.unfit_reason(choice, hw, keys.has)
            info = models.CATALOG.get(choice.model or "")
            size = f"{info.size_mb:>6,} MB" if info else " " * 9
            state = "installed" if info and models.is_installed(info.id) else ""
            print(f"  {'✗' if reason else '✓'} {str(choice):<32}{size}  {state:<10}{reason or ''}")
    return 0


def cmd_keys(args) -> int:
    import getpass

    from tfv import keys

    if args.action == "list":
        for service, env in keys.SERVICES.items():
            print(f"{service:<9} {'set' if keys.has(service) else 'not set':<8} (or set {env})")
        return 0
    if not args.service:
        raise ValueError(f"name the service to {args.action}")
    if args.action == "set":
        key = getpass.getpass(f"{args.service} API key (input hidden): ")
        if not key.strip():
            raise ValueError("no key entered")
        keys.store(args.service, key)
        print(f"Stored the {args.service} key in the system credential store.")
    else:
        print("Deleted." if keys.delete(args.service) else "No stored key.")
    return 0


def cmd_models(args) -> int:
    from tfv import models

    if args.action == "list":
        print(f"{'model':<22}{'task':<5}{'size':>9}  {'installed':<10}license")
        for info in models.CATALOG.values():
            mark = "yes" if models.is_installed(info.id) else ""
            print(f"{info.id:<22}{info.task:<5}{info.size_mb:>6,} MB  {mark:<10}{info.license}")
        print(f"\nStored in {models.models_dir()} (set TFV_MODELS_DIR to change).")
        return 0
    if not args.ids:
        raise ValueError(f"name the model(s) to {args.action}")
    for model_id in args.ids:
        if args.action == "download":
            models.ensure(model_id, progress=_progress_printer())
            print(f"{model_id}: installed")
        else:
            print(f"{model_id}: {'removed' if models.remove(model_id) else 'not installed'}")
    return 0


def _progress_printer():
    last = [0.0]

    def progress(label: str, done: int, total: int) -> None:
        now = time.monotonic()
        if now - last[0] >= 0.5 or done == total:
            last[0] = now
            pct = f"{100 * done / total:5.1f}%" if total else ""
            print(f"\r{label}: {done / 1e6:,.0f} MB {pct}   ", end="", file=sys.stderr, flush=True)
            if done == total:
                print(file=sys.stderr)

    return progress


def cmd_listen(args) -> int:
    from tfv.engine.pipeline import Caption
    from tfv.engine.segmenter import SegmenterConfig
    from tfv.session import Session, SessionConfig

    config = SessionConfig(
        source_lang=args.source_lang, target_lang=args.target_lang,
        audio_device=args.audio_device, file=args.file, realtime=args.realtime,
        profile=args.profile, asr=args.asr, mt=args.mt, mt_url=args.mt_url,
        threads=args.threads, adapt=not args.no_adapt,
        segmenter=SegmenterConfig(threshold=args.vad_threshold, min_silence_ms=args.min_silence_ms, max_segment_s=args.max_segment_s),
    )  # fmt: skip
    logfile = open(args.log, "a", encoding="utf-8") if args.log else None
    print_lock = threading.Lock()

    def show(c: Caption) -> None:
        stamp = f"[{int(c.start // 60):02d}:{c.start % 60:04.1f}]"
        if c.stage == "text":
            cont = "…" if c.forced else " "  # cut mid-speech; the sentence continues on the next line
            line = f"{stamp}{cont} {c.text}"
            if args.verbose:
                line += f"   ({c.duration:.1f}s audio, asr {c.asr_s:.2f}s, waited {c.waited_s:.2f}s)"
        elif c.stage == "translated":
            line = f"{stamp} → {c.translation}"
            if args.verbose:
                line += f"   (mt {c.mt_s:.2f}s)"
        elif args.verbose:
            line = f"{stamp} → (not translated)"
        else:
            return
        with print_lock:
            print(line, flush=True)
            if logfile:
                logfile.write(line + "\n")
                logfile.flush()

    session = Session(config, show)
    t0 = time.monotonic()
    try:
        session.start()
        if not args.file:
            log.info("Press Ctrl+C to stop")
        while not session.wait(0.2):
            if args.duration and time.monotonic() - t0 >= args.duration:
                break
    except KeyboardInterrupt:
        log.info("Stopping")
    finally:
        session.stop()
        if logfile:
            logfile.close()
    if args.file:
        log.info("Done in %.1fs", time.monotonic() - t0)
    p = session.pipeline
    if p and (p.dropped_utterances or p.dropped_translations):
        log.warning(
            "Could not keep up: skipped %d utterances and %d translations", p.dropped_utterances, p.dropped_translations
        )
    return 0
