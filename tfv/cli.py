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
    if (sys.argv[1:] if argv is None else argv)[:1] == ["--separator-worker"]:
        # the music separator's helper process in a packaged build (tfv.engine.separate)
        from tfv.engine.separate import worker_main

        return worker_main((sys.argv[1:] if argv is None else argv)[1:])
    # Transcripts contain CJK text; don't let a legacy console code page crash printing.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    args = _parser().parse_args(argv)
    if not hasattr(args, "func"):  # no command: start the desktop app
        from tfv.ui.app import main as ui_main

        return ui_main()
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
    sub = p.add_subparsers(metavar="COMMAND", help="with no command, the desktop app starts")

    ui = sub.add_parser("ui", help="start the desktop app: subtitle overlay and tray menu (the default)")
    ui.set_defaults(func=lambda args: __import__("tfv.ui.app", fromlist=["main"]).main())

    d = sub.add_parser("devices", help="list output devices that can be captured")
    d.set_defaults(func=cmd_devices)
    d.add_argument("--inputs", action="store_true", help="list input devices (microphones, virtual cables) instead")

    hw = sub.add_parser("hardware", help="show this machine, the chosen profile and which models fit")
    hw.set_defaults(func=cmd_hardware)
    hw.add_argument("--profile", default="auto", help="show another profile instead of the automatic choice")

    k = sub.add_parser("keys", help="store API keys for paid services in the OS credential store")
    k.set_defaults(func=cmd_keys)
    k.add_argument("action", choices=("list", "set", "delete"))
    k.add_argument("service", nargs="?", choices=tuple(keys.SERVICES))

    listen = sub.add_parser("listen", help="caption (and translate) what is playing, or a file")
    listen.set_defaults(func=cmd_listen)
    _language_args(listen)
    src = listen.add_argument_group("audio source")
    src.add_argument("--capture", choices=("output", "input"), default="output", help="output = what the PC plays (default), input = microphone")
    src.add_argument("--audio-device", metavar="NAME", help="device index or name substring (default: the system default)")
    src.add_argument("--file", type=Path, help="use an audio/video file instead of live audio")
    src.add_argument("--realtime", action="store_true", help="with --file: feed the file at playback speed")
    src.add_argument("--duration", type=float, metavar="SECONDS", help="stop after this many seconds")
    remote = listen.add_argument_group("remote engine", "Run the models on another computer started with `tfv serve`.")
    remote.add_argument("--remote", metavar="URL", help="engine server, e.g. ws://192.168.1.20:8765")
    remote.add_argument("--token", help="the server's token (default: `tfv keys set remote` or $TFV_REMOTE_TOKEN)")
    _model_args(listen)
    other = listen.add_argument_group("other")
    other.add_argument("--log", type=Path, help="also append captions to this file")

    s = sub.add_parser("serve", help="run the models for other computers on the network (remote engine)")
    s.set_defaults(func=cmd_serve)
    _language_args(s)
    s.add_argument("--host", default="127.0.0.1", help="address to listen on; 0.0.0.0 for the whole LAN (default: this PC only)")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--token", help="shared secret clients must send (generated if listening beyond this PC)")
    _model_args(s)

    e = sub.add_parser("eval", help="measure recognition/translation quality and speed on this machine")
    e.set_defaults(func=cmd_eval)
    e.add_argument("action", choices=("download", "run"))
    e.add_argument(
        "--set", choices=("fleurs", "reazonspeech"), default="fleurs",
        help="fleurs: read speech in all 3 languages, with translations (default); reazonspeech: Japanese TV speech",
    )  # fmt: skip
    e.add_argument("--n", type=int, help="clips to download: per language for fleurs (default 20), in total for reazonspeech (default 200)")
    e.add_argument("--asr", nargs="*", help="recognizers to measure (default: a set suited to the data)")
    e.add_argument("--mt", nargs="*", default=["nllb-600m@cpu", "qwen3.5-2b@gpu", "qwen3.5-4b@gpu", "translategemma-4b@gpu"])
    e.add_argument("--snr", type=float, default=10.0, help="noise level for the noisy condition, dB (default 10)")
    e.add_argument("--out", type=Path, help="Markdown report (default EVAL.md, or EVAL-reazonspeech.md)")

    m = sub.add_parser("models", help="list, download or remove models")
    m.set_defaults(func=cmd_models)
    m.add_argument("action", nargs="?", choices=("list", "download", "remove"), default="list")
    m.add_argument("ids", nargs="*", metavar="MODEL", help="model ids for download/remove")
    return p


def _language_args(parser) -> None:
    lang = parser.add_argument_group("languages")
    lang.add_argument("--from", dest="source_lang", required=True, choices=languages.SUPPORTED, help="language being spoken")
    lang.add_argument("--to", dest="target_lang", choices=languages.SUPPORTED, help="translate into this language")


def _model_args(parser) -> None:
    group = parser.add_argument_group("models", "By default the hardware profile picks the models (see `tfv hardware`).")
    group.add_argument("--profile", default="auto", help="hardware profile to use (default: auto)")
    group.add_argument("--asr", default="auto", metavar="CHOICE", help="recognizer, e.g. whisper-small@cuda, sensevoice-small")
    group.add_argument(
        "--mt", default="auto", metavar="CHOICE",
        help="translator, e.g. qwen3.5-2b@gpu, nllb-600m, deepl, claude, openai-compatible:MODEL, or none",
    )  # fmt: skip
    group.add_argument("--mt-url", help="server URL for openai-compatible (e.g. http://localhost:11434 for Ollama)")
    group.add_argument("--no-adapt", action="store_true", help="never switch to lighter models while running")
    group.add_argument("--threads", type=int, help="CPU threads for the models")
    group.add_argument("--vad-threshold", type=float, default=0.5, help="speech probability that starts an utterance")
    group.add_argument("--no-vad-denoise", action="store_true", help="let the VAD hear the raw audio (by default a denoised copy, which finds speech under music)")
    group.add_argument("--no-music-filter", action="store_true", help="don't take the music out from under voices on lines with music (needs a GPU)")
    group.add_argument("--song-mode", action="store_true", help="while music plays, separate it continuously (finds more singing; captions ~1 s later)")
    group.add_argument("--min-silence-ms", type=int, default=500, help="pause that ends an utterance")
    group.add_argument("--max-segment-s", type=float, default=8.0, help="split longer speech (keeps translations coming)")


def _session_config(args):
    from tfv.engine.segmenter import SegmenterConfig
    from tfv.session import SessionConfig

    return SessionConfig(
        source_lang=args.source_lang, target_lang=args.target_lang,
        capture=getattr(args, "capture", "output"), audio_device=getattr(args, "audio_device", None),
        file=getattr(args, "file", None), realtime=getattr(args, "realtime", False),
        profile=args.profile, asr=args.asr, mt=args.mt, mt_url=args.mt_url,
        threads=args.threads, adapt=not args.no_adapt, vad_denoise=not args.no_vad_denoise,
        separate_vocals=not args.no_music_filter, song_mode=args.song_mode,
        segmenter=SegmenterConfig(threshold=args.vad_threshold, min_silence_ms=args.min_silence_ms, max_segment_s=args.max_segment_s),
    )  # fmt: skip


def cmd_devices(args) -> int:
    if args.inputs:
        from tfv.capture.input_device import list_input_devices

        for dev in list_input_devices():
            mark = "*" if dev["default"] else " "
            print(f"{mark} [{dev['index']:>2}] {dev['name']}  ({dev['rate']} Hz, {dev['channels']} ch)")
        print("\n* = default input. Use `listen --capture input --audio-device NAME`.")
        return 0
    from tfv.capture.loopback import list_output_devices

    for dev in list_output_devices():
        mark = "*" if dev["default"] else " "
        print(f"{mark} [{dev['index']:>2}] {dev['name']}  ({dev['rate']} Hz, {dev['channels']} ch)")
    print("\n* = current default output. Use --audio-device with the index or part of the name.")
    return 0


def cmd_eval(args) -> int:
    from tfv import evaluation, hardware

    japanese_tv = args.set == "reazonspeech"
    if args.action == "download":
        if japanese_tv:
            evaluation.download_reazonspeech(args.n or 200, status=log.info)
        else:
            evaluation.download(args.n or 20, status=log.info)
        return 0
    samples = evaluation.load_manifest(args.set)
    default_asr = (
        ["sensevoice-small@cpu", "reazonspeech-k2-v2@cpu", "whisper-large-v3-turbo@cuda", "kotoba-whisper-v2.0@cuda"]
        if japanese_tv
        else ["whisper-large-v3-turbo@cuda", "whisper-small@cuda", "sensevoice-small@cpu"]
    )
    asr_clean, asr_noisy, mt = [], [], []
    for choice in args.asr if args.asr is not None else default_asr:
        log.info("Recognition %s (clean)", choice)
        asr_clean.append(evaluation.run_asr(choice, samples, None, status=log.info, data=args.set))
        log.info("Recognition %s (noisy)", choice)
        asr_noisy.append(evaluation.run_asr(choice, samples, args.snr, status=log.info, data=args.set))
    if japanese_tv:
        out = args.out or Path("EVAL-reazonspeech.md")
        out.write_text(evaluation.report_reazonspeech(hardware.probe(), len(samples), asr_clean, asr_noisy), "utf-8")
        log.info("Wrote %s", out)
        return 0
    for choice in args.mt:
        log.info("Translation %s", choice)
        mt.append(evaluation.run_mt(choice, samples, status=log.info))
    n = max(sum(1 for s in samples if s["lang"] == lang) for lang in languages.SUPPORTED)
    out = args.out or Path("EVAL.md")
    out.write_text(evaluation.report(hardware.probe(), n, asr_clean, asr_noisy, mt), "utf-8")
    log.info("Wrote %s", out)
    return 0


def cmd_serve(args) -> int:
    from tfv.remote import EngineServer

    server = EngineServer(_session_config(args), host=args.host, port=args.port, token=args.token)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Stopping")
    return 0


def cmd_hardware(args) -> int:
    from tfv import hardware, keys, models, resources

    models.set_progress_listener(_progress_printer())  # checking GPU fit may fetch llama.cpp first
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
    ladders = [("Recognition", profile.asr)]
    ladders += [(f"Recognition when {lang} is spoken", r) for lang, r in profile.asr_by_lang.items()]
    ladders += [("Translation", profile.mt)]
    ladders += [(f"Translation into {lang}", r) for lang, r in profile.mt_by_lang.items()]
    for title, rungs in ladders:
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
        print(f"{'model':<24}{'task':<5}{'size':>9}  {'installed':<10}license")
        for info in models.CATALOG.values():
            mark = "yes" if models.is_installed(info.id) else ""
            print(f"{info.id:<24}{info.task:<5}{info.size_mb:>6,} MB  {mark:<10}{info.license}")
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
    from tfv import keys
    from tfv.engine.pipeline import Caption
    from tfv.session import Session

    config = _session_config(args)
    logfile = open(args.log, "a", encoding="utf-8") if args.log else None
    print_lock = threading.Lock()

    def show(c: Caption) -> None:
        if c.stage in ("partial", "dropped") or (c.stage == "text" and c.target_lang is not None):
            return  # live text isn't printed; a line is printed together with its translation
        stamp = f"[{int(c.start // 60):02d}:{c.start % 60:04.1f}]"
        cont = "…" if c.forced else " "  # cut mid-speech; the sentence continues on the next line
        line = f"{stamp}{cont} {c.text}"
        if args.verbose:
            line += f"   ({c.duration:.1f}s audio, asr {c.asr_s:.2f}s, waited {c.waited_s:.2f}s)"
        if c.stage == "translated":
            line += f"\n{' ' * len(stamp)}  → {c.translation}"
            if args.verbose:
                line += f"   (mt {c.mt_s:.2f}s)"
        elif c.stage == "untranslated" and args.verbose:
            line += f"\n{' ' * len(stamp)}  → (not translated)"
        with print_lock:
            print(line, flush=True)
            if logfile:
                logfile.write(line + "\n")
                logfile.flush()

    if args.remote:
        from tfv.remote import RemoteSession

        session = RemoteSession(args.remote, args.token or keys.get("remote"), config, show)
    else:
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
