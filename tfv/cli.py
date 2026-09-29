"""Command line: `python -m tfv devices` and `python -m tfv listen`."""

from __future__ import annotations

import argparse
import logging
import sys
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
    for noisy in ("httpx", "huggingface_hub", "faster_whisper"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        log.error("%s", exc)
        return 1


def _parser() -> argparse.ArgumentParser:
    from tfv.engine.asr import BACKENDS

    p = argparse.ArgumentParser(prog="tfv", description="Live captions for the audio this computer is playing.")
    p.add_argument("-v", "--verbose", action="count", default=0, help="show timings (-vv: debug logging)")
    sub = p.add_subparsers(required=True, metavar="COMMAND")

    d = sub.add_parser("devices", help="list output devices that can be captured")
    d.set_defaults(func=cmd_devices)

    listen = sub.add_parser("listen", help="transcribe what is playing (or a file)")
    listen.set_defaults(func=cmd_listen)
    src = listen.add_argument_group("audio source")
    src.add_argument("--audio-device", metavar="NAME", help="output device index or name substring (default: Windows default output)")
    src.add_argument("--file", type=Path, help="transcribe an audio/video file instead of live audio")
    src.add_argument("--realtime", action="store_true", help="with --file: feed the file at playback speed")
    src.add_argument("--duration", type=float, metavar="SECONDS", help="stop after this many seconds")
    rec = listen.add_argument_group("recognition")
    rec.add_argument(
        "--from", dest="source_lang", required=True, choices=languages.SUPPORTED, help="language being spoken"
    )
    rec.add_argument("--asr", choices=BACKENDS, default="auto", help="recognizer (default: auto, based on hardware)")
    rec.add_argument("--compute", choices=("auto", "cuda", "cpu"), default="auto", help="where Whisper runs")
    rec.add_argument("--model", help="Whisper size (e.g. large-v3-turbo, small) or a model directory")
    rec.add_argument("--threads", type=int, help="CPU threads for recognition")
    seg = listen.add_argument_group("segmentation")
    seg.add_argument("--vad-threshold", type=float, default=0.5, help="speech probability that starts an utterance")
    seg.add_argument("--min-silence-ms", type=int, default=500, help="pause that ends an utterance")
    seg.add_argument("--max-segment-s", type=float, default=12.0, help="split longer speech")
    out = listen.add_argument_group("output")
    out.add_argument("--log", type=Path, help="also append transcripts to this file")
    return p


def cmd_devices(args) -> int:
    from tfv.capture.loopback import list_output_devices

    for dev in list_output_devices():
        mark = "*" if dev["default"] else " "
        print(f"{mark} [{dev['index']:>2}] {dev['name']}  ({dev['rate']} Hz, {dev['channels']} ch)")
    print("\n* = current default output. Use --audio-device with the index or part of the name.")
    return 0


def cmd_listen(args) -> int:
    from tfv import models
    from tfv.engine.asr import create_asr
    from tfv.engine.pipeline import Pipeline, Utterance
    from tfv.engine.segmenter import Segmenter, SegmenterConfig
    from tfv.engine.textproc import TextPostProcessor
    from tfv.engine.vad import SileroVAD

    if args.file:
        from tfv.capture.file import FileSource

        source = FileSource(args.file, realtime=args.realtime)
    else:
        from tfv.capture.loopback import LoopbackSource

        source = LoopbackSource(args.audio_device)

    asr = create_asr(args.asr, language=args.source_lang, compute=args.compute, model=args.model, threads=args.threads)
    segmenter = Segmenter(
        SileroVAD(models.silero_vad()),
        SegmenterConfig(
            threshold=args.vad_threshold, min_silence_ms=args.min_silence_ms, max_segment_s=args.max_segment_s
        ),
    )
    logfile = open(args.log, "a", encoding="utf-8") if args.log else None

    def show(u: Utterance) -> None:
        stamp = f"{int(u.segment.start // 60):02d}:{u.segment.start % 60:04.1f}"
        cont = "…" if u.segment.forced else " "  # cut mid-speech; the sentence continues on the next line
        line = f"[{stamp}]{cont} {u.text}"
        if args.verbose:
            line += f"   ({u.segment.duration:.1f}s audio, asr {u.result.compute_s:.2f}s, waited {u.queued_s:.2f}s)"
        print(line, flush=True)
        if logfile:
            logfile.write(line + "\n")
            logfile.flush()

    pipeline = Pipeline(source, segmenter, asr, show, postprocess=TextPostProcessor())
    pipeline.start()
    log.info("Recognizer: %s", asr.description)
    log.info("Listening to %s%s", source.description, "" if args.file else " — press Ctrl+C to stop")
    t0 = time.monotonic()
    try:
        while not pipeline.wait(0.2):
            if args.duration and time.monotonic() - t0 >= args.duration:
                break
    except KeyboardInterrupt:
        log.info("Stopping")
    finally:
        pipeline.stop()
        if logfile:
            logfile.close()
    if args.file:
        log.info("Done in %.1fs", time.monotonic() - t0)
    if pipeline.dropped:
        log.warning("%d utterances were dropped because recognition could not keep up", pipeline.dropped)
    return 0
