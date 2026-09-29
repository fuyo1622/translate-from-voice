# PyInstaller spec: builds dist/tfv/ with tfv-app.exe (desktop app) and tfv.exe (command line).
#
#   pyinstaller packaging/tfv.spec                 CPU build (Whisper on CPU, local LLM via Vulkan)
#   set TFV_NVIDIA=1 & pyinstaller packaging/tfv.spec   also bundles cuDNN/cuBLAS for NVIDIA GPUs (~1.3 GB more)
#
# Models are not bundled; they download on first use (see `tfv models`).
import os

from PyInstaller.utils.hooks import collect_all

NVIDIA = os.environ.get("TFV_NVIDIA") == "1"
ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))

datas, binaries, hiddenimports = [], [], []
for package in (
    "tfv", "faster_whisper", "ctranslate2", "sherpa_onnx", "opencc", "onnxruntime", "tokenizers",
    "soxr", "pyaudiowpatch", "sounddevice", "_sounddevice_data", "keyring", "anthropic", "websockets",
):  # fmt: skip
    d, b, h = collect_all(package)
    datas += d
    binaries += b
    hiddenimports += h

if NVIDIA:
    import nvidia

    for root in nvidia.__path__:
        for lib in ("cudnn", "cublas"):
            bin_dir = os.path.join(root, lib, "bin")
            if os.path.isdir(bin_dir):
                binaries += [(os.path.join(bin_dir, f), f"nvidia/{lib}/bin") for f in os.listdir(bin_dir) if f.endswith(".dll")]

excludes = ["tkinter", "matplotlib", "torch", "IPython", "pytest"] + ([] if NVIDIA else ["nvidia"])


def analysis(script):
    return Analysis(
        [os.path.join(SPECPATH, script)], pathex=[ROOT], binaries=binaries, datas=datas,
        hiddenimports=hiddenimports, excludes=excludes,
    )  # fmt: skip


app = analysis("app_main.py")
cli = analysis("cli_main.py")
app_exe = EXE(PYZ(app.pure), app.scripts, [], exclude_binaries=True, name="tfv-app", console=False)
cli_exe = EXE(PYZ(cli.pure), cli.scripts, [], exclude_binaries=True, name="tfv", console=True)
COLLECT(app_exe, cli_exe, app.binaries, app.datas, cli.binaries, cli.datas, name="tfv")
