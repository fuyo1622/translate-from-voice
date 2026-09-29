from __future__ import annotations

from pathlib import Path

import numpy as np

from tfv import SAMPLE_RATE


class SileroVAD:
    """Speech probability for consecutive 512-sample (32 ms) windows of 16 kHz audio.

    Runs the Silero VAD v5/v6 ONNX model directly with onnxruntime, so no
    PyTorch is needed. The model is stateful: feed windows in order.
    """

    window_size = 512
    _context_size = 64  # v5+ models expect the last 64 samples of the previous window prepended

    def __init__(self, model_path: str | Path):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.log_severity_level = 3
        self._session = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, self._context_size), dtype=np.float32)

    def __call__(self, window: np.ndarray) -> float:
        x = np.concatenate([self._context, window.reshape(1, -1).astype(np.float32, copy=False)], axis=1)
        prob, self._state = self._session.run(None, {"input": x, "state": self._state, "sr": self._sr})
        self._context = x[:, -self._context_size :]
        return float(prob[0, 0])
