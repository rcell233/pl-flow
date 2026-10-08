"""Tensor interface to the separately running speaker embedding tool."""

import base64
import json
import math
import os
import queue
import subprocess
import sys
import tempfile
import threading
import weakref
from pathlib import Path

import numpy as np
import torch
from torch import nn


def _exchange(process, requests, responses):
    """Keep both pipe writes and reads outside the caller's timeout boundary."""
    try:
        responses.put(process.stdout.readline())
        while True:
            message = requests.get()
            if message is None:
                return
            process.stdin.write(message + "\n")
            process.stdin.flush()
            responses.put(process.stdout.readline())
    except (OSError, ValueError) as error:
        responses.put(error)


class _Worker:
    def __init__(self):
        self.owner = os.getpid()
        self.process = self.stderr = self.thread = None
        self.requests = queue.Queue()
        self.responses = queue.Queue()

    def diagnostics(self):
        if self.stderr is None:
            return ""
        # pread leaves the child's logging position alone.
        size = os.fstat(self.stderr.fileno()).st_size
        return (
            os.pread(self.stderr.fileno(), 8192, max(0, size - 8192))
            .decode("utf-8", errors="replace")
            .strip()
        )

    def close(self):
        process = self.process
        if process is None or os.getpid() != self.owner:
            return
        self.requests.put(None)
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        else:
            process.wait()
        if self.thread is not None:
            self.thread.join(timeout=3)
        process.stdin.close()
        process.stdout.close()
        self.stderr.close()
        self.process = self.stderr = self.thread = None
        self.requests = queue.Queue()
        self.responses = queue.Queue()


class SpeakerEncoder(nn.Module):
    """Frozen 16 kHz audio -> float32 [batch, 256] embeddings via a persistent process.

    The tool is started on demand and reused until ``close()`` or a device change.
    Each Python process must create its own encoder. Context-manager use is optional;
    garbage collection and normal interpreter exit also stop the child process.
    """

    embedding_dim = 256

    def __init__(
        self, model_directory, device="cpu", *, weights_filename="model.safetensors", timeout=300
    ):
        super().__init__()
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Speaker timeout must be positive and finite")
        self.model_directory = Path(model_directory).resolve()
        self.weights_filename = weights_filename
        self.timeout = timeout
        self.register_buffer("_device", torch.empty(0, device=device), persistent=False)
        self._worker = _Worker()
        self._lock = threading.RLock()
        self._finalizer = weakref.finalize(self, _Worker.close, self._worker)
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    def _apply(self, function, recurse=True):
        self.close()
        return super()._apply(function, recurse)

    def _check_owner(self):
        if self._worker.owner != os.getpid():
            raise RuntimeError("Create a separate SpeakerEncoder in each Python process")

    def _command(self):
        return [
            sys.executable,
            "-u",
            "-m",
            "wavlm_ecapa",
            "--model-dir",
            str(self.model_directory),
            "--weights",
            self.weights_filename,
            "--device",
            str(self._device.device),
            "--threads",
            str(torch.get_num_threads()),
            "--serve",
        ]

    def _receive(self):
        try:
            message = self._worker.responses.get(timeout=self.timeout)
        except queue.Empty as error:
            raise TimeoutError(f"Speaker tool did not respond within {self.timeout}s") from error
        if isinstance(message, Exception):
            raise RuntimeError("Speaker tool pipe failed") from message
        if not message:
            raise RuntimeError("Speaker tool exited without a response")
        try:
            response = json.loads(message)
        except (TypeError, ValueError) as error:
            raise RuntimeError("Invalid speaker tool response") from error
        if not isinstance(response, dict) or response.get("ok") is not True:
            detail = (
                response.get("error", "Invalid response")
                if isinstance(response, dict)
                else response
            )
            raise RuntimeError(f"Speaker tool: {detail}")
        return response

    def _abort(self, error):
        detail = self._worker.diagnostics()
        self._worker.close()
        if detail:
            error.args = (f"{error}\nSpeaker stderr:\n{detail}",)

    def start(self):
        """Start the tool and strictly load its local checkpoint before returning."""
        self._check_owner()
        with self._lock:
            worker = self._worker
            if worker.process is not None and worker.process.poll() is None:
                return self
            worker.close()
            worker.stderr = tempfile.TemporaryFile()
            try:
                worker.process = subprocess.Popen(
                    self._command(),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=worker.stderr,
                    text=True,
                    encoding="utf-8",
                    bufsize=1,
                    # Works both from a source checkout and from an installed wheel.
                    cwd=Path(__file__).resolve().parents[2],
                )
            except BaseException:
                worker.stderr.close()
                worker.stderr = None
                raise
            worker.thread = threading.Thread(
                target=_exchange,
                args=(worker.process, worker.requests, worker.responses),
                daemon=True,
            )
            worker.thread.start()
            try:
                ready = self._receive()
                if ready.get("ready") is not True or ready.get("embedding_dim") != 256:
                    raise RuntimeError("Incompatible speaker tool protocol")
            except BaseException as error:
                self._abort(error)
                raise
        return self

    def close(self):
        """Stop the child and release its model memory; a later call restarts it."""
        self._check_owner()
        with self._lock:
            self._worker.close()

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.close()

    @torch.no_grad()
    def forward(self, waves, lengths):
        if (
            waves.ndim != 2
            or waves.size(0) == 0
            or waves.size(1) == 0
            or not waves.is_floating_point()
            or lengths.shape != (len(waves),)
            or lengths.dtype not in (torch.int32, torch.int64)
            or (lengths <= 0).any()
            or (lengths > waves.size(1)).any()
        ):
            raise ValueError("Expected floating audio [batch, samples] and valid integer lengths")
        audio = (
            waves.detach().to(device="cpu", dtype=torch.float32).numpy().astype("<f4", copy=False)
        )
        sizes = lengths.cpu().tolist()
        if any(not np.isfinite(row[:size]).all() for row, size in zip(audio, sizes)):
            raise ValueError("Speaker audio must be finite")
        request = {
            "op": "embed",
            "shape": list(audio.shape),
            "lengths": sizes,
            "audio": base64.b64encode(audio.tobytes()).decode("ascii"),
            "precision": torch.get_float32_matmul_precision(),
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "deterministic": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        }
        self._check_owner()
        with self._lock:
            self.start()
            try:
                self._worker.requests.put(json.dumps(request, separators=(",", ":")))
                response = self._receive()
                if response.get("shape") != [len(waves), self.embedding_dim]:
                    raise ValueError("Unexpected speaker embedding shape")
                raw = base64.b64decode(response["embedding"], validate=True)
                if len(raw) != len(waves) * self.embedding_dim * 4:
                    raise ValueError("Unexpected speaker embedding size")
                result = np.frombuffer(raw, dtype="<f4").reshape(len(waves), self.embedding_dim)
                if not np.isfinite(result).all():
                    raise ValueError("Non-finite speaker embedding")
                return torch.from_numpy(result.copy()).to(self._device.device)
            except BaseException as error:
                self._abort(error)
                raise
