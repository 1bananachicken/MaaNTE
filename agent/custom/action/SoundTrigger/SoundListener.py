import os
import threading
import time
import warnings
from typing import Callable, Optional

import librosa
import numpy as np
import soundcard as sc

warnings.filterwarnings("ignore", message="data discontinuity in recording")

logger = None


def _log():
    global logger
    if logger is None:
        from custom.action.Common.logger import get_logger

        logger = get_logger(__name__)
    return logger


class Ear:
    sr = 32000
    ch = 2
    chunk = 1600
    sample_len = 0.2
    interval = 0.1
    log_every = 20
    degree = 4
    cut_off = 1000

    def __init__(
        self,
        sample_path: str,
        counter_path: str,
        threshold: float = 0.13,
        counter_threshold: float = 0.12,
        stop_check=None,
    ):
        self.threshold = threshold
        self.counter_threshold = counter_threshold
        self.stop_check = stop_check

        self._running = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_trigger = 0.0
        self._trigger_cd = 0.5

        self._sample = None
        self._counter = None
        self._b = None
        self._a = None
        self.on_dodge = None
        self.on_counter = None

        self._load(sample_path, counter_path)

    def _load(self, sample_path, counter_path):
        from scipy.signal import butter

        self._b, self._a = butter(
            self.degree, self.cut_off, btype="highpass", output="ba", fs=self.sr
        )
        self._sample = self._cache_load(sample_path)
        if counter_path:
            self._counter = self._cache_load(counter_path)
        _log().info(f"[Sample] 加载 {sample_path}_{self.sr}_{self.cut_off}.npy")
        if counter_path:
            _log().info(f"[Sample] 加载 {counter_path}_{self.sr}_{self.cut_off}.npy")

    def _cache_load(self, path: str):
        cache = f"{path}_{self.sr}_{self.degree}_{self.cut_off}.npy"
        if os.path.exists(cache) and os.path.getmtime(cache) > os.path.getmtime(path):
            return np.load(cache)

        wav, _ = librosa.load(path, sr=self.sr)
        wav = self._filt(wav)
        np.save(cache, wav)
        return wav

    def _filt(self, wav):
        from scipy.signal import filtfilt

        return filtfilt(self._b, self._a, wav)

    def match(self, stream, sample):
        from scipy.signal import correlate

        stream = self._filt(stream)
        s1 = self._norm(stream)
        s2 = self._norm(sample)

        if s1.shape[0] > s2.shape[0]:
            corr = correlate(s1, s2, mode="same", method="fft") / s1.shape[0]
        else:
            corr = correlate(s2, s1, mode="same", method="fft") / s2.shape[0]

        return np.max(corr)

    def _norm(self, wf):
        rms = np.sqrt(np.mean(wf**2) + 1e-6)
        return wf / rms

    def start(self):
        if self._running.is_set():
            return
        self._running.set()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        _log().info("Ear started")

    def stop(self):
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=3.0)
        self._thread = None
        _log().info("Ear stopped")

    def _open_device(self):
        speaker = sc.default_speaker()
        mic = sc.get_microphone(id=str(speaker.name), include_loopback=True)
        return mic.recorder(samplerate=self.sr, channels=self.ch)

    def _loop(self):
        rec = None
        try:
            import ctypes

            ctypes.windll.ole32.CoInitialize(None)
            rec = self._open_device()
            rec.__enter__()

            n = 0
            max_s = int(self.sr * self.sample_len)
            chunks = int(self.sr * self.interval / self.chunk)
            new_s = chunks * self.chunk

            buf = np.zeros(max_s * 2, dtype=np.float64)
            pos = 0
            written = 0

            while self._running.is_set():
                if self.stop_check and self.stop_check():
                    break

                frame = np.empty(new_s, dtype=np.float64)
                idx = 0
                for _ in range(chunks):
                    data = rec.record(numframes=self.chunk)
                    frame[idx : idx + self.chunk] = librosa.to_mono(data.T)
                    idx += self.chunk

                end = pos + new_s
                if end <= max_s * 2:
                    buf[pos:end] = frame
                else:
                    first = max_s * 2 - pos
                    buf[pos:] = frame[:first]
                    buf[: end - max_s * 2] = frame[first:]

                pos = end % (max_s * 2)
                written += new_s

                if written >= max_s:
                    if pos >= max_s:
                        win = buf[pos - max_s : pos]
                    else:
                        win = np.concatenate([buf[-(max_s - pos) :], buf[:pos]])

                    d_score = self.match(win, self._sample)
                    c_score = 0.0
                    if self._counter is not None:
                        c_score = self.match(win, self._counter)

                    self._check(d_score, c_score)

                    n += 1
                    if n % self.log_every == 0:
                        _log().info(
                            f"dodge={d_score:.4f}({self.threshold}), counter={c_score:.4f}({self.counter_threshold})"
                        )
        except Exception as e:
            _log().error(f"Ear error: {e}", exc_info=True)
        finally:
            if rec is not None:
                try:
                    rec.__exit__(None, None, None)
                except Exception:
                    pass

    def _check(self, d_score, c_score):
        now = time.time()
        if now - self._last_trigger < self._trigger_cd:
            return

        if d_score >= self.threshold:
            self._last_trigger = now
            _log().info(f"闪避触发分数: {d_score:.5f}")
            if self.on_dodge:
                self.on_dodge()
            return

        if c_score >= self.counter_threshold:
            self._last_trigger = now
            _log().info(f"反击触发分数: {c_score:.5f}")
            if self.on_counter:
                self.on_counter()
