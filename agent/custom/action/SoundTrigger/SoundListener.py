import os
import threading
import time
import warnings
from typing import Optional

import librosa
import numpy as np
import soundcard as sc
from proctap import ProcessAudioCapture

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
        audio_device: Optional[str] = None,
        capture_mode: str = "process",
        process_name: str = "",
        stop_check=None,
    ):
        self.threshold = threshold
        self.counter_threshold = counter_threshold
        self.stop_check = stop_check
        self.audio_device = (audio_device or "").strip()
        self.capture_mode = (capture_mode or "process").strip().lower()
        self.process_name = (process_name or "").strip()

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
            if self._thread.is_alive():
                _log().warning("Ear thread 未在超时内退出，已请求后台自行结束")
        self._thread = None
        _log().info("Ear stopped")

    def _open_device(self):
        if self.audio_device:
            try:
                mic = sc.get_microphone(id=self.audio_device, include_loopback=True)
                _log().info(f"使用自定义音频接口: {self.audio_device}")
                return mic.recorder(samplerate=self.sr, channels=self.ch)
            except Exception as e:
                _log().warning(f"自定义音频接口不可用: {self.audio_device}, error: {e}")
                try:
                    devs = [m.name for m in sc.all_microphones(include_loopback=True)]
                    _log().info(f"可用回环接口: {devs}")
                except Exception:
                    pass

        speaker = sc.default_speaker()
        mic = sc.get_microphone(id=str(speaker.name), include_loopback=True)
        _log().info(f"使用默认扬声器回环: {speaker.name}")
        return mic.recorder(samplerate=self.sr, channels=self.ch)

    def _open_process_capture(self):
        if not self.process_name:
            raise RuntimeError("process capture mode requires process_name")
        from utils.win32_process import get_pids_by_name

        pids = get_pids_by_name(self.process_name)
        if not pids:
            raise RuntimeError(f"target process not found: {self.process_name}")

        pid = int(pids[0])
        _log().info(f"使用进程音频流: {self.process_name} (pid={pid})")
        cap = ProcessAudioCapture(pid=pid)
        cap.start()
        return cap

    def _close_process_capture(self, cap):
        if cap is None:
            return
        try:
            cap.stop()
        except Exception:
            pass
        try:
            cap.close()
        except Exception:
            pass

    def _check_window(self, buf, pos, max_s):
        if pos >= max_s:
            win = buf[pos - max_s : pos]
        else:
            win = np.concatenate([buf[-(max_s - pos) :], buf[:pos]])

        d_score = self.match(win, self._sample)
        c_score = 0.0
        if self._counter is not None:
            c_score = self.match(win, self._counter)
        return d_score, c_score

    def _push_frame(self, buf, pos, frame, max_total):
        new_s = frame.shape[0]
        end = pos + new_s
        if end <= max_total:
            buf[pos:end] = frame
        else:
            first = max_total - pos
            buf[pos:] = frame[:first]
            buf[: end - max_total] = frame[first:]
        return end % max_total, new_s

    def _loop_device(self):
        rec = None
        n = 0
        max_s = int(self.sr * self.sample_len)
        chunks = int(self.sr * self.interval / self.chunk)
        new_s = chunks * self.chunk
        max_total = max_s * 2
        buf = np.zeros(max_total, dtype=np.float64)
        pos = 0
        written = 0

        rec = self._open_device()
        rec.__enter__()
        while self._running.is_set():
            if self.stop_check and self.stop_check():
                break
            frame = np.empty(new_s, dtype=np.float64)
            idx = 0
            for _ in range(chunks):
                data = rec.record(numframes=self.chunk)
                frame[idx : idx + self.chunk] = librosa.to_mono(data.T)
                idx += self.chunk

            pos, pushed = self._push_frame(buf, pos, frame, max_total)
            written += pushed
            if written < max_s:
                continue
            d_score, c_score = self._check_window(buf, pos, max_s)
            self._check(d_score, c_score)
            n += 1
            if n % self.log_every == 0:
                _log().info(
                    f"[device] dodge={d_score:.4f}({self.threshold}), "
                    f"counter={c_score:.4f}({self.counter_threshold})"
                )

        if rec is not None:
            rec.__exit__(None, None, None)

    def _loop_process(self):
        cap = None
        n = 0
        no_data_ticks = 0
        reconnects = 0
        max_s = int(self.sr * self.sample_len)
        max_total = max_s * 2
        buf = np.zeros(max_total, dtype=np.float64)
        pos = 0
        written = 0
        while self._running.is_set():
            if self.stop_check and self.stop_check():
                break

            if cap is None:
                try:
                    cap = self._open_process_capture()
                    no_data_ticks = 0
                except Exception as e:
                    _log().warning(f"进程流连接失败: {e}")
                    time.sleep(0.3)
                    continue

            try:
                raw = cap.read(timeout=min(self.interval, 0.2))
            except Exception as e:
                _log().warning(f"进程流读取异常，准备重连: {e}")
                self._close_process_capture(cap)
                cap = None
                reconnects += 1
                continue
            if not raw:
                no_data_ticks += 1
                if no_data_ticks >= 20:
                    _log().warning("进程流长时间无数据，尝试重连")
                    self._close_process_capture(cap)
                    cap = None
                    reconnects += 1
                    no_data_ticks = 0
                continue
            no_data_ticks = 0
            pcm = np.frombuffer(raw, dtype=np.float32)
            if pcm.size < 2:
                continue
            if pcm.size % 2 != 0:
                pcm = pcm[:-1]
            stereo = pcm.reshape(-1, 2)
            mono = librosa.to_mono(stereo.T)
            # ProcTap 返回 48kHz 流，转换到匹配器采样率。
            frame = librosa.resample(mono, orig_sr=48000, target_sr=self.sr).astype(
                np.float64
            )
            if frame.size == 0:
                continue

            pos, pushed = self._push_frame(buf, pos, frame, max_total)
            written += pushed
            if written < max_s:
                continue
            d_score, c_score = self._check_window(buf, pos, max_s)
            self._check(d_score, c_score)
            n += 1
            if n % self.log_every == 0:
                _log().info(
                    f"[process] dodge={d_score:.4f}({self.threshold}), "
                    f"counter={c_score:.4f}({self.counter_threshold}), "
                    f"reconnects={reconnects}"
                )

        self._close_process_capture(cap)

    def _loop(self):
        try:
            import ctypes

            ctypes.windll.ole32.CoInitialize(None)
            if self.capture_mode == "process":
                self._loop_process()
            else:
                self._loop_device()
        except Exception as e:
            _log().error(f"Ear error: {e}", exc_info=True)

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
