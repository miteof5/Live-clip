"""audio_events 单元测试：合成音频验证 burst / 语速突变 / 静音前峰（纯函数+临时 wav）。"""
from __future__ import annotations

import json
import math
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liveclip.audio_events import (  # noqa: E402
    build_audio_events,
    detect_bursts,
    detect_pre_silence_peaks,
    detect_speech_rate_jumps,
    load_wav_mono,
    rms_series,
)


SR = 16000


def _write_wav(path: Path, samples: np.ndarray, sr: int = SR) -> Path:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16).tobytes())
    return path


def _tone(seconds: float, amp: float = 0.1, freq: float = 440.0, sr: int = SR) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return amp * np.sin(2 * math.pi * freq * t)


def test_rms_series_shape():
    s = _tone(1.0)
    times, rms, db = rms_series(s, SR)
    assert len(times) == len(rms) == len(db)
    assert times[1] - times[0] == pytest.approx(0.05, abs=1e-6)
    # 0.1 幅度正弦 → RMS ≈ 0.1/√2 ≈ -23dB
    assert np.median(db) == pytest.approx(-23.0, abs=2.0)


def test_burst_detection_on_synthetic_spike():
    # 8s：前 3s 安静(0.02)，3-4s 突增(0.5)，后 4s 恢复
    parts = [_tone(3.0, 0.02), _tone(1.0, 0.5), _tone(4.0, 0.02)]
    s = np.concatenate(parts)
    times, _, db = rms_series(s, SR)
    bursts = detect_bursts(times, db, context_s=2.0, burst_db=9.0, burst_min_db=-25.0)
    assert len(bursts) == 1
    assert 2.5 <= bursts[0]["t"] <= 3.5
    assert bursts[0]["detail"]["peak_db"] > -15.0


def test_pre_silence_peak_detection():
    # 7s：前 4.5s 有声音(0.4)，4.5-7s 静音(0.0)
    parts = [_tone(4.5, 0.4), np.zeros(int(2.5 * SR))]
    s = np.concatenate(parts)
    times, _, db = rms_series(s, SR)
    sil = [{"start": 4.5, "end": 7.0, "duration": 2.5}]
    peaks = detect_pre_silence_peaks(sil, times, db, min_silence_s=1.2, pre_window_s=2.0, peak_min_db=-25.0)
    assert len(peaks) == 1
    assert peaks[0]["t"] == pytest.approx(4.5, abs=0.15)
    assert peaks[0]["detail"]["silence_duration"] == pytest.approx(2.5)


def test_speech_rate_jump_up():
    # 模拟 20s：0-8s 稀疏(每 1s 一个词)，9-14s 密集(每 0.2s 一个词)
    words = []
    t = 0.5
    while t < 8.0:
        words.append({"word": "x", "start": t, "end": t + 0.3})
        t += 1.0
    t = 9.0
    while t < 14.0:
        words.append({"word": "x", "start": t, "end": t + 0.2})
        t += 0.2
    jumps = detect_speech_rate_jumps(words, [], 20.0, rate_window_s=4.0, rate_hop_s=1.0,
                                     jump_up_factor=1.5, min_rate=2.0, smooth=3)
    ups = [j for j in jumps if j["detail"]["direction"] == "up"]
    assert ups, "应有语速陡增事件"
    # 平滑使上升沿起点提前（t≈7），断言事件确实覆盖密集语速区（9-14s）
    assert any(u["end"] >= 12.0 and u["detail"]["rate"] >= 3.0 for u in ups)


def test_build_audio_events_end_to_end(tmp_path):
    parts = [_tone(3.0, 0.02), _tone(1.0, 0.5), np.zeros(int(2.0 * SR)), _tone(2.0, 0.1)]
    s = np.concatenate(parts)
    wav = _write_wav(tmp_path / "syn.wav", s)
    words = [{"word": "x", "start": 0.5, "end": 0.8}, {"word": "x", "start": 1.0, "end": 1.3}]
    words_path = tmp_path / "words.json"
    words_path.write_text(json.dumps({"words": words}, ensure_ascii=False), encoding="utf-8")
    sil_path = tmp_path / "silences.json"
    sil_path.write_text(json.dumps([{"start": 4.0, "end": 6.0, "duration": 2.0}], ensure_ascii=False), encoding="utf-8")
    ev = build_audio_events(str(wav), str(words_path), str(sil_path))
    assert ev["duration"] == pytest.approx(8.0, abs=0.05)
    types = {e["type"] for e in ev["events"]}
    assert "burst" in types
    assert "pre_silence_peak" in types


def test_load_wav_mono_rejects_non_16bit(tmp_path):
    """非 16bit wav 应显式报错，而非静默读错数据。"""
    p = tmp_path / "bad8bit.wav"
    with wave.open(str(p), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(1)  # 8bit
        wf.setframerate(SR)
        wf.writeframes(b"\x00" * 1600)
    with pytest.raises(ValueError, match="16bit"):
        load_wav_mono(str(p))


def test_rms_series_prefixsum_matches_strided():
    """前缀和实现与原始 strided 实现输出一致（内存优化的正确性回归）。"""
    rng = np.random.default_rng(42)
    s = rng.uniform(-0.5, 0.5, int(SR * 2.0)).astype(np.float32)
    times, rms, db = rms_series(s, SR)
    # 参考实现（strided 复制窗口）
    win = int(SR * 0.1)
    hop = int(SR * 0.05)
    n = (len(s) - win) // hop + 1
    idx = hop * np.arange(n)[:, None] + np.arange(win)[None, :]
    ref = np.sqrt(np.mean(s[idx] * s[idx], axis=1) + 1e-12)
    assert np.allclose(rms, ref, atol=1e-7)
    assert np.allclose(db, np.clip(20.0 * np.log10(np.maximum(ref, 1e-9)), -120.0, 0.0), atol=1e-5)
