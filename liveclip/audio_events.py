"""audio_events.py — 音频事件信号（判断层副语言线索，规则层零成本）。

ASR 把音频压成文字丢了副语言信息，而切片"炸点"高度依赖笑声/惊呼/BGM 起落。
本模块用确定性检测（不调 LLM）从 wav + words.json + silences.json 提取：

1. burst 能量爆发        —— 短窗 RMS 突增（笑声/惊呼/掌声/音乐爆点）
2. speech_rate_jump 语速突变 —— 词级时间戳滑动窗语速陡增/骤降（兴奋/冷场）
3. pre_silence_peak 静音前能量峰 —— 长静音前 2s 内的能量峰（笑声后冷场=名场面留白）

输出 audio_events.json（判断层输入）：
    {video, duration, config, events: [{t, end, type, score, detail}]}

用法（CLI）：
    python -m liveclip.audio_events --audio <audio.wav> --words <words.json> [--silences <silences.json>]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:  # 模块方式
    from .config import load_config, output_dir
except ImportError:  # 直接运行（python liveclip/audio_events.py）
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from liveclip.config import load_config, output_dir

_WORD_KEYS = ("start", "end", "word")


# ---------------------------------------------------------------------------
# 音频读取 / RMS 序列
# ---------------------------------------------------------------------------
def load_wav_mono(path: str) -> tuple[np.ndarray, int]:
    """读 16bit PCM wav → (float32 [-1,1], sample_rate)。

    仅支持 16bit（sampwidth=2）；24bit/32bit/float wav 显式报错，
    避免静默读错数据（np.int16 强读 24bit 会得到错误采样）。
    """
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        n_ch = wf.getnchannels()
        n = wf.getnframes()
        if wf.getsampwidth() != 2:
            raise ValueError(
                f"仅支持 16bit PCM wav（当前 {wf.getsampwidth() * 8}bit），"
                "请用 ffmpeg 转成 pcm_s16le")
        raw = np.frombuffer(wf.readframes(n), dtype=np.int16)
    if n_ch > 1:
        raw = raw[:: n_ch]  # 取第一声道
    return raw.astype(np.float32) / 32768.0, sr


def rms_series(
    samples: np.ndarray,
    sr: int,
    window_s: float = 0.1,
    hop_s: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """滑动窗 RMS → (times[s], rms(线性), db[0..0dBFS])。

    用平方和前缀和（O(n) 内存）而非 strided 复制窗口数组
    （后者 10min 音频约 77MB 临时内存，长视频更严重）。
    """
    win = max(1, int(sr * window_s))
    hop = max(1, int(sr * hop_s))
    n = max(0, (len(samples) - win) // hop + 1)
    if n == 0:
        return np.array([]), np.array([]), np.array([])
    # 前缀和：rms² 的滑动窗均值 = (cumsum 差分) / win
    sq = samples.astype(np.float64) ** 2
    cs = np.empty(len(sq) + 1, dtype=np.float64)
    cs[0] = 0.0
    np.cumsum(sq, out=cs[1:])
    starts = hop * np.arange(n)
    ends = starts + win
    rms2 = (cs[ends] - cs[starts]) / win
    rms = np.sqrt(np.maximum(rms2, 0.0) + 1e-12)
    db = np.clip(20.0 * np.log10(np.maximum(rms, 1e-9)), -120.0, 0.0)
    times = hop * np.arange(n) / sr
    return times, rms, db


# ---------------------------------------------------------------------------
# 检测器 1：burst 能量爆发
# ---------------------------------------------------------------------------
def detect_bursts(
    times: np.ndarray,
    db: np.ndarray,
    *,
    context_s: float = 3.0,
    burst_db: float = 9.0,
    burst_min_db: float = -22.0,
    min_dur_s: float = 0.2,
    smooth: int = 3,
) -> List[Dict[str, Any]]:
    """短窗 RMS 相对局部基线突增 → 事件列表（按 t 排序）。

    局部基线 = 前后 context_s 内 RMS 的中位数（dB），BGM 全程在场时基线自动抬高。
    score = (peak_db - baseline_db) / 20 归一化到 0-1。
    """
    if times.size == 0:
        return []
    if smooth > 1:
        k = np.ones(smooth) / smooth
        db = np.convolve(db, k, mode="same")
    n = times.size
    dt = times[1] - times[0] if n > 1 else 0.05
    ctx = max(1, int(context_s / dt))

    # 局部基线（滑动中位数窗口）
    baseline = np.empty(n, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - ctx)
        hi = min(n, i + ctx + 1)
        baseline[i] = np.median(db[lo:hi])

    mask = (db >= baseline + burst_db) & (db >= burst_min_db)
    min_wins = max(1, int(min_dur_s / dt))
    events: List[Dict[str, Any]] = []
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        if (j - i) >= min_wins:
            seg = slice(i, j)
            k_peak = i + int(np.argmax(db[seg]))
            events.append({
                "t": round(float(times[i]), 3),
                "end": round(float(times[j - 1]) + dt, 3),
                "type": "burst",
                "score": round(float(np.clip((db[k_peak] - baseline[k_peak]) / 20.0, 0.0, 1.0)), 3),
                "detail": {
                    "peak_db": round(float(db[k_peak]), 2),
                    "baseline_db": round(float(baseline[k_peak]), 2),
                    "peak_t": round(float(times[k_peak]), 3),
                },
            })
        i = j
    return events


# ---------------------------------------------------------------------------
# 检测器 2：语速突变
# ---------------------------------------------------------------------------
def detect_speech_rate_jumps(
    words: List[Dict[str, Any]],
    silences: List[Dict[str, Any]],
    duration: float,
    *,
    rate_window_s: float = 5.0,
    rate_hop_s: float = 1.0,
    lookback_s: float = 10.0,
    jump_up_factor: float = 1.5,
    jump_down_factor: float = 0.3,
    min_rate: float = 2.0,
    smooth: int = 3,
    min_down_windows: int = 2,
) -> List[Dict[str, Any]]:
    """词级时间戳滑动窗语速 → 陡增（兴奋/金句密集）/ 骤降（冷场）。

    语速 = 窗内词数 / 有效时长（窗长 - 与静音相交时长）。
    先对 rate 序列移动平均（去瞬时毛刺），再与"前 lookback_s 中位数"比较；
    连续同向窗合并为一个事件（t=起始, end=终止, score=峰）。
    """
    if not words or duration <= 0:
        return []
    ts = [(float(w.get("start", -1)), float(w.get("end", -1))) for w in words]
    ts = [(s, e) for s, e in ts if s >= 0 and e >= s]
    if not ts:
        return []
    starts = np.array([s for s, _ in ts])
    sil = [(float(s.get("start", 0)), float(s.get("end", 0))) for s in silences or []]

    def _valid_dur(s0: float, s1: float) -> float:
        """窗内非静音时长。"""
        if s1 <= s0:
            return 0.0
        dur = s1 - s0
        for a, b in sil:
            ov = min(s1, b) - max(s0, a)
            if ov > 0:
                dur -= ov
        return max(dur, 0.05)

    n = int(duration / rate_hop_s) + 1
    raw_rates: List[float] = []
    times: List[float] = []
    for k in range(n):
        t0 = k * rate_hop_s
        t1 = t0 + rate_window_s
        if t0 >= duration:
            break
        cnt = int(np.sum((starts >= t0) & (starts < t1)))
        vd = _valid_dur(t0, t1)
        raw_rates.append(cnt / vd if cnt >= 1 else 0.0)
        times.append(t0)

    rates = raw_rates
    if smooth > 1 and len(rates) >= smooth:
        kk = np.ones(smooth) / smooth
        rates = np.convolve(rates, kk, mode="same").tolist()

    events: List[Dict[str, Any]] = []
    lookback = max(1, int(lookback_s / rate_hop_s))
    directions: List[str] = [""] * len(times)
    scores: List[float] = [0.0] * len(times)
    for k in range(len(times)):
        r = rates[k]
        hist = [rates[j] for j in range(max(0, k - lookback), k) if rates[j] >= 0.5]
        if len(hist) < 3:
            continue
        base = float(np.median(hist))
        if base < 0.8:
            continue  # 之前本来就没话，无突变可言
        if r >= max(base * jump_up_factor, min_rate):
            directions[k] = "up"
            scores[k] = float(np.clip((r - base) / max(r, base), 0.0, 1.0))
        elif r <= base * jump_down_factor and r < 0.8:
            directions[k] = "down"
            scores[k] = float(np.clip((base - r) / max(base, 0.5), 0.0, 1.0))

    # 合并连续同向窗（down 至少持续 min_down_windows 窗）
    i = 0
    while i < len(directions):
        d = directions[i]
        if not d:
            i += 1
            continue
        j = i
        while j < len(directions) and directions[j] == d:
            j += 1
        if d == "down" and (j - i) < min_down_windows:
            i = j
            continue
        seg = slice(i, j)
        k_peak = i + int(np.argmax(scores[seg]))
        events.append({
            "t": round(times[i], 3),
            "end": round(times[j - 1] + rate_window_s, 3),
            "type": "speech_rate_jump",
            "score": round(scores[k_peak], 3),
            "detail": {
                "rate": round(rates[k_peak], 2),
                "baseline": round(float(np.median([rates[x] for x in range(max(0, k_peak - lookback), k_peak) if rates[x] >= 0.5])), 2),
                "direction": d,
            },
        })
        i = j
    return events


def _merge_adjacent(events: List[Dict[str, Any]], gap_s: float) -> List[Dict[str, Any]]:
    """同类型、间隔 ≤ gap_s 的事件合并（t 取首个, end 取最后, score/detail 取峰值）。"""
    if not events:
        return []
    merged: List[Dict[str, Any]] = []
    for e in sorted(events, key=lambda e: (e["t"], e["type"])):
        if merged and merged[-1]["type"] == e["type"] and (e["t"] - merged[-1]["end"]) <= gap_s:
            last = merged[-1]
            if e["score"] > last["score"]:
                last["score"] = e["score"]
                last["detail"] = e["detail"]
            last["end"] = max(last["end"], e["end"])
        else:
            merged.append(dict(e))
    return merged


# ---------------------------------------------------------------------------
# 检测器 3：静音前能量峰
# ---------------------------------------------------------------------------
def detect_pre_silence_peaks(
    silences: List[Dict[str, Any]],
    times: np.ndarray,
    db: np.ndarray,
    *,
    min_silence_s: float = 1.2,
    pre_window_s: float = 2.0,
    peak_min_db: float = -25.0,
) -> List[Dict[str, Any]]:
    """长静音段前 pre_window_s 内的能量峰（笑声/惊呼后突然安静）。"""
    if times.size == 0 or not silences:
        return []
    dt = times[1] - times[0] if times.size > 1 else 0.05
    pre_wins = max(1, int(pre_window_s / dt))
    events: List[Dict[str, Any]] = []
    for s in silences:
        start = float(s.get("start", -1))
        dur = float(s.get("duration", 0))
        if start < 0 or dur < min_silence_s:
            continue
        t0 = max(0.0, start - pre_window_s)
        lo = int(t0 / dt)
        hi = max(lo, int(start / dt))  # [t0, start) 开区间
        if hi - lo < 1:
            continue
        seg = slice(lo, hi)
        k_peak = lo + int(np.argmax(db[seg]))
        peak_db = float(db[k_peak])
        if peak_db >= peak_min_db:
            events.append({
                "t": round(start, 3),
                "end": round(start + 0.5, 3),
                "type": "pre_silence_peak",
                "score": round(float(np.clip((peak_db + 35.0) / 25.0, 0.0, 1.0)), 3),
                "detail": {
                    "peak_db": round(peak_db, 2),
                    "peak_t": round(float(times[k_peak]), 3),
                    "silence_start": round(start, 3),
                    "silence_duration": round(dur, 3),
                },
            })
    return events


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def build_audio_events(
    audio_path: str,
    words_path: str,
    silences_path: Optional[str],
    *,
    cfg: Optional[Dict[str, Any]] = None,
    duration: Optional[float] = None,
    out_dir: Optional[Path] = None,
    **overrides: Any,
) -> Dict[str, Any]:
    """构建音频事件 JSON（判断层副语言线索）。"""
    cfg = cfg or load_config()
    ae = cfg.get("audio_events") or {}
    samples, sr = load_wav_mono(audio_path)
    wav_dur = len(samples) / sr if sr else 0.0
    duration = duration or wav_dur

    words_data = json.loads(Path(words_path).read_text(encoding="utf-8"))
    words = words_data.get("words", words_data) if isinstance(words_data, dict) else words_data

    silences: List[Dict[str, Any]] = []
    if silences_path and Path(silences_path).exists():
        data = json.loads(Path(silences_path).read_text(encoding="utf-8"))
        silences = data if isinstance(data, list) else data.get("silences", [])

    # 统一参数（overrides > config > 默认）
    def _p(name: str, default: float) -> float:
        return float(overrides.get(name, ae.get(name, default)))

    times, rms, db = rms_series(samples, sr, window_s=_p("window_s", 0.1), hop_s=_p("hop_s", 0.05))

    bursts = detect_bursts(
        times, db,
        context_s=_p("burst_context_s", 3.0), burst_db=_p("burst_db", 9.0),
        burst_min_db=_p("burst_min_db", -22.0), min_dur_s=_p("burst_min_dur_s", 0.2),
        smooth=int(_p("burst_smooth", 3)),
    )
    jumps = detect_speech_rate_jumps(
        words, silences, duration,
        rate_window_s=_p("rate_window_s", 5.0), rate_hop_s=_p("rate_hop_s", 1.0),
        lookback_s=_p("rate_lookback_s", 10.0), jump_up_factor=_p("rate_jump_up", 1.5),
        jump_down_factor=_p("rate_jump_down", 0.3), min_rate=_p("rate_min", 2.0),
        smooth=int(_p("rate_smooth", 3)), min_down_windows=int(_p("rate_min_down_windows", 2)),
    )
    peaks = detect_pre_silence_peaks(
        silences, times, db,
        min_silence_s=_p("pre_min_silence_s", 1.2), pre_window_s=_p("pre_window_s", 2.0),
        peak_min_db=_p("pre_peak_min_db", -25.0),
    )

    # 相邻同类事件合并（burst 0.5s / 语速 1.5s 内视为同一事件）
    bursts = _merge_adjacent(bursts, float(_p("burst_merge_gap_s", 0.5)))
    jumps = _merge_adjacent(jumps, float(_p("rate_merge_gap_s", 1.5)))
    events = sorted(bursts + jumps + peaks, key=lambda e: e["t"])
    return {
        "audio": str(audio_path),
        "words": str(words_path),
        "silences": str(silences_path) if silences_path else None,
        "duration": round(duration, 3),
        "sample_rate": sr,
        "config": {
            "window_s": _p("window_s", 0.1), "hop_s": _p("hop_s", 0.05),
            "burst_db": _p("burst_db", 9.0), "burst_min_db": _p("burst_min_db", -22.0),
            "rate_window_s": _p("rate_window_s", 5.0), "rate_jump_up": _p("rate_jump_up", 1.5),
            "rate_jump_down": _p("rate_jump_down", 0.3), "pre_window_s": _p("pre_window_s", 2.0),
            "pre_peak_min_db": _p("pre_peak_min_db", -25.0),
        },
        "counts": {"burst": len(bursts), "speech_rate_jump": len(jumps), "pre_silence_peak": len(peaks)},
        "events": events,
    }


def save_audio_events(ev: Dict[str, Any], out_path: str) -> str:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(ev, ensure_ascii=False, indent=1), encoding="utf-8")
    return str(p)


def main() -> None:
    ap = argparse.ArgumentParser(prog="audio_events.py", description="音频事件信号：burst/语速突变/静音前峰")
    ap.add_argument("--audio", required=True, help="音频 wav（16kHz 单声道）")
    ap.add_argument("--words", required=True, help="词级转写 words.json")
    ap.add_argument("--silences", default=None, help="静音段 silences.json（可选）")
    ap.add_argument("--out", default=None, help="输出路径（默认 outputs/work/<名>_audio_events.json）")
    args = ap.parse_args()

    cfg = load_config()
    out = args.out
    if out is None:
        safe = re.sub(r"[^\w\-]", "_", Path(args.audio).stem)
        out = str(output_dir(cfg) / "work" / f"{safe}_audio_events.json")
    ev = build_audio_events(args.audio, args.words, args.silences, cfg=cfg)
    save_audio_events(ev, out)
    print(json.dumps({
        "ok": True,
        "duration": ev["duration"],
        "counts": ev["counts"],
        "events": len(ev["events"]),
        "out": out,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
