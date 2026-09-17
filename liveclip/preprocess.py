"""预处理：抽音频 + 静音检测。

- extract_audio: 16kHz 单声道 wav（whisper 与 silencedetect 的输入）
- detect_silence: ffmpeg silencedetect 滤镜 → 静音段表（气口候选）
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import find_ffmpeg

_SILENCE_START_RE = re.compile(r"silence_start:\s*([\d.]+)")
_SILENCE_END_RE = re.compile(r"silence_end:\s*([\d.]+)\s*\| silence_duration:\s*([\d.]+)")


@dataclass
class Silence:
    start: float
    end: float
    duration: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def extract_audio(
    video_path: str,
    wav_path: str,
    ffmpeg: Optional[str] = None,
    sample_rate: int = 16000,
    channels: int = 1,
) -> str:
    """抽取音频为 wav；返回 wav 路径。"""
    ff = ffmpeg or find_ffmpeg()
    if not ff:
        raise RuntimeError("未找到 ffmpeg，请安装或配置 ffmpeg_path")
    wav = Path(wav_path)
    wav.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            ff, "-y", "-hide_banner", "-loglevel", "error",
            "-i", video_path,
            "-vn", "-ac", str(channels), "-ar", str(sample_rate),
            "-c:a", "pcm_s16le",
            str(wav),
        ],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if proc.returncode != 0 or not wav.exists():
        raise RuntimeError(f"抽音频失败: {proc.stderr.strip()}")
    return str(wav)


def detect_silence(
    audio_path: str,
    noise_db: float = -30.0,
    min_duration: float = 0.8,
    ffmpeg: Optional[str] = None,
) -> List[Silence]:
    """检测静音段（气口候选）。

    silencedetect 输出到 stderr，逐条解析 silence_start / silence_end。
    """
    ff = ffmpeg or find_ffmpeg()
    if not ff:
        raise RuntimeError("未找到 ffmpeg，请安装或配置 ffmpeg_path")
    filter_expr = f"silencedetect=noise={noise_db}dB:d={min_duration}"
    proc = subprocess.run(
        [ff, "-hide_banner", "-i", audio_path, "-af", filter_expr, "-f", "null", "-"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"静音检测失败: {proc.stderr.strip()}")

    silences: List[Silence] = []
    pending_start: Optional[float] = None
    for line in proc.stderr.splitlines():
        m = _SILENCE_END_RE.search(line)
        if m:
            end = float(m.group(1))
            dur = float(m.group(2))
            if pending_start is not None:
                silences.append(Silence(pending_start, end, dur))
                pending_start = None
            continue
        m = _SILENCE_START_RE.search(line)
        if m:
            pending_start = float(m.group(1))
    return silences


def find_chunk_boundaries(silences: List[Silence], chunk_max_seconds: float) -> List[float]:
    """根据静音段与最大块时长，计算转写切块边界（秒，含 0 与总时长）。

    切在静音段中点，避免切断句子。
    """
    if not silences:
        return [0.0, chunk_max_seconds]
    boundaries = [0.0]
    last = 0.0
    for s in silences:
        mid = (s.start + s.end) / 2.0
        if mid - last >= chunk_max_seconds * 0.8 and mid > last:
            boundaries.append(mid)
            last = mid
    return boundaries
