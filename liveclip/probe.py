"""媒体探测：获取视频分辨率/时长/帧率/音频信息。

后端顺序：ffprobe（若可用）→ ffmpeg -i（stderr 解析）→ pymediainfo。
剪映自带 ffmpeg.exe 但没有 ffprobe，因此 ffmpeg -i 解析是主要路径。
"""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional

from .config import find_ffmpeg, find_ffprobe


@dataclass
class MediaInfo:
    path: str
    duration_s: float
    width: int
    height: int
    fps: float
    has_audio: bool
    audio_codec: str = ""
    video_codec: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_STREAM_RE = re.compile(r"Stream #\d+:\d+.*?Video:.*?(\d{2,5})x(\d{2,5}).*?([\d.]+)\s*fps")
_VIDEO_CODEC_RE = re.compile(r"Stream #\d+:\d+.*?Video:\s*(\S+)")
_AUDIO_RE = re.compile(r"Stream #\d+:\d+.*?Audio:\s*(\S+)")


def _parse_ffmpeg_output(stderr_text: str) -> Optional[Dict[str, Any]]:
    info: Dict[str, Any] = {}
    m = _DURATION_RE.search(stderr_text)
    if m:
        h, mm, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
        info["duration_s"] = h * 3600 + mm * 60 + s
    m = _STREAM_RE.search(stderr_text)
    if m:
        info["width"], info["height"] = int(m.group(1)), int(m.group(2))
        info["fps"] = float(m.group(3))
    m = _VIDEO_CODEC_RE.search(stderr_text)
    if m:
        info["video_codec"] = m.group(1)
    m = _AUDIO_RE.search(stderr_text)
    if m:
        info["has_audio"] = True
        info["audio_codec"] = m.group(1)
    else:
        info["has_audio"] = False
    return info if ("duration_s" in info or "width" in info) else None


def probe_with_ffmpeg(ffmpeg: str, video_path: str) -> Optional[Dict[str, Any]]:
    proc = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", video_path],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    return _parse_ffmpeg_output(proc.stderr)


def probe_with_ffprobe(ffprobe: str, video_path: str) -> Optional[Dict[str, Any]]:
    proc = subprocess.run(
        [ffprobe, "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", video_path],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if proc.returncode != 0:
        return None
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None
    info: Dict[str, Any] = {}
    fmt = data.get("format", {})
    try:
        info["duration_s"] = float(fmt.get("duration", 0))
    except (TypeError, ValueError):
        info["duration_s"] = 0.0
    for st in data.get("streams", []):
        if st.get("codec_type") == "video":
            info["width"] = int(st.get("width", 0))
            info["height"] = int(st.get("height", 0))
            info["video_codec"] = st.get("codec_name", "")
            r = st.get("r_frame_rate", "0/1")
            try:
                num, den = r.split("/")
                info["fps"] = float(num) / float(den) if float(den) else 0.0
            except (ValueError, ZeroDivisionError):
                info["fps"] = 0.0
        elif st.get("codec_type") == "audio":
            info["has_audio"] = True
            info["audio_codec"] = st.get("codec_name", "")
    if "has_audio" not in info:
        info["has_audio"] = False
    return info if info.get("width") or info.get("duration_s") else None


def probe_with_pymediainfo(video_path: str) -> Optional[Dict[str, Any]]:
    try:
        from pymediainfo import MediaInfo
    except ImportError:
        return None
    try:
        mi = MediaInfo.parse(video_path)
    except Exception:
        return None
    info: Dict[str, Any] = {}
    for track in mi.tracks:
        if track.track_type == "General" and track.duration:
            try:
                info["duration_s"] = float(track.duration) / 1000.0
            except (TypeError, ValueError):
                pass
        elif track.track_type == "Video":
            if track.width and track.height:
                info["width"] = int(track.width)
                info["height"] = int(track.height)
            if track.frame_rate:
                try:
                    info["fps"] = float(track.frame_rate)
                except (TypeError, ValueError):
                    pass
            if track.format:
                info["video_codec"] = track.format
        elif track.track_type == "Audio":
            info["has_audio"] = True
            if track.format:
                info["audio_codec"] = track.format
    if "has_audio" not in info:
        info["has_audio"] = False
    return info if info.get("width") or info.get("duration_s") else None


def probe_media(video_path: str, cfg: Optional[Dict[str, Any]] = None) -> MediaInfo:
    """综合探测媒体信息，返回 MediaInfo。"""
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    info: Optional[Dict[str, Any]] = None
    fp = find_ffprobe(cfg)
    if fp:
        info = probe_with_ffprobe(fp, str(path))
    if info is None:
        ff = find_ffmpeg(cfg)
        if ff:
            info = probe_with_ffmpeg(ff, str(path))
    if info is None:
        info = probe_with_pymediainfo(str(path))
    if info is None:
        raise RuntimeError(f"无法探测媒体信息: {video_path}（请确认 ffmpeg 可用）")

    return MediaInfo(
        path=str(path),
        duration_s=float(info.get("duration_s", 0.0)),
        width=int(info.get("width", 0)),
        height=int(info.get("height", 0)),
        fps=float(info.get("fps", 0.0)),
        has_audio=bool(info.get("has_audio", False)),
        audio_codec=info.get("audio_codec", ""),
        video_codec=info.get("video_codec", ""),
    )


def orientation(info: MediaInfo) -> str:
    """横竖屏判断：高>宽 为竖屏，反之为横屏。"""
    if info.height > info.width:
        return "portrait"
    return "landscape"
