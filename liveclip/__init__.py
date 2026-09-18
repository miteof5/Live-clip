"""doubao-live-clip：直播切片自动剪辑工具层。

工具层只做确定性工作：探测 / 抽音频 / 静音检测 / 词级转写 / 短语合并 / 落剪映草稿。
剪辑判断（删废话、留金句、排节奏）由豆包 LLM 完成，产出 EDL JSON 交给本工具层。

**EDL 唯一标准格式（v2）**：{"keep": [{"source_start","source_end",...}], ...}
见 liveclip.edl.NormalizedEDL 的模块文档。

公开 API（轻量模块，顶层导入）：
- config / probe / preprocess / transcribe / merge_words / edl / styles
- draft_builder / make_draft / jy_draftc / enrich_edl

重型依赖模块（import cv2 / requests 异步，按需导入，避免顶层 import 变慢）：
- visual_signal（opencv）/ speaker_diarization（百炼异步）/ speaker_binding
"""
from __future__ import annotations

from .config import (
    ConfigError,
    load_config,
    find_ffmpeg,
    find_ffprobe,
    find_draft_folder,
    output_dir,
    load_asr_backends,
    resolve_api_key,
    normalize_backend,
)
from .probe import probe_media, MediaInfo, orientation
from .preprocess import detect_silence, extract_audio, Silence, find_chunk_boundaries
from .transcribe import (
    transcribe_audio,
    transcribe_with_backends,
    save_transcript,
    Transcript,
    Word,
    TranscribeError,
)
from .merge_words import merge_to_phrases, Phrase, phrases_to_srt, save_srt
from .edl import NormalizedEDL, KeepSegment
from .styles import subtitle_style, DEFAULT_RATIO
from .draft_builder import build_draft_from_edl
from .make_draft import make_draft, DraftOptions
from .jy_draftc import load_draft_content, decrypt_file, encrypt_file
from .enrich_edl import enrich_edl, interval_speaker_stats

__all__ = [
    # config
    "ConfigError", "load_config", "find_ffmpeg", "find_ffprobe", "find_draft_folder",
    "output_dir", "load_asr_backends", "resolve_api_key", "normalize_backend",
    # probe / preprocess / transcribe / merge
    "probe_media", "MediaInfo", "orientation",
    "detect_silence", "extract_audio", "Silence", "find_chunk_boundaries",
    "transcribe_audio", "transcribe_with_backends", "save_transcript",
    "Transcript", "Word", "TranscribeError",
    "merge_to_phrases", "Phrase", "phrases_to_srt", "save_srt",
    # edl（v2 统一模型）
    "NormalizedEDL", "KeepSegment",
    # styles / draft
    "subtitle_style", "DEFAULT_RATIO",
    "build_draft_from_edl", "make_draft", "DraftOptions",
    # 剪映草稿读写
    "load_draft_content", "decrypt_file", "encrypt_file",
    # 说话人富化
    "enrich_edl", "interval_speaker_stats",
]

# 重型模块（opencv / 异步 API）：from liveclip import visual_signal 时惰性加载
_LAZY_MODULES = {
    "visual_signal": "liveclip.visual_signal",
    "speaker_diarization": "liveclip.speaker_diarization",
    "speaker_binding": "liveclip.speaker_binding",
    "storyboard": "liveclip.storyboard",
}


def __getattr__(name: str):
    """惰性加载重型模块：from liveclip import visual_signal 触发时导入。"""
    if name in _LAZY_MODULES:
        import importlib
        module = importlib.import_module(_LAZY_MODULES[name])
        globals()[name] = module
        return module
    raise AttributeError(f"module 'liveclip' has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY_MODULES.keys()))
