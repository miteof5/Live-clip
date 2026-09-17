"""doubao-live-clip：直播切片自动剪辑工具层。

工具层只做确定性工作：探测 / 抽音频 / 静音检测 / 词级转写 / 短语合并 / 落剪映草稿。
剪辑判断（删废话、留金句、排节奏）由豆包 LLM 完成，产出 EDL JSON 交给本工具层。
"""
from .config import load_config, find_ffmpeg, find_ffprobe, find_draft_folder
from .probe import probe_media, MediaInfo, orientation
from .preprocess import detect_silence, extract_audio, Silence
from .transcribe import transcribe_audio, Transcript, Word
from .merge_words import merge_to_phrases, Phrase, phrases_to_srt
from .edl import EDL, EDLSegment
from .make_draft import make_draft, DraftOptions

__all__ = [
    "load_config", "find_ffmpeg", "find_ffprobe", "find_draft_folder",
    "probe_media", "MediaInfo", "orientation",
    "detect_silence", "extract_audio", "Silence",
    "transcribe_audio", "Transcript", "Word",
    "merge_to_phrases", "Phrase", "phrases_to_srt",
    "EDL", "EDLSegment",
    "make_draft", "DraftOptions",
]
