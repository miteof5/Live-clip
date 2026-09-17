"""词级时间戳 → 短语/句子合并。

whisper 中文输出的 word 是字/词混合粒度，直接逐词展示会碎。
本模块按时间间隔把相邻词合并成短语（剪辑的最小语义单元），并生成 SRT。

合并逻辑借鉴 VectCutAPI pattern/001-words.py 的 group_sentences 思路：
- 相邻词间隔 ≤ max_gap 且短语未超 max_chars → 并入当前短语
- 否则新开短语（即"停顿"作为短语边界，停顿也是剪辑切点的候选）
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List

from .transcribe import Word


@dataclass
class Phrase:
    text: str
    start: float
    end: float
    words: List[Dict[str, Any]]  # 原始词列表（含概率）

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def merge_to_phrases(
    words: List[Word],
    max_gap: float = 0.3,
    max_chars: int = 24,
) -> List[Phrase]:
    """按时间间隔把词合并为短语。

    规则：
    1. 相邻词间隔 > max_gap → 断句（停顿边界）
    2. 短语累计字符数 > max_chars → 强制断开
    """
    if not words:
        return []
    phrases: List[Phrase] = []
    cur_words: List[Word] = [words[0]]
    cur_chars = len(words[0].word)

    def flush() -> None:
        nonlocal cur_words, cur_chars
        if not cur_words:
            return
        text = "".join(w.word for w in cur_words)
        phrases.append(Phrase(
            text=text,
            start=cur_words[0].start,
            end=cur_words[-1].end,
            words=[w.to_dict() for w in cur_words],
        ))
        cur_words = []
        cur_chars = 0

    for w in words[1:]:
        gap = w.start - cur_words[-1].end
        if gap > max_gap or cur_chars + len(w.word) > max_chars:
            flush()
        cur_words.append(w)
        cur_chars += len(w.word)
    flush()
    return phrases


def phrases_to_srt(phrases: List[Phrase], min_duration: float = 0.6) -> str:
    """短语 → SRT 字幕文本（毫秒时间戳）。

    过短的短语扩展至 min_duration，避免字幕闪跳。
    """
    lines: List[str] = []
    for i, p in enumerate(phrases, start=1):
        start = p.start
        end = max(p.end, p.start + min_duration)
        lines.append(str(i))
        lines.append(f"{_fmt_ts(start)} --> {_fmt_ts(end)}")
        lines.append(p.text)
        lines.append("")
    return "\n".join(lines)


def save_srt(phrases: List[Phrase], out_path: str) -> str:
    from pathlib import Path
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(phrases_to_srt(phrases), encoding="utf-8")
    return str(p)


def _fmt_ts(seconds: float) -> str:
    """秒 → SRT 时间戳 HH:MM:SS,mmm"""
    ms = int(round(seconds * 1000))
    h, rem = divmod(ms, 3600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
