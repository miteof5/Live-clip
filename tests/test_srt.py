"""SRT 时间戳格式化统一测试：liveclip.merge_words._fmt_ts 与 build_edl 复用一致。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liveclip.merge_words import _fmt_ts
from scripts.build_edl import build, map_srt, write_srt, fmt_srt_time


def test_fmt_ts_known_values():
    assert _fmt_ts(3661.5) == "01:01:01,500"
    assert _fmt_ts(0.1234) == "00:00:00,123"
    assert _fmt_ts(59.999) == "00:01:00,000" or _fmt_ts(59.999) == "00:00:59,999"


def test_fmt_ts_unified_with_build_edl():
    """build_edl.fmt_srt_time 与 merge_words._fmt_ts 是同一实现。"""
    assert fmt_srt_time is _fmt_ts
    assert fmt_srt_time(3661.5) == _fmt_ts(3661.5)


def test_build_edl_output_v2_format():
    """build_edl 输出保持 v2 标准：keep 段含 source_start/source_end/target_start/duration。"""
    phrases = [
        {"start": 0.0, "end": 3.0, "text": "开头"},
        {"start": 5.0, "end": 8.0, "text": "正文"},
    ]
    edl = build(phrases, duration=20.0)
    assert "keep" in edl and "cut" in edl and "target_duration" in edl
    k = edl["keep"][0]
    for field in ("source_start", "source_end", "target_start", "duration"):
        assert field in k, f"keep 段缺少 {field}"


def test_map_srt_remap():
    phrases = [{"start": 0.0, "end": 3.0, "text": "开头"},
               {"start": 5.0, "end": 8.0, "text": "正文"}]
    edl = build(phrases, duration=20.0)
    entries = map_srt(phrases, edl["keep"])
    assert len(entries) == 2
    assert entries[0]["text"] == "开头"


if __name__ == "__main__":
    test_fmt_ts_known_values()
    test_fmt_ts_unified_with_build_edl()
    test_build_edl_output_v2_format()
    test_map_srt_remap()
    print("\nALL SRT TESTS PASSED")
