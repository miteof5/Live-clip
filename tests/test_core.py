"""merge_words 与 edl 的离线单测（不需要 ffmpeg/API/素材）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liveclip.edl import EDL, EDLSegment
from liveclip.merge_words import merge_to_phrases, phrases_to_srt, _fmt_ts
from liveclip.transcribe import Word


def _words():
    """模拟一段词级转写：停顿>0.3s 处应断句。"""
    return [
        Word("这", 0.0, 0.2),
        Word("波", 0.2, 0.4),
        Word("操", 0.5, 0.7),      # 间隔 0.1，合并
        Word("直接", 0.7, 1.1),    # 间隔 0.0
        Word("拿下", 1.1, 1.6),    # 间隔 0.0
        # 停顿 1.5s（>0.3）
        Word("兄弟", 3.1, 3.7),    # 间隔 1.5 → 新短语
        Word("们", 3.7, 3.9),
        # 停顿 0.5s（>0.3）
        Word("下", 4.4, 4.7),      # 间隔 0.5 → 新短语
        Word("一把", 4.7, 5.2),
    ]


def test_merge_basic():
    ps = merge_to_phrases(_words())
    assert len(ps) == 3, f"应为 3 个短语，实际 {len(ps)}"
    assert ps[0].text == "这波操直接拿下", ps[0].text
    assert ps[1].text == "兄弟们", ps[1].text
    assert ps[2].text == "下一把", ps[2].text
    assert abs(ps[0].start - 0.0) < 1e-6
    assert abs(ps[0].end - 1.6) < 1e-6
    print("test_merge_basic OK")


def test_merge_max_chars():
    # 连续词无停顿，靠 max_chars 强制断开
    ws = [Word("字", i * 0.2, i * 0.2 + 0.2) for i in range(30)]
    ps = merge_to_phrases(ws, max_gap=10.0, max_chars=24)
    assert len(ps) >= 2
    assert all(len(p.text) <= 24 for p in ps)
    print("test_merge_max_chars OK")


def test_srt_format():
    ps = merge_to_phrases(_words())
    srt = phrases_to_srt(ps)
    assert "00:00:00,000 --> 00:00:01,600" in srt
    assert "这波操直接拿下" in srt
    print("test_srt_format OK")


def _edl():
    return EDL(
        source_path="D:/demo/input.mp4",
        width=1920, height=1080, source_duration_s=100.0, fps=30,
        segments=[
            EDLSegment("keep", 0.0, 10.0, reason="开场金句", is_golden=True),
            EDLSegment("cut", 10.0, 20.0, reason="气口"),
            EDLSegment("keep", 20.0, 35.0, reason="高光"),
        ],
    )


def test_edl_timeline():
    edl = _edl()
    assert edl.validate() == []
    edl.compute_target_timeline()
    assert abs(edl.target_duration_s - 25.0) < 1e-6
    segs = edl.segments
    assert abs(segs[0].target_start - 0.0) < 1e-6
    assert abs(segs[2].target_start - 10.0) < 1e-6
    print("test_edl_timeline OK")


def test_edl_mapping():
    edl = _edl()
    edl.compute_target_timeline()
    # 源 5s（keep 内）→ 目标 5s
    assert abs(edl.map_source_to_target(5.0) - 5.0) < 1e-6
    # 源 15s（cut 内）→ None
    assert edl.map_source_to_target(15.0) is None
    # 源 25s（第二 keep 内）→ 目标 10 + 5 = 15
    assert abs(edl.map_source_to_target(25.0) - 15.0) < 1e-6
    # 短语跨 cut：源 8~22 → 只保留 8~10 + 20~22 → 目标 8~12
    m = edl.map_phrase_to_target(8.0, 22.0)
    assert m is not None and abs(m[0] - 8.0) < 1e-6 and abs(m[1] - 12.0) < 1e-6
    print("test_edl_mapping OK")


def test_edl_validate():
    edl = EDL(source_path="x", width=1, height=1, source_duration_s=10,
              segments=[EDLSegment("cut", 0.0, 5.0)])
    assert edl.validate()  # 无 keep 段应报错
    edl2 = EDL(source_path="x", width=1, height=1, source_duration_s=10,
               segments=[EDLSegment("keep", 5.0, 12.0)])
    assert edl2.validate()  # 越界应报错
    print("test_edl_validate OK")


def test_fmt_ts():
    assert _fmt_ts(3661.5) == "01:01:01,500"
    assert _fmt_ts(0.1234) == "00:00:00,123"
    print("test_fmt_ts OK")


if __name__ == "__main__":
    test_merge_basic()
    test_merge_max_chars()
    test_srt_format()
    test_edl_timeline()
    test_edl_mapping()
    test_edl_validate()
    test_fmt_ts()
    print("\nALL TESTS PASSED")
