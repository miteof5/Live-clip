"""story_map 单元测试：合成数据验证幕骨架/挂载/emotion/全局统计/规则回退。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liveclip.story_map import (  # noqa: E402
    build_story_map,
    load_acts_from_visual,
    load_acts_rule_based,
    load_phrases,
    synthesize_emotion,
)


# ---------------------------------------------------------------------------
# 输入解析
# ---------------------------------------------------------------------------

def test_load_phrases_from_phrases_json(tmp_path):
    p = tmp_path / "phrases.json"
    p.write_text(json.dumps([
        {"text": "大家好", "start": 0.0, "end": 1.0, "words": []},
        {"text": "今天看AI图", "start": 1.5, "end": 3.0, "words": []},
    ], ensure_ascii=False), encoding="utf-8")
    out = load_phrases(str(p))
    assert [x["text"] for x in out] == ["大家好", "今天看AI图"]


def test_load_phrases_from_words_json(tmp_path):
    p = tmp_path / "words.json"
    p.write_text(json.dumps({"words": [
        {"word": "你", "start": 0.0, "end": 0.2},
        {"word": "好", "start": 0.21, "end": 0.4},
        {"word": "啊", "start": 2.0, "end": 2.3},  # 间隔 > max_gap → 断句
    ]}, ensure_ascii=False), encoding="utf-8")
    out = load_phrases(str(p))
    assert len(out) == 2
    assert out[0]["text"] == "你好"
    assert out[1]["text"] == "啊"


# ---------------------------------------------------------------------------
# 幕骨架
# ---------------------------------------------------------------------------

def _tl(segs):
    return {"duration": 30.0, "segment_count": len(segs), "segments": segs}


def test_load_acts_from_visual():
    tl = _tl([
        {"start": 0.0, "end": 10.0, "layout": "single", "frames": 3,
         "content": ["主播单人画面"], "ui_text": [{"t": 2.0, "text": "加我粉丝团"}],
         "danmaku": [{"t": 5.0, "text": "笑死我了"}], "gifts": []},
        {"start": 10.0, "end": 30.0, "layout": "grid4", "frames": 4,
         "content": ["四宫格连麦"], "ui_text": [], "danmaku": [], "gifts": []},
    ])
    acts = load_acts_from_visual(tl)
    assert len(acts) == 2
    assert acts[0]["layout"] == "single"
    assert acts[0]["ui_text"] == ["加我粉丝团"]
    assert acts[0]["danmaku"][0]["t"] == 5.0
    assert acts[0]["visual"] == ["主播单人画面"]


def test_absorb_point_acts():
    from liveclip.story_map import _absorb_point_acts
    def mk(s, e, lay=""):
        return {"id": 0, "start": s, "end": e, "layout": lay,
                "visual": [], "ui_text": [], "danmaku": [], "gifts": [], "frames": 0}
    acts = [
        mk(1.0, 1.0, "split"),    # 开头点幕：并入下一幕
        mk(5.0, 5.0, "grid3"),    # 开头点幕
        mk(10.0, 30.0, "grid4"),  # 正常幕（吸收点幕）
        mk(40.0, 40.0, "single"), # 中间点幕
        mk(40.0, 60.0, "single"), # 正常幕
        mk(70.0, 70.0, "grid3"),  # 末尾点幕
    ]
    out = _absorb_point_acts(acts)
    assert len(out) == 2
    # 前两点幕并入：起点取最早 1.0，layout split+grid3+grid4 → mixed
    assert out[0]["start"] == 1.0 and out[0]["end"] == 30.0
    assert out[0]["layout"] == "mixed"
    # 中间点幕并入：layout 相同保持 single；末尾点幕 gap 10s > 2s → 丢弃（不虚假覆盖）
    assert out[1]["start"] == 40.0 and out[1]["end"] == 60.0
    assert out[1]["layout"] == "single"
    # 贴边点幕（gap ≤ 2s）并入前一幕 end
    out2 = _absorb_point_acts([mk(10.0, 30.0, "single"), mk(31.0, 31.0, "grid3")])
    assert out2[0]["end"] == 31.0


def test_load_acts_rule_based_on_speaker_switch():
    phrases = [{"text": "a", "start": 0.0, "end": 1.0},
               {"text": "b", "start": 2.0, "end": 3.0},
               {"text": "c", "start": 12.0, "end": 13.0}]
    speakers = [
        {"start": 0.0, "end": 3.0, "speaker_id": 0},
        {"start": 6.0, "end": 7.0, "speaker_id": 1},   # 与上段间隔 3s → 切换边界
        {"start": 12.0, "end": 13.0, "speaker_id": 0},
    ]
    acts = load_acts_rule_based(phrases, [], speakers, 20.0)
    assert len(acts) >= 2
    # 说话人切换点 6.0 附近应有幕边界
    starts = [a["start"] for a in acts]
    assert any(5.5 <= s <= 6.5 for s in starts)


# ---------------------------------------------------------------------------
# 挂载
# ---------------------------------------------------------------------------

def test_text_and_speaker_and_audio_mount():
    acts = [{"id": 0, "start": 0.0, "end": 10.0, "layout": "single", "visual": [],
             "ui_text": [], "danmaku": [], "gifts": [], "frames": 0},
            {"id": 1, "start": 10.0, "end": 20.0, "layout": "single", "visual": [],
             "ui_text": [], "danmaku": [], "gifts": [], "frames": 0}]
    phrases = [{"text": "第一幕台词", "start": 1.0, "end": 2.0},
               {"text": "第二幕台词", "start": 11.0, "end": 12.0}]
    speakers = [{"start": 0.0, "end": 5.0, "speaker_id": 0},
                {"start": 5.5, "end": 9.0, "speaker_id": 1},
                {"start": 11.0, "end": 13.0, "speaker_id": 2}]
    events = [
        {"t": 2.0, "end": 2.5, "type": "burst", "score": 1.0, "detail": {}},
        {"t": 3.0, "end": 8.0, "type": "speech_rate_jump", "score": 1.0,
         "detail": {"direction": "up"}},
        {"t": 9.0, "end": 9.5, "type": "pre_silence_peak", "score": 1.0,
         "detail": {"silence_start": 9.0, "silence_duration": 1.0}},
        {"t": 15.0, "end": 15.5, "type": "burst", "score": 1.0, "detail": {}},
    ]
    sil = [{"start": 9.0, "end": 10.0, "duration": 1.0}]

    from liveclip.story_map import _audio_stats, _speaker_stats, _text_summary
    a0 = acts[0]
    a0["text"] = _text_summary(phrases, a0["start"], a0["end"])
    a0["speakers"] = _speaker_stats(speakers, a0["start"], a0["end"])
    a0["audio"] = _audio_stats(events, sil, a0["start"], a0["end"])

    assert a0["text"]["text"] == "第一幕台词"
    assert a0["speakers"]["speakers"] == [0, 1]          # 按重叠时长排序
    assert a0["speakers"]["speaker_switches"] == 1       # 0→1 一次切换
    assert a0["audio"]["bursts"] == 1
    assert a0["audio"]["rate_jumps"]["up"] == 1
    assert len(a0["audio"]["pre_silence_peaks"]) == 1
    assert a0["audio"]["longest_silence"] == pytest.approx(1.0)


def test_text_summary_truncation():
    from liveclip.story_map import _text_summary
    phrases = [{"text": "好" * 300, "start": 0.0, "end": 1.0}]
    out = _text_summary(phrases, 0.0, 2.0)
    assert len(out["text"]) == 201  # 200 + 省略号


# ---------------------------------------------------------------------------
# emotion 合成
# ---------------------------------------------------------------------------

def test_synthesize_emotion_formula():
    acts = [
        {"id": 0, "start": 0.0, "end": 10.0, "audio": {"bursts": 5, "rate_jumps": {"up": 5, "down": 0},
                                                        "pre_silence_peaks": [], "longest_silence": 0.0}},
        {"id": 1, "start": 10.0, "end": 20.0, "audio": {"bursts": 1, "rate_jumps": {"up": 1, "down": 0},
                                                         "pre_silence_peaks": [], "longest_silence": 0.0}},
    ]
    synthesize_emotion(acts, None)
    # 无直听：分母 0.7；幕0 burst 密度与 rate 密度均全局最大 → emotion=1.0
    assert acts[0]["emotion"] == pytest.approx(1.0)
    assert acts[0]["cold_start"] is None
    assert 0.0 < acts[1]["emotion"] < 1.0


def test_cold_start_from_pre_silence_peak():
    acts = [{"id": 0, "start": 0.0, "end": 10.0,
             "audio": {"bursts": 0, "rate_jumps": {"up": 0, "down": 0},
                       "pre_silence_peaks": [{"t": 8.0, "silence_start": 8.0, "silence_duration": 2.0}],
                       "longest_silence": 2.0}}]
    synthesize_emotion(acts, None)
    assert acts[0]["cold_start"]["t"] == 8.0


# ---------------------------------------------------------------------------
# 全局统计
# ---------------------------------------------------------------------------

def test_global_stats_bgm_and_long_silence():
    from liveclip.story_map import _global_stats
    acts = [
        {"id": 0, "start": 0.0, "end": 10.0, "emotion": 0.9,
         "audio": {"bursts": 10, "rate_jumps": {"up": 5, "down": 0}, "pre_silence_peaks": [], "longest_silence": 0.0}},
        {"id": 1, "start": 10.0, "end": 30.0, "emotion": 0.2,
         "audio": {"bursts": 0, "rate_jumps": {"up": 0, "down": 0}, "pre_silence_peaks": [], "longest_silence": 3.0}},
    ]
    sil = [{"start": 15.0, "end": 22.0, "duration": 7.0}]
    st = _global_stats(acts, sil, None, 30.0)
    assert len(st["long_silences"]) == 1 and st["long_silences"][0]["dur"] == 7.0
    # 幕1 低能量 + 长静音 → 归冷场区（lull），不是 BGM
    assert len(st["lull_regions"]) == 1 and st["lull_regions"][0]["t"] == [10.0, 30.0]
    assert len(st["bgm_regions"]) == 0
    assert len(st["high_emotion_regions"]) == 1 and st["high_emotion_regions"][0]["emotion_peak"] == 0.9


def test_build_story_map_end_to_end(tmp_path):
    phrases_p = tmp_path / "phrases.json"
    phrases_p.write_text(json.dumps([{"text": "大家好", "start": 0.0, "end": 2.0}], ensure_ascii=False), encoding="utf-8")
    tl_p = tmp_path / "visual.json"
    tl_p.write_text(json.dumps(_tl([
        {"start": 0.0, "end": 20.0, "layout": "single", "frames": 2,
         "content": ["单人"], "ui_text": [], "danmaku": [], "gifts": []},
    ])), encoding="utf-8")
    sm = build_story_map(str(phrases_p), visual_timeline_path=str(tl_p))
    assert sm["duration"] == 30.0  # 视觉时间轴的 duration 字段
    assert sm["global"]["act_count"] == 1
    assert sm["acts"][0]["text"]["text"] == "大家好"
    assert "emotion" in sm["acts"][0]
    assert sm["sources"]["visual_timeline"] == str(tl_p)


# ---------------------------------------------------------------------------
# 叙事幕整形：分割 + 合并
# ---------------------------------------------------------------------------

def test_split_acts_on_pre_silence_peak():
    from liveclip.story_map import _split_acts
    acts = [{"id": 0, "start": 0.0, "end": 30.0, "layout": "single"}]
    events = [{"t": 20.0, "end": 20.5, "type": "pre_silence_peak", "score": 1.0,
               "detail": {"silence_start": 20.0, "silence_duration": 2.0}}]
    out = _split_acts(acts, events, [])
    assert len(out) == 2
    assert out[0]["start"] == 0.0 and out[0]["end"] == 20.0
    assert out[1]["start"] == 20.0 and out[1]["end"] == 30.0
    assert all(a.get("_hard") for a in out)  # 切分边界不可合并


def test_split_skips_peak_near_edge():
    from liveclip.story_map import _split_acts
    acts = [{"id": 0, "start": 0.0, "end": 10.0, "layout": "single"}]
    events = [{"t": 9.0, "end": 9.5, "type": "pre_silence_peak", "score": 1.0,
               "detail": {"silence_start": 9.0, "silence_duration": 1.0}}]
    out = _split_acts(acts, events, [])
    assert len(out) == 1  # peak 距边界 < 2s，不切


def test_merge_respects_hard_boundary():
    from liveclip.story_map import _merge_acts
    acts = [
        {"id": 0, "start": 0.0, "end": 10.0, "layout": "single", "_hard": True,
         "visual": [], "ui_text": [], "danmaku": [], "gifts": [], "frames": 0},
        {"id": 1, "start": 10.0, "end": 20.0, "layout": "single", "_hard": True,
         "visual": [], "ui_text": [], "danmaku": [], "gifts": [], "frames": 0},
    ]
    out = _merge_acts(acts)
    assert len(out) == 2  # 切分边界成对保留
    assert "_hard" not in out[0] and "_hard" not in out[1]


def test_merge_short_pair_ignores_layout():
    from liveclip.story_map import _merge_acts
    acts = [
        {"id": 0, "start": 0.0, "end": 3.0, "layout": "split",
         "visual": [], "ui_text": [], "danmaku": [], "gifts": [], "frames": 0},
        {"id": 1, "start": 3.0, "end": 8.0, "layout": "single",   # 布局不同但短
         "visual": [], "ui_text": [], "danmaku": [], "gifts": [], "frames": 0},
        {"id": 2, "start": 8.0, "end": 20.0, "layout": "split",  # 长幕布局不同 → 边界
         "visual": [], "ui_text": [], "danmaku": [], "gifts": [], "frames": 0},
    ]
    out = _merge_acts(acts)
    assert len(out) == 2  # [0,8] 合并（短幕抖动→mixed），[8,20] 独立
    assert out[0]["layout"] == "mixed"
    assert out[0]["end"] == 8.0
    assert out[1]["layout"] == "split"


def test_global_lull_vs_bgm_split():
    from liveclip.story_map import _global_stats
    acts = [
        # 冷场幕：零 burst + cold_start
        {"id": 0, "start": 0.0, "end": 20.0, "emotion": 0.1, "cold_start": {"t": 5.0},
         "audio": {"bursts": 0, "rate_jumps": {"up": 0, "down": 0}, "pre_silence_peaks": [],
                   "longest_silence": 1.0}},
        # BGM 幕：零 burst、无 cold、无长静音
        {"id": 1, "start": 20.0, "end": 40.0, "emotion": 0.1, "cold_start": None,
         "audio": {"bursts": 0, "rate_jumps": {"up": 0, "down": 0}, "pre_silence_peaks": [],
                   "longest_silence": 0.5}},
        # 说话幕：高密度（非零中位数基准）
        {"id": 2, "start": 40.0, "end": 60.0, "emotion": 0.6,
         "audio": {"bursts": 20, "rate_jumps": {"up": 5, "down": 0}, "pre_silence_peaks": [],
                   "longest_silence": 0.0}},
    ]
    st = _global_stats(acts, None, None, 60.0)
    assert [r["act_ids"] for r in st["lull_regions"]] == [[0]]
    assert [r["act_ids"] for r in st["bgm_regions"]] == [[1]]
    assert [r["act_ids"] for r in st["high_emotion_regions"]] == [[2]]


def test_high_emotion_requires_bursts():
    from liveclip.story_map import _global_stats
    # emotion 高但 burst=0（纯语速），不应进高情绪区
    acts = [{"id": 0, "start": 0.0, "end": 10.0, "emotion": 0.9, "cold_start": None,
             "audio": {"bursts": 0, "rate_jumps": {"up": 9, "down": 0}, "pre_silence_peaks": [],
                       "longest_silence": 0.0}}]
    st = _global_stats(acts, None, None, 10.0)
    assert len(st["high_emotion_regions"]) == 0


def test_load_phrases_tolerates_extra_word_fields(tmp_path):
    """词条带 ASR 后端额外字段（如 conf）不应崩：显式构造忽略多余字段。"""
    p = tmp_path / "words.json"
    p.write_text(json.dumps({"words": [
        {"word": "你", "start": 0.0, "end": 0.2, "conf": 0.98, "speaker": "A"},
        {"word": "好", "start": 0.21, "end": 0.4, "conf": 0.95},
    ]}, ensure_ascii=False), encoding="utf-8")
    out = load_phrases(str(p))
    assert out[0]["text"] == "你好"


def test_emotion_per_act_normalization_partial_listen():
    """listen 只覆盖部分幕：无探针幕按分母 0.7 归一，不被系统性压低。"""
    acts = [
        {"id": 0, "start": 0.0, "end": 10.0,
         "audio": {"bursts": 5, "rate_jumps": {"up": 5, "down": 0},
                   "pre_silence_peaks": [], "longest_silence": 0.0}},
        {"id": 1, "start": 10.0, "end": 20.0,
         "audio": {"bursts": 5, "rate_jumps": {"up": 5, "down": 0},
                   "pre_silence_peaks": [], "longest_silence": 0.0}},
    ]
    probes = {"probes": [{"t": 5.0, "excitement": 1.0}]}  # 只覆盖幕 0
    synthesize_emotion(acts, probes)
    # 两幕 burst/rate 密度相同（归一后均 1.0）：幕0 分母 1.0 → 1.0；
    # 幕1 无探针、分母 0.7 → 0.7/0.7 = 1.0（与幕0 同分，而非被压低）
    assert acts[0]["emotion"] == pytest.approx(1.0, abs=0.001)
    assert acts[1]["emotion"] == pytest.approx(1.0, abs=0.001)


def test_duration_uses_signal_max_without_visual(tmp_path):
    """无 visual_timeline 时 duration 取各信号最大 end（转写提前停不截尾）。"""
    phrases_p = tmp_path / "phrases.json"
    phrases_p.write_text(json.dumps(
        [{"text": "话", "start": 0.0, "end": 30.0}], ensure_ascii=False), encoding="utf-8")
    sil_p = tmp_path / "silences.json"
    sil_p.write_text(json.dumps([{"start": 40.0, "end": 60.0, "duration": 20.0}]),
                     encoding="utf-8")
    sm = build_story_map(str(phrases_p), silences_path=str(sil_p))
    assert sm["duration"] == 60.0       # 静音覆盖到 60s > 转写 30s
    assert sm["acts"][-1]["end"] == 60.0  # 尾幕不被截断
