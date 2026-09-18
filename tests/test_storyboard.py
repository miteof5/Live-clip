"""storyboard 单元测试：候选排序 / 预算前缀 / 去重 / 指标（纯函数，不碰 ffmpeg）。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liveclip.storyboard import (  # noqa: E402
    avg_gap,
    build_visual_timeline,
    frames_for_budget,
    moment_coverage,
    rank_frame_candidates,
    scene_recall,
)


def _frame(t, desc):
    return {"t": t, "desc": desc}


def test_timeline_layout_normalization_and_danmaku_t():
    # 措辞不同的"桌面截图"归一到 desktop；弹幕带出现时刻
    sb = {"frames": [
        _frame(1.0, "布局：电脑桌面截图，含多个窗口叠加；文字：Benny 送你一鸣茶考拉 11个星光闪；内容：桌面"),
        _frame(5.0, "布局：多层窗口叠加的桌面截图；文字：小K：挺还原；内容：不可见"),
        _frame(9.0, "布局：电脑桌面截图；文字：与上帧同"),
    ]}
    tl = build_visual_timeline(sb)
    assert tl["segment_count"] == 1
    seg = tl["segments"][0]
    assert seg["layout"] == "desktop"
    assert seg["gifts"][0]["text"].startswith("Benny")
    assert seg["gifts"][0]["t"] == 1.0
    assert seg["danmaku"][0]["text"] == "小K：挺还原"


def test_timeline_split_on_grid_change():
    # 单画面 → 四宫格 → 新幕
    sb = {"frames": [
        _frame(1.0, "布局：单画面；文字：无"),
        _frame(20.0, "布局：四宫格；文字：加我粉丝团"),
    ]}
    tl = build_visual_timeline(sb)
    assert tl["segment_count"] == 2
    assert [s["layout"] for s in tl["segments"]] == ["single", "grid4"]


def test_timeline_meta_line_filtered():
    # "文字模糊/不可辨"等元描述行不进入 danmaku
    sb = {"frames": [
        _frame(1.0, "布局：四宫格；文字：左上角挂件文字\"加我粉丝团\"，其余文字模糊不可辨"),
    ]}
    tl = build_visual_timeline(sb)
    seg = tl["segments"][0]
    assert all("模糊" not in x["text"] for x in seg["ui_text"])
    assert any("粉丝团" in x["text"] for x in seg["ui_text"])


def _scene(ts, score=10.0):
    return [{"t": t, "score": score} for t in ts]


def test_scene_anchors_rank_first():
    ranked = rank_frame_candidates(
        100.0, _scene([10.0, 50.0]), [],
        merge_window=0.8, max_candidates=100, fill_grid_step=5.0,
    )
    top = frames_for_budget(ranked, 2)
    # 预算极小也要先保场景锚点
    assert [c.t for c in top] == [10.0, 50.0]
    assert all(c.anchor for c in top)


def test_dedup_merge_window():
    # 两个切换点相距 0.3s（< 0.8s 窗口）→ 只保留 score 高的那个
    ranked = rank_frame_candidates(
        100.0, [{"t": 10.0, "score": 5.0}, {"t": 10.3, "score": 20.0}], [],
        merge_window=0.8, max_candidates=100, fill_grid_step=5.0,
    )
    anchors = [c for c in ranked if c.anchor]
    assert len(anchors) == 1
    assert anchors[0].t == 10.3


def test_budget_prefix_property():
    ranked = rank_frame_candidates(
        200.0, _scene([20.0, 80.0, 140.0]), [],
        merge_window=0.8, max_candidates=200, fill_grid_step=2.0,
    )
    small = {round(c.t, 2) for c in frames_for_budget(ranked, 60)}
    large = {round(c.t, 2) for c in frames_for_budget(ranked, 120)}
    assert small.issubset(large)
    assert len(large) > len(small)


def test_max_candidates_cap():
    ranked = rank_frame_candidates(
        5000.0, [], [],
        merge_window=0.8, max_candidates=120, fill_grid_step=1.0,
    )
    assert len(ranked) <= 120


def test_fill_favors_large_gaps():
    ranked = rank_frame_candidates(
        100.0, _scene([50.0]), [],
        merge_window=0.8, max_candidates=200, fill_grid_step=1.0,
    )
    fills = [c for c in ranked if c.tier == 2]
    fills.sort(key=lambda c: c.score, reverse=True)
    # 距锚点 50s 处空隙最大，优先补
    assert abs(fills[0].t - 0.0) < 1.0 or abs(fills[0].t - 100.0) < 1.0


def test_small_budget_spreads_across_timeline():
    # 冻结锚点密集且都在前半段时，小预算不能全挤在开头：
    # 61 个冻结锚点均匀分布 0-600s，b30 应铺开而不是取前 28 个
    frozen = [{"start": 10.0 * (i + 1), "end": 10.0 * (i + 1) + 3.0} for i in range(61)]
    ranked = rank_frame_candidates(
        600.0, [], frozen,
        merge_window=0.8, max_candidates=360, fill_grid_step=1.0,
    )
    top = frames_for_budget(ranked, 30)
    ts = [c.t for c in top]
    # 覆盖范围：首尾都要有帧
    assert max(ts) > 500.0
    assert min(ts) < 100.0
    # 最大相邻间隙不应超过约 120s（600/30 的 3 倍内）
    gaps = [b - a for a, b in zip(ts, ts[1:])]
    assert max(gaps) < 120.0


def test_scene_anchors_stay_priority_in_small_budget():
    # 场景锚点少且价值高：b=3 时两个场景点都必须进，再加一个补帧
    ranked = rank_frame_candidates(
        100.0, _scene([20.0, 80.0]), [],
        merge_window=0.8, max_candidates=200, fill_grid_step=1.0,
    )
    top = frames_for_budget(ranked, 3)
    ts = [round(c.t) for c in top]
    assert 20 in ts and 80 in ts


def test_no_duplicate_frames():
    # 回归：FPS 循环必须移除已选点，预算>候选池时返回池大小且无重复
    ranked = rank_frame_candidates(
        100.0, _scene([20.0, 80.0]), [],
        merge_window=0.8, max_candidates=200, fill_grid_step=1.0,
    )
    big = frames_for_budget(ranked, 500)
    ts = [round(c.t, 3) for c in big]
    assert len(ts) == len(set(ts))          # 无重复帧
    assert len(ts) <= len(ranked)           # 不超候选池
    assert len(big) == len(ranked)          # 预算充足时全池


def test_scene_recall_metric():
    assert scene_recall([10.0, 20.0], [10.5, 25.0], tol=2.0) == 0.5
    assert scene_recall([10.0], [], tol=2.0) is None


def test_moment_coverage_metric():
    # 帧落在 [s-1, e+1] 内即算覆盖
    assert moment_coverage([30.0], [(29.5, 31.0)]) == 1.0
    assert moment_coverage([5.0], [(20.0, 25.0)]) == 0.0
    assert moment_coverage([1.0], []) is None


def test_avg_gap_metric():
    # 两帧覆盖 0-10s：探测网格平均最近距离应较小
    g = avg_gap([2.0, 8.0], 10.0)
    assert 0.0 < g < 2.0
