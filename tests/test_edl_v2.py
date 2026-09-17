"""NormalizedEDL（v2 统一格式）测试：新旧归一化 / 校验 / 时间轴 / 映射 / 序列化。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liveclip.edl import NormalizedEDL


def test_v2_keep_parse_and_timeline():
    """v2 格式（build_edl 输出形态：source_end + target_duration）。"""
    data = {
        "keep": [{"source_start": 0.0, "source_end": 12.5},
                 {"source_start": 20.0, "source_end": 30.0}],
        "cut": [{"start": 12.5, "end": 20.0, "reason": "silence"}],
        "target_duration": 22.5, "source_duration_s": 100.0,
    }
    edl = NormalizedEDL.from_dict(data)
    assert len(edl.keep) == 2
    assert edl.keep[0].duration == 12.5            # duration 自动补齐
    assert len(edl.cut) == 1 and edl.cut[0]["end"] == 20.0
    edl.compute_target_timeline()
    assert edl.keep[1].target_start == 12.5
    assert abs(edl.target_duration_s - 22.5) < 1e-6
    assert edl.validate() == []


def test_v1_legacy_auto_normalize():
    """旧版 v1（segments[]）自动归一化为 v2。"""
    v1 = {
        "source_path": "D:/demo.mp4", "width": 1920, "height": 1080,
        "source_duration_s": 100.0,
        "segments": [{"kind": "keep", "source_start": 0.0, "source_end": 10.0,
                      "reason": "金句", "is_golden": True},
                     {"kind": "cut", "source_start": 10.0, "source_end": 20.0}],
    }
    edl = NormalizedEDL.from_dict(v1)
    assert len(edl.keep) == 1
    assert edl.keep[0].is_golden and edl.keep[0].reason == "金句"
    assert len(edl.cut) == 1
    assert edl.source_path == "D:/demo.mp4"
    edl.compute_target_timeline()
    assert abs(edl.target_duration_s - 10.0) < 1e-6


def test_v2_duration_only_fills_source_end():
    """只有 duration 无 source_end 时自动补齐 source_end。"""
    data = {"keep": [{"source_start": 5.0, "duration": 7.0}]}
    edl = NormalizedEDL.from_dict(data)
    assert abs(edl.keep[0].source_end - 12.0) < 1e-6


def test_validate_errors():
    """重叠 / 越界 / 空 keep 均应报错。"""
    edl = NormalizedEDL.from_dict({"keep": [
        {"source_start": 0.0, "source_end": 10.0},
        {"source_start": 9.0, "source_end": 15.0},   # 重叠
    ], "source_duration_s": 100.0})
    errs = edl.validate()
    assert any("重叠" in e for e in errs)

    edl2 = NormalizedEDL.from_dict({"keep": [], "source_duration_s": 10.0})
    assert any("没有保留段" in e for e in edl2.validate())


def test_mapping_cross_keep():
    """短语跨 keep 映射到目标时间轴；完全落在 cut 返回 None。"""
    edl = NormalizedEDL.from_dict({"keep": [
        {"source_start": 0.0, "source_end": 12.5},
        {"source_start": 20.0, "source_end": 30.0},
    ]})
    edl.compute_target_timeline()
    m = edl.map_phrase_to_target(11.0, 21.0)
    assert m == (11.0, 13.5)
    assert edl.map_phrase_to_target(15.0, 16.0) is None
    assert abs(edl.map_source_to_target(25.0) - 17.5) < 1e-6
    assert edl.map_source_to_target(15.0) is None


def test_roundtrip_serialize_v2_only():
    """序列化只输出 v2（无 segments 字段）；to_segments 供 draft_builder。"""
    edl = NormalizedEDL.from_dict({"keep": [
        {"source_start": 0.0, "source_end": 10.0, "video": "D:/a.mp4"},
    ], "source_path": "D:/b.mp4", "width": 1920, "height": 1440})
    edl.compute_target_timeline()
    out = edl.to_dict()
    assert "keep" in out and "segments" not in out
    assert abs(out["target_duration"] - 10.0) < 0.01
    segs = edl.to_segments()
    assert segs[0]["video"] == "D:/a.mp4"           # 段级 video 优先
    assert segs[0]["duration"] == 10.0


def test_load_save_roundtrip(tmp_path):
    """save / load 回环一致。"""
    p = tmp_path / "edl.json"
    edl = NormalizedEDL.from_dict({"keep": [{"source_start": 1.0, "source_end": 3.0}],
                                   "title": "测试"})
    edl.compute_target_timeline()
    edl.save(str(p))
    loaded = NormalizedEDL.load(str(p))
    assert loaded.title == "测试"
    assert abs(loaded.keep[0].duration - 2.0) < 1e-6


def test_unsorted_keep_auto_sort():
    """乱序 keep 输入（判断层组装 bug 防御）：自动排序，校验不误报重叠。"""
    data = {"keep": [
        {"source_start": 20.0, "source_end": 30.0},
        {"source_start": 0.0, "source_end": 10.0},
        {"source_start": 10.0, "source_end": 20.0},
    ]}
    edl = NormalizedEDL.from_dict(data)
    assert [k.source_start for k in edl.keep] == [0.0, 10.0, 20.0]   # 已排序
    assert edl.validate() == []
    edl.compute_target_timeline()
    assert edl.keep[-1].target_start == 20.0                         # 时间轴正确
    assert abs(edl.target_duration_s - 30.0) < 1e-6


def test_multivideo_keep_preserves_target_order():
    """多素材混排：列表顺序 = 目标顺序，不按 source_start 全局排序；
    校验按 video 分组（跨素材的 source_start 交叉不误报重叠）。"""
    data = {"keep": [
        {"source_start": 20.0, "source_end": 30.0, "target_start": 0.0,
         "duration": 10.0, "video": "P2.mp4"},
        {"source_start": 0.0, "source_end": 10.0, "target_start": 10.0,
         "duration": 10.0, "video": "P1.mp4"},
        {"source_start": 5.0, "source_end": 12.0, "target_start": 20.0,
         "duration": 7.0, "video": "P2.mp4"},
    ]}
    edl = NormalizedEDL.from_dict(data)
    # 保持输入顺序（目标顺序），不做全局排序
    assert [k.video for k in edl.keep] == ["P2.mp4", "P1.mp4", "P2.mp4"]
    assert edl.validate() == []   # P2 组内 20-30 与 5-12 排序后不重叠；P1 组单段
    edl.compute_target_timeline()
    assert [k.target_start for k in edl.keep] == [0.0, 10.0, 20.0]
    assert abs(edl.target_duration_s - 27.0) < 1e-6


def test_multivideo_overlap_within_same_video_reported():
    """同一 video 内源时间轴重叠仍应报错（跨 video 不算重叠）。"""
    data = {"keep": [
        {"source_start": 0.0, "source_end": 10.0, "video": "P1.mp4"},
        {"source_start": 5.0, "source_end": 15.0, "video": "P1.mp4"},   # 同素材重叠
        {"source_start": 0.0, "source_end": 8.0, "video": "P2.mp4"},    # 跨素材不重叠
    ]}
    edl = NormalizedEDL.from_dict(data)
    errs = edl.validate()
    assert any("重叠" in e for e in errs)
    assert len([e for e in errs if "重叠" in e]) == 1   # 只有 P1 组内一处重叠


if __name__ == "__main__":
    import tempfile, shutil
    d = tempfile.mkdtemp()
    try:
        test_v2_keep_parse_and_timeline()
        test_v1_legacy_auto_normalize()
        test_v2_duration_only_fills_source_end()
        test_validate_errors()
        test_mapping_cross_keep()
        test_roundtrip_serialize_v2_only()
        test_load_save_roundtrip(Path(d))
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print("\nALL EDL V2 TESTS PASSED")
