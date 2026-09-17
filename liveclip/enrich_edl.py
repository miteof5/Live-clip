"""enrich_edl.py — EDL 说话人富化（判断层辅助）。

把 bound_speaker_timeline.json（融合层输出）合并进 v2 标准 EDL：
- 每个 keep 段：重叠时长最长的 speaker 为该段 speaker_id
- dialogue_zoom：段内存在说话人切换（≥2 个不同 speaker 交替）→ True
  这是"谁说话放大谁"的镜头切换预标记
- speaker_switches：段内切换点列表（判断层可据此下 dialogue_zoom 指令）

**输入格式（统一 v2）**：{"keep": [...], ...}。自动兼容：
- 旧版 v1 {"segments": [...]}（EDL dataclass 格式）
- judged 清单（[{keep, source_start, source_end, ...}]，判断层中间产物）→ 自动包装
输出统一为 v2 格式（keep 段补 speaker_id / dialogue_zoom + 顶层 speaker_switches）。

用法（CLI）：
    python liveclip/enrich_edl.py --edl merged_edl.json --speaker bound_speaker_timeline.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:  # 模块方式
    from .config import load_config
    from .edl import NormalizedEDL
except ImportError:  # 直接运行
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from liveclip.config import load_config
    from liveclip.edl import NormalizedEDL


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def interval_speaker_stats(
    start: float,
    end: float,
    spk_segments: List[Dict[str, Any]],
    *,
    min_switch_overlap: float = 0.3,
) -> Tuple[Optional[int], int, List[Dict[str, Any]]]:
    """统计源区间 [start,end] 内的说话人重叠时长。

    返回 (主 speaker_id, 段内不同 speaker 数, 切换点列表)。
    切换点 = 相邻语音段 speaker 变化的位置（取后段起点）；
    仅当切换两侧的语音段与区间重叠都 ≥ min_switch_overlap 才记录，
    避免段边界的几毫秒跨界语音凑出假切换（如冷场段混入 BGM 歌词）。
    """
    dur: Dict[int, float] = {}
    ordered: List[Tuple[float, float, float, Optional[int]]] = []
    for s in spk_segments:
        ov = _overlap(start, end, s.get("start", 0.0), s.get("end", 0.0))
        if ov <= 0:
            continue
        spk = s.get("speaker_id")
        if spk is not None:
            dur[spk] = dur.get(spk, 0.0) + ov
        ordered.append((s.get("start", 0.0), s.get("end", 0.0), ov, spk))

    if not dur:
        return None, 0, []

    main_spk = max(dur, key=lambda k: dur[k])
    ordered.sort(key=lambda x: x[0])
    switches: List[Dict[str, Any]] = []
    prev_spk = None
    prev_ov = 0.0
    for t_start, _t_end, ov, spk in ordered:
        if (spk is not None and prev_spk is not None and spk != prev_spk
                and ov >= min_switch_overlap and prev_ov >= min_switch_overlap):
            switches.append({"t": round(t_start, 3), "from": prev_spk, "to": spk})
        if spk is not None:
            prev_spk = spk
            prev_ov = ov
    return main_spk, len(dur), switches


def enrich_edl(
    edl: NormalizedEDL,
    bound_timeline: Dict[str, Any],
) -> NormalizedEDL:
    """给 EDL 的 keep 段补说话人字段（in-place 修改并返回）。"""
    spk_segments = bound_timeline.get("segments", [])
    all_switches: List[Dict[str, Any]] = []
    for seg in edl.keep:
        main_spk, _n, switches = interval_speaker_stats(
            seg.source_start, seg.source_end, spk_segments)
        seg.speaker_id = main_spk
        seg.dialogue_zoom = bool(switches)
        all_switches.extend(switches)
    edl.speaker_switches = sorted(all_switches, key=lambda x: x["t"])
    return edl


def _load_input(raw_path: str) -> NormalizedEDL:
    """读取输入并归一化为 v2 标准 EDL。

    - {"keep": [...]}（v2）→ 直接解析
    - {"segments": [...]}（v1）→ 自动归一化
    - [...]（judged 清单）→ 包装成 keep 格式（keep=True 项）
    """
    raw = json.loads(Path(raw_path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        return NormalizedEDL.from_dict(raw)
    if isinstance(raw, list):
        keeps = [k for k in raw if isinstance(k, dict) and k.get("keep")]
        data = {"keep": keeps,
                "source_duration_s": max((float(k.get("source_end", 0.0))
                                          for k in keeps), default=0.0)}
        return NormalizedEDL.from_dict(data)
    raise ValueError(f"无法识别的 EDL 输入: {raw_path}（应为 dict 或 list）")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(prog="enrich_edl.py", description="EDL 说话人富化")
    ap.add_argument("--edl", required=True, help="EDL JSON（v2 keep 格式；兼容 v1 / judged 清单）")
    ap.add_argument("--speaker", required=True, help="bound_speaker_timeline.json")
    ap.add_argument("--out", default=None, help="输出路径（默认 <edl 同目录>_speaker.json）")
    args = ap.parse_args()

    edl = _load_input(args.edl)
    bound = json.loads(Path(args.speaker).read_text(encoding="utf-8"))
    out = args.out or str(Path(args.edl).with_name(Path(args.edl).stem + "_speaker.json"))

    enrich_edl(edl, bound)
    edl.save(out)

    keeps = edl.keep
    print(json.dumps({
        "ok": True,
        "keep_segments": len(keeps),
        "with_speaker": sum(1 for k in keeps if k.speaker_id is not None),
        "dialogue_zoom": sum(1 for k in keeps if k.dialogue_zoom),
        "speaker_switches": len(edl.speaker_switches),
        "out": out,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
