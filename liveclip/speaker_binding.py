"""speaker_binding.py — 说话人 ↔ 画面位置绑定（融合层）。

输入：
- speaker_timeline.json（speaker_diarization 输出）：
  segments[{start, end, text, speaker_id, words[]}]
- visual_events.json（visual_signal 输出）：
  face_events[{t, n_faces, subjects[{track_id, region, ...}], dominant}]

算法（三步）：
1. 时间对齐：每个 speech segment [s,e] 窗口内的视觉采样点，
   dominant.region 投票 → 该段的画面主导区域
2. 投票绑定：每个 speaker_id 累计其所有段的区域票 →
   speaker → region 映射表（置信度 = 得票占比）
3. 基率校验 + 兜底：
   - 整片各区域"一直有人"的基率（base_rate）一并输出——基率高的位置，
     不能单靠位置判断说话人（借鉴 VideoHighlighter signal_combinations.py）
   - 无视觉数据/低置信度 → region=null、bind_confidence=0，不中断
     （借鉴 video-auto-edit-agent 的 baseline fallback）

用法（CLI）：
    python liveclip/speaker_binding.py --audio speaker_timeline.json --visual visual_events.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:  # 模块方式
    from .config import load_config
except ImportError:  # 直接运行
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from liveclip.config import load_config


# 视觉采样点落入语音段的容差（秒）：语音边界与画面切换常有 ~0.5s 错位
TIME_PAD = 0.5


# ---------------------------------------------------------------------------
# 1. 时间对齐：段的画面主导区域
# ---------------------------------------------------------------------------
def segment_dominant_region(
    seg_start: float,
    seg_end: float,
    face_events: List[Dict[str, Any]],
    *,
    pad: float = TIME_PAD,
) -> Tuple[Optional[str], Dict[str, int]]:
    """统计段窗口内每个区域的 dominant 票数，返回 (票最多区域, 各区域票数)。

    采样点无 dominant（画面无脸）不投票。
    """
    votes: Dict[str, int] = {}
    for e in face_events:
        t = e.get("t", 0.0)
        if not (seg_start - pad <= t <= seg_end + pad):
            continue
        dom = e.get("dominant")
        if dom:
            r = str(dom.get("region") or "whole")
            votes[r] = votes.get(r, 0) + 1
    if not votes:
        return None, {}
    best = max(votes, key=lambda r: votes[r])
    return best, votes


# ---------------------------------------------------------------------------
# 2. 全局投票：speaker → region 映射
# ---------------------------------------------------------------------------
def build_binding_map(
    segments: List[Dict[str, Any]],
    face_events: List[Dict[str, Any]],
) -> Tuple[Dict[int, Dict[str, Any]], List[Optional[str]]]:
    """每个 speaker_id 累计其所有段的区域票 → 绑定映射。

    返回 (binding_map, 每段的 best_region 列表，顺序与 segments 一致)。

    binding_map[spk] = {region, confidence, votes}
    confidence = 该区域得票 / 该 speaker 总得票。
    """
    seg_regions: List[Optional[str]] = []
    for seg in segments:
        best, _ = segment_dominant_region(seg.get("start", 0.0),
                                          seg.get("end", 0.0), face_events)
        seg_regions.append(best)

    spk_votes: Dict[int, Dict[str, int]] = {}
    for seg, best in zip(segments, seg_regions):
        spk = seg.get("speaker_id")
        if spk is None or best is None:
            continue
        d = spk_votes.setdefault(spk, {})
        d[best] = d.get(best, 0) + 1

    binding_map: Dict[int, Dict[str, Any]] = {}
    for spk, votes in spk_votes.items():
        total = sum(votes.values())
        best_region = max(votes, key=lambda r: votes[r])
        binding_map[spk] = {
            "region": best_region,
            "confidence": round(votes[best_region] / total, 3) if total else 0.0,
            "votes": total,
        }
    return binding_map, seg_regions


# ---------------------------------------------------------------------------
# 3. 基率校验（借鉴 signal_combinations：组合是否真的罕见）
# ---------------------------------------------------------------------------
def compute_region_base_rate(face_events: List[Dict[str, Any]]) -> Dict[str, float]:
    """整片中各区域成为 dominant 的采样占比（"该位置一直有人"的基率）。

    基率高（如 left=0.95）说明左脸几乎全程存在——此时"speaker 在左"不构成
    强证据，判断层应结合 motion/音频轮次再定。
    """
    total = len(face_events)
    if total == 0:
        return {}
    counts: Dict[str, int] = {}
    for e in face_events:
        dom = e.get("dominant")
        if dom:
            r = str(dom.get("region") or "whole")
            counts[r] = counts.get(r, 0) + 1
    return {r: round(c / total, 3) for r, c in counts.items()}


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def bind_speakers(
    speaker_timeline: Dict[str, Any],
    visual_events: Dict[str, Any],
    *,
    min_confidence: float = 0.0,
) -> Dict[str, Any]:
    """执行完整绑定，输出 bound_speaker_timeline.json 结构。

    - segments[].region：音频段的画面主导区域（无视觉数据为 None）
    - segments[].bind_confidence：绑定可信度（基于该 speaker 全局映射置信度）
    - segments[].region_conflict：段级区域与全局映射冲突（抢话/切换瞬间/误检）
    - binding_map：speaker → region 映射
    - regions_base_rate：各区域基率（供判断层参考，不参与硬过滤）
    """
    segments = speaker_timeline.get("segments", [])
    face_events = visual_events.get("face_events", [])
    binding_map, seg_regions = build_binding_map(segments, face_events)
    base_rate = compute_region_base_rate(face_events)

    out_segs: List[Dict[str, Any]] = []
    for seg, best in zip(segments, seg_regions):
        spk = seg.get("speaker_id")
        region: Optional[str] = best
        bind_conf = 0.0
        conflict = False
        if spk is not None and spk in binding_map:
            m = binding_map[spk]
            if best is None:
                # 无视觉数据兜底：沿用映射区域但置信度置 0（无法验证）
                region = m["region"]
                bind_conf = 0.0
            else:
                bind_conf = m["confidence"]
                if best != m["region"]:
                    conflict = True  # 段级证据与全局映射不一致
                if bind_conf < min_confidence:
                    region = None  # 映射太弱，不标位置
        rec = dict(seg)
        rec["region"] = region
        rec["bind_confidence"] = round(bind_conf, 3)
        rec["region_conflict"] = conflict
        out_segs.append(rec)

    return {
        "video": speaker_timeline.get("video"),
        "duration": speaker_timeline.get("duration"),
        "speakers_total": speaker_timeline.get("speakers_total"),
        "binding_map": binding_map,
        "regions_base_rate": base_rate,
        "segments": out_segs,
    }


def save_bound(out: Dict[str, Any], out_path: str) -> str:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(p)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(prog="speaker_binding.py", description="说话人↔位置绑定")
    ap.add_argument("--audio", required=True, help="speaker_timeline.json（speaker_diarization 输出）")
    ap.add_argument("--visual", required=True, help="visual_events.json（visual_signal 输出）")
    ap.add_argument("--out", default=None, help="输出路径（默认 outputs/work/bound_speaker_timeline.json）")
    args = ap.parse_args()

    cfg = load_config()
    out = args.out or str(Path(cfg.get("output_dir", "outputs")) / "work" / "bound_speaker_timeline.json")

    audio = json.loads(Path(args.audio).read_text(encoding="utf-8"))
    visual = json.loads(Path(args.visual).read_text(encoding="utf-8"))
    bound = bind_speakers(audio, visual)
    save_bound(bound, out)

    print(json.dumps({
        "ok": True,
        "speakers_total": bound["speakers_total"],
        "binding_map": bound["binding_map"],
        "regions_base_rate": bound["regions_base_rate"],
        "segments": len(bound["segments"]),
        "with_region": sum(1 for s in bound["segments"] if s.get("region")),
        "with_conflict": sum(1 for s in bound["segments"] if s.get("region_conflict")),
        "out": out,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
