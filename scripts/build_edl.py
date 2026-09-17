"""build_edl.py — 由短语级转写生成简版剪辑决策（EDL）+ 映射后字幕。

当前规则（先跑通，后续可换"剪法"规则集）：
- 保留所有有语音的短语区间（前后各留 lead 秒缓冲）
- 句间停顿 <= pause_keep 秒的保留（自然停顿），> pause_keep 的剪掉（长沉默）
- 输出：
  - edl.json：keep（源区间→目标区间映射）、cut、目标总时长
  - mapped.srt：剪掉静音后新时间轴上的字幕
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 允许从项目根直接运行 scripts/build_edl.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liveclip.merge_words import _fmt_ts  # noqa: E402

fmt_srt_time = _fmt_ts  # 统一 SRT 时间戳格式化（唯一实现：liveclip.merge_words._fmt_ts）


def build(phrases: list[dict], duration: float, *, pause_keep: float = 0.8, lead: float = 0.15) -> dict:
    # 1) 短语 → 保留窗口（前后缓冲）
    windows = [(max(0.0, p["start"] - lead), min(duration, p["end"] + lead)) for p in phrases]
    # 防御：窗口倒序/越界
    windows = sorted(windows)
    # 2) 合并间隙 <= pause_keep 的相邻窗口
    merged: list[tuple[float, float]] = []
    for s, e in windows:
        if merged and s - merged[-1][1] <= pause_keep:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    # 3) 映射到新时间线
    keep, cursor = [], 0.0
    for s, e in merged:
        dur = e - s
        keep.append({"source_start": round(s, 3), "source_end": round(e, 3),
                     "target_start": round(cursor, 3), "duration": round(dur, 3)})
        cursor += dur
    # cut 段 = 被剪掉的长静音（保留区间之间的缺口）
    cut = []
    for i in range(len(merged) - 1):
        gap_s, gap_e = merged[i][1], merged[i + 1][0]
        if gap_e - gap_s > 0.01:
            cut.append({"start": round(gap_s, 3), "end": round(gap_e, 3), "reason": "silence"})
    return {"keep": keep, "cut": cut, "target_duration": round(cursor, 3)}


def map_srt(phrases: list[dict], keep: list[dict]) -> list[dict]:
    """把短语字幕重映射到新时间轴。"""
    out = []
    for p in phrases:
        seg = None
        for k in keep:
            if k["source_start"] - 0.01 <= p["start"] <= k["source_end"] + 0.01:
                seg = k
                break
        if seg is None:
            continue
        offset = seg["source_start"] - seg["target_start"]
        out.append({
            "index": len(out) + 1,
            "start": p["start"] - offset,
            "end": p["end"] - offset,
            "text": p["text"],
        })
    return out


def write_srt(entries: list[dict], path: Path) -> None:
    lines = []
    for e in entries:
        lines.append(str(e["index"]))
        lines.append(f"{fmt_srt_time(e['start'])} --> {fmt_srt_time(e['end'])}")
        lines.append(e["text"].strip())
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phrases", required=True, help="phrases.json 路径")
    ap.add_argument("--duration", type=float, required=True, help="原视频时长（秒）")
    ap.add_argument("--pause-keep", type=float, default=0.8, help="句间停顿保留阈值（秒）")
    ap.add_argument("--lead", type=float, default=0.15, help="每段前后缓冲（秒）")
    ap.add_argument("--out", default="outputs/work/edl.json")
    ap.add_argument("--srt", default="outputs/work/mapped.srt")
    args = ap.parse_args()

    phrases = json.loads(Path(args.phrases).read_text(encoding="utf-8"))
    edl = build(phrases, args.duration, pause_keep=args.pause_keep, lead=args.lead)
    entries = map_srt(phrases, edl["keep"])

    out_p = Path(args.out); out_p.parent.mkdir(parents=True, exist_ok=True)
    srt_p = Path(args.srt)
    out_p.write_text(json.dumps(edl, ensure_ascii=False, indent=2), encoding="utf-8")
    write_srt(entries, srt_p)

    print(json.dumps({
        "ok": True,
        "source_duration": args.duration,
        "target_duration": edl["target_duration"],
        "keep_segments": len(edl["keep"]),
        "cut_segments": len(edl["cut"]),
        "srt_entries": len(entries),
        "edl": str(out_p), "srt": str(srt_p),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
