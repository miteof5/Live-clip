"""test_storyboard_budgets.py — 抽帧数测试跑批（历史实验脚本，非正式入口）。

同一视频不同预算 → 指标对比：
1. 每个视频只跑一次 scdet / freezedetect / 全量抽帧（候选池 top N）
2. 各预算（30/60/120/180/240）取候选前缀 → 计算覆盖指标
3. 输出对比表 + 各预算 storyboard.json

指标（判断层"画面理解"的信息量代理）：
- scene_recall@2s：场景切换点被 ±2s 内帧覆盖的比例
- moment_coverage：已知关键时间段被帧覆盖的比例
- avg_gap：1s 网格上最近帧平均距离（时间分辨率）

用法（视频路径用 --video 传入，勿硬编码本机路径）：
    .venv\\Scripts\\python.exe scripts\\scratch\\test_storyboard_budgets.py --video P1=D:\\素材\\P1.mp4 --video P2=D:\\素材\\P2.mp4
    .venv\\Scripts\\python.exe scripts\\scratch\\test_storyboard_budgets.py --video P1=D:\\素材\\P1.mp4 --describe
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from liveclip.config import find_ffmpeg, load_config, output_dir  # noqa: E402
from liveclip.probe import probe_media  # noqa: E402
from liveclip.storyboard import (  # noqa: E402
    avg_gap,
    extract_frames,
    frames_for_budget,
    moment_coverage,
    rank_frame_candidates,
    save_storyboard,
    scene_recall,
)
from liveclip.visual_signal import extract_frozen_segments, extract_scene_changes  # noqa: E402

BUDGETS = [30, 60, 120, 180, 240]
MAX_CANDIDATES = 360
MERGE_WINDOW = 0.8

# 关键时间段（取自 judged 表人工确认内容；按 --video 名称匹配，可自行补充）
GT_MOMENTS = {
    "P1": [
        (12.33, 14.21, "开场评价"),
        (16.29, 19.29, "翻相册说明"),
        (29.51, 32.27, "评价浅头发"),
        (74.20, 81.80, "看AI手艺人"),
        (129.06, 135.66, "姐姐嘉年华梗"),
        (142.02, 146.22, "结尾评价"),
    ],
    "P2": [
        (7.78, 9.10, "调音乐"),
        (14.57, 16.61, "铺垫"),
        (29.91, 43.26, "奶蛙脸P图冷场"),
        (54.74, 60.46, "看小宝"),
        (74.16, 82.04, "这张挺漂亮"),
        (131.23, 135.51, "比例贼怪"),
    ],
}


def _safe(name: str) -> str:
    return re.sub(r"[^\w\-]", "_", name)


def run_matrix(cfg, out_root: Path, videos: Dict[str, str], describe: bool = False) -> Dict:
    summary: Dict = {"videos": {}, "budgets": BUDGETS}
    for name, video in videos.items():
        print(f"\n===== {name}: {video} =====", flush=True)
        info = probe_media(video, cfg)
        scene_changes = extract_scene_changes(video)
        frozen = extract_frozen_segments(video)
        ranked = rank_frame_candidates(
            info.duration_s, scene_changes, frozen,
            merge_window=MERGE_WINDOW, max_candidates=MAX_CANDIDATES)

        # 全量抽一次（候选池 top N），各预算取前缀复用
        cands_max = frames_for_budget(ranked, MAX_CANDIDATES)
        frames_dir = out_root / "frames" / _safe(name)
        names = extract_frames(video, cands_max, frames_dir, info.fps,
                               find_ffmpeg(cfg), width=info.width, height=info.height)
        frame_by_t: Dict[float, Dict] = {}
        for c, n in zip(cands_max, names):
            frame_by_t[round(c.t, 3)] = {
                "t": round(c.t, 3), "reason": c.reason, "anchor": c.anchor,
                "frame": n, "path": str(frames_dir / n) if n else "",
            }

        video_entry = {
            "video": video,
            "duration": round(info.duration_s, 3),
            "fps": info.fps,
            "width": info.width,
            "height": info.height,
            "scene_changes": len(scene_changes),
            "frozen_segments": len(frozen),
            "sampling": {
                "scene_anchors": sum(1 for c in ranked if c.tier == 0),
                "frozen_anchors": sum(1 for c in ranked if c.tier == 1),
                "fill": sum(1 for c in ranked if c.tier == 2),
            },
            "budgets": {},
        }
        scene_ts = [float(s["t"]) for s in scene_changes]
        for b in BUDGETS:
            cands = frames_for_budget(ranked, b)
            ts = [c.t for c in cands]
            frames = [frame_by_t[round(t, 3)] for t in ts if round(t, 3) in frame_by_t]
            sr = scene_recall(ts, scene_ts, tol=2.0)
            mc = moment_coverage(ts, [(s, e) for s, e, _ in GT_MOMENTS.get(name, [])])
            entry = {
                "frames": len(frames),
                "scene_recall_2s": sr,
                "moment_coverage": mc,
                "avg_gap_s": round(avg_gap(ts, info.duration_s), 3),
            }
            video_entry["budgets"][b] = entry
            sb = {
                "video": video, "duration": round(info.duration_s, 3),
                "fps": info.fps, "width": info.width, "height": info.height,
                "budget": b,
                "sampling": video_entry["sampling"],
                "metrics": {
                    "scene_recall_2s": round(sr, 3) if sr is not None else None,
                    "moment_coverage": round(mc, 3) if mc is not None else None,
                    "avg_gap_s": entry["avg_gap_s"],
                },
                "frames": frames,
            }
            save_storyboard(sb, str(out_root / f"{_safe(name)}_b{b}.json"))
        summary["videos"][name] = video_entry

        if describe:
            _describe_budget(cfg, out_root, name)
    return summary


def _describe_budget(cfg, out_root: Path, name: str, budget: int = 120) -> None:
    """给指定预算的 storyboard 生成 VL 帧描述（in-place 落 _described.json）。"""
    target = out_root / f"{_safe(name)}_b{budget}.json"
    if not target.exists():
        print(f"  [{name}] 缺 {target.name}，跳过描述", flush=True)
        return
    from liveclip.storyboard import describe_frames, resolve_vl_config
    sb = json.loads(target.read_text(encoding="utf-8"))
    vlc = resolve_vl_config(cfg)
    if not vlc["api_key"]:
        print(f"  [{name}] 未配置 VL API Key，跳过描述", flush=True)
        return
    descs = describe_frames(
        sb["frames"], vlc["api_key"], vlc["model"], vlc["base_url"],
        batch_size=int((cfg.get("storyboard") or {}).get("describe_batch_size", 1)))
    miss = sum(1 for d in descs if d is None)
    for f, d in zip(sb["frames"], descs):
        f["desc"] = d
    sb["described_by"] = vlc["model"]
    sb["desc_missing"] = miss
    save_storyboard(sb, str(target.with_name(target.stem + "_described.json")))
    print(f"  [{name} b{budget}] 描述完成 缺失{miss}/{len(descs)}", flush=True)


def print_table(summary: Dict) -> None:
    print("\n===== 抽帧预算对比 =====", flush=True)
    print(f"{'视频':<9}{'预算':<6}{'帧数':<5}{'场景召回@2s':<12}{'关键点覆盖':<11}{'平均间隔s':<9}", flush=True)
    print("-" * 55, flush=True)
    for name, ve in summary["videos"].items():
        for b, e in ve["budgets"].items():
            sr = f"{e['scene_recall_2s']:.2f}" if e["scene_recall_2s"] is not None else "n/a"
            mc = f"{e['moment_coverage']:.2f}" if e["moment_coverage"] is not None else "n/a"
            print(f"{name:<9}{b:<6}{e['frames']:<5}{sr:<12}{mc:<11}{e['avg_gap_s']:.2f}", flush=True)
        print(f"  └ 时长{ve['duration']:.0f}s 场景{ve['scene_changes']}个 锚点(场景{ve['sampling']['scene_anchors']}/冻结{ve['sampling']['frozen_anchors']}/补{ve['sampling']['fill']})", flush=True)
    print(flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(prog="test_storyboard_budgets.py")
    ap.add_argument("--video", action="append", default=None, metavar="NAME=PATH",
                    help="素材视频（可多次传入，如 --video P1=D:\\素材\\P1.mp4）")
    ap.add_argument("--describe", action="store_true", help="对每个视频 b120 生成 VL 帧描述")
    args = ap.parse_args()
    if not args.video:
        ap.error("至少需要一个 --video NAME=PATH（勿硬编码本机路径）")
    videos: Dict[str, str] = {}
    for item in args.video:
        name, _, path = item.partition("=")
        if not name or not path:
            ap.error(f"--video 需 NAME=PATH 格式: {item}")
        videos[name.strip()] = path.strip()
    cfg = load_config()
    out_root = output_dir(cfg) / "work" / "storyboard"
    out_root.mkdir(parents=True, exist_ok=True)
    summary = run_matrix(cfg, out_root, videos, describe=args.describe)
    save_storyboard(summary, str(out_root / "summary.json"))
    print_table(summary)
    print(f"产物目录: {out_root}", flush=True)


if __name__ == "__main__":
    main()