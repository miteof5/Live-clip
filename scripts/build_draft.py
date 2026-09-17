"""build_draft.py — 由 EDL + 映射字幕生成剪映草稿（CLI 入口）。

核心逻辑在 liveclip/draft_builder.build_draft_from_edl（所有入口共用）。
CLI 只负责解析参数、读取 EDL、调用核心。

流程：
1. 原视频若是 .ts 等非标准格式，先用 ffmpeg 无损转 .mp4（copy 编码）
2. 用 pyJianYingDraft 创建草稿：
   - 视频轨道：每个保留区间一个 VideoSegment（source_timerange 取素材对应段）
   - 字幕轨道：import_srt 导入映射后字幕（传承样式 + 发光注入）
3. save() 到剪映草稿目录，用户可在剪映中打开

画布比例与字幕字号：9:16 -> 15 / 4:3 -> 8 / 16:9 -> 5（--ratio 指定，默认 4:3）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# 项目根入 path，保证 scripts/ 下可 import liveclip.*
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liveclip.draft_builder import build_draft_from_edl, find_ffmpeg, to_mp4  # noqa: E402
from liveclip.styles import DEFAULT_RATIO  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(prog="build_draft.py", description="EDL + SRT → 剪映草稿")
    ap.add_argument("--edl", required=True, help="edl.json 路径（keep 段可带 video 字段做多素材混排）")
    ap.add_argument("--srt", default=None, help="mapped.srt 路径（目标时间轴字幕）")
    ap.add_argument("--video", default=None, help="原视频路径（EDL 段带 video 字段时可省略）")
    ap.add_argument("--draft-dir", default=None, help="剪映草稿目录（默认 LOCALAPPDATA）")
    ap.add_argument("--name", default="doubao-clip-test", help="草稿名")
    ap.add_argument("--fps", type=int, default=60, help="草稿帧率")
    ap.add_argument("--ratio", choices=["9:16", "4:3", "16:9"], default=None,
                    help="输出画布比例（默认 4:3；字幕字号按比例映射 15/8/5）")
    ap.add_argument("--orientation", choices=["landscape", "portrait"], default=None,
                    help="兼容旧调用：landscape->16:9，portrait->9:16（优先用 --ratio）")
    args = ap.parse_args()

    draft_dir = args.draft_dir or str(Path(os.environ["LOCALAPPDATA"]) / "JianyingPro" / "User Data" / "Projects" / "com.lveditor.draft")

    # 输出画布比例：--ratio 优先，其次 --orientation 兼容映射，最后默认 4:3
    ratio = args.ratio
    if ratio is None and args.orientation is not None:
        ratio = {"landscape": "16:9", "portrait": "9:16"}[args.orientation]
    if ratio is None:
        ratio = DEFAULT_RATIO

    edl = json.loads(Path(args.edl).read_text(encoding="utf-8"))

    # 段列表：EDL 段带 video 字段则用各自素材；否则统一用 --video
    segments = []
    for seg in edl["keep"]:
        vp = seg.get("video") or args.video
        if not vp:
            raise SystemExit("EDL 段缺少 video 字段且未提供 --video")
        segments.append({
            "video": vp,
            "source_start": seg["source_start"],
            "duration": seg["duration"],
            "target_start": seg["target_start"],
        })

    result = build_draft_from_edl(
        segments=segments,
        srt_path=args.srt,
        ratio=ratio,
        name=args.name,
        draft_dir=draft_dir,
        fps=args.fps,
    )
    # 附带原始 EDL 目标时长（兼容旧输出字段）
    result["target_duration"] = result.get("target_duration", edl.get("target_duration", 0.0))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
