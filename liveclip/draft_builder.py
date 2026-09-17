"""draft_builder.py — 剪映草稿生成的唯一核心（增强版）。

所有入口共用这一套逻辑：
- scripts/build_draft.py（CLI）→ 调 build_draft_from_edl
- liveclip/make_draft.py（旧库兼容层）→ 委托本模块
- scripts/run_pipeline.py（主入口 draft 子命令）→ 经 make_draft 兼容层

特性（与用户确认参数对齐）：
- 画布按比例创建：9:16 / 4:3 / 16:9（默认 4:3），字号按比例映射 15/8/5
- 字幕：后现代体 + 白字 + 红描边 + 红阴影（liveclip.styles 常量）
- 发光：保存后注入"轮廓光" bloom（styles.SUBTITLE_GLOW）
- 多素材：段带 video 字段即可混排（同一素材自动去重转码）
- 兼容旧 EDL 对象（liveclip.edl.EDL）：宽高自动推断比例
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import pyJianYingDraft as draft
from pyJianYingDraft import trange
from pyJianYingDraft import TextBorder, TextSegment, TextStyle, TextShadow

from liveclip.styles import (
    SUBTITLE_FONT_SIZE_BY_RATIO,
    SUBTITLE_BORDER_COLOR,
    SUBTITLE_BORDER_WIDTH,
    SUBTITLE_SHADOW_COLOR,
    SUBTITLE_SHADOW_ALPHA,
    SUBTITLE_SHADOW_DIFFUSE,
    SUBTITLE_SHADOW_DISTANCE,
    SUBTITLE_SHADOW_ANGLE,
    SUBTITLE_GLOW,
    DEFAULT_RATIO,
)

# 画布尺寸（按比例）
CANVAS_SIZE = {
    "9:16": (1080, 1920),
    "4:3": (1920, 1440),
    "16:9": (1920, 1080),
}


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def find_ffmpeg() -> str:
    """优先系统 PATH，其次剪映自带。"""
    p = shutil.which("ffmpeg")
    if p:
        return p
    cands = [
        Path(r"C:\MyApp\剪映JianyingPro 免V1P"),  # 用户实际使用的免V1P 版
        Path(os.environ.get("LOCALAPPDATA", "")) / "JianyingPro" / "Apps",
    ]
    for c in cands:
        if c.is_file():
            return str(c)
        if c.is_dir():
            hit = next(c.rglob("ffmpeg.exe"), None)
            if hit:
                return str(hit)
    raise RuntimeError("未找到 ffmpeg，请安装或提供路径")


def to_mp4(video: Path, ffmpeg: str) -> Path:
    """TS 等格式无损转 MP4；已是 mp4 则原样返回。"""
    if video.suffix.lower() == ".mp4":
        return video
    out = video.with_name(video.stem + "_clip.mp4")
    if out.exists():
        return out
    print(f"[ffmpeg] {video.name} -> {out.name} (copy)")
    r = subprocess.run([ffmpeg, "-y", "-i", str(video), "-c", "copy",
                        "-movflags", "+faststart", str(out)],
                       capture_output=True, text=True)
    if r.returncode != 0 or not out.exists():
        # copy 失败则重编码（h264+aac）
        print("[ffmpeg] copy 失败，重编码中…")
        r = subprocess.run([ffmpeg, "-y", "-i", str(video),
                            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                            "-c:a", "aac", "-movflags", "+faststart", str(out)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg 转码失败: {r.stderr[-500:]}")
    return out


def ratio_from_size(w: int, h: int) -> str:
    """按画布/视频宽高推断比例（兼容旧 EDL 对象）。"""
    if w <= 0 or h <= 0:
        return DEFAULT_RATIO
    if abs(w / h - 9 / 16) < 0.05:
        return "9:16"
    if abs(w / h - 16 / 9) < 0.05:
        return "16:9"
    if abs(w / h - 4 / 3) < 0.05:
        return "4:3"
    # 其他比例：按方向就近归入
    return "9:16" if h > w else "16:9"


def edl_to_segments(edl) -> List[Dict[str, Any]]:
    """兼容：liveclip.edl.EDL 对象 → 段列表 [{video, source_start, duration, target_start}]。"""
    segs = []
    for s in edl.segments:
        if s.kind != "keep":
            continue
        segs.append({
            "video": edl.source_path,
            "source_start": s.source_start,
            "duration": s.source_end - s.source_start,
            "target_start": s.target_start,
        })
    return segs


# ---------------------------------------------------------------------------
# 样式构造
# ---------------------------------------------------------------------------
def make_style_reference(font_size: float) -> TextSegment:
    """生成一个仅供 import_srt 作样式参考的字幕段（文本/时长会被忽略）。"""
    return TextSegment(
        "样式参考", trange("0.5s", "1s"),
        font=draft.FontType.后现代体,          # font_id 6740435494053614093，与草稿一致
        style=TextStyle(size=font_size, color=(1.0, 1.0, 1.0), align=1),  # 白色、居中
        border=TextBorder(alpha=1.0, color=SUBTITLE_BORDER_COLOR, width=SUBTITLE_BORDER_WIDTH),
        shadow=TextShadow(alpha=SUBTITLE_SHADOW_ALPHA, color=SUBTITLE_SHADOW_COLOR,
                          diffuse=SUBTITLE_SHADOW_DIFFUSE, distance=SUBTITLE_SHADOW_DISTANCE,
                          angle=SUBTITLE_SHADOW_ANGLE),
    )


TEXT_GLOW = dict(SUBTITLE_GLOW)


def inject_text_glow(draft_dir: str, name: str) -> int:
    """给刚生成的草稿注入字幕发光（轮廓光）。

    pyJianYingDraft 不原生支持发光，这里在 save() 后直接修改 draft_content.json：
    - 为每条字幕创建独立 bloom 素材（参数与用户剪映里调的一致）
    - 字幕轨道段 extra_material_refs 挂上素材 id，素材 check_flag 置 +64（发光位）
    返回注入的字幕条数。
    """
    p = Path(draft_dir) / name / "draft_content.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    materials = d["materials"]
    texts = {t["id"]: t for t in materials.get("texts", [])}
    new_fx = []
    count = 0
    for tr in d.get("tracks", []):
        if tr.get("type") != "text":
            continue
        for seg in tr.get("segments", []):
            mid = seg.get("material_id")
            if mid not in texts:
                continue
            fx = dict(TEXT_GLOW)
            fx["id"] = uuid.uuid4().hex.upper()
            fx["adjust_params"] = []
            fx["algorithm_artifact_path"] = ""
            fx["apply_target_type"] = 0
            fx["category_id"] = ""
            fx["category_name"] = ""
            fx["color_match_info"] = None
            fx["enable_skin_tone_correction"] = False
            fx["exclusion_group"] = []
            fx["face_adjust_params"] = []
            fx["formula_id"] = ""
            fx["intensity_key"] = ""
            fx["multi_language_current"] = ""
            fx["platform"] = "all"
            fx["request_id"] = ""
            fx["source_platform"] = 0
            fx["sub_type"] = "none"
            fx["time_range"] = None
            fx["version"] = ""
            new_fx.append(fx)
            # 清掉 pyJianYingDraft 残留的 speed 幽灵引用，只保留发光 id（与剪映手调草稿一致）
            seg["extra_material_refs"] = [fx["id"]]
            texts[mid]["check_flag"] = (texts[mid].get("check_flag", 0) | 64)
            count += 1
    if new_fx:
        materials.setdefault("effects", []).extend(new_fx)
        p.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    return count


# ---------------------------------------------------------------------------
# 核心：EDL 段列表 → 剪映草稿
# ---------------------------------------------------------------------------
def build_draft_from_edl(
    segments: List[Dict[str, Any]],
    srt_path: Optional[str] = None,
    ratio: Optional[str] = None,
    name: str = "live-clip",
    draft_dir: Optional[str] = None,
    fps: int = 60,
) -> Dict[str, Any]:
    """按段列表生成剪映草稿，返回结果 dict。

    Args:
        segments: [{video, source_start, duration, target_start}, ...]
        srt_path: 目标时间轴字幕 SRT（None 则不加字幕）
        ratio: 画布比例 9:16/4:3/16:9（None 取 DEFAULT_RATIO）
        name: 草稿名
        draft_dir: 剪映草稿目录（默认 LOCALAPPDATA\JianyingPro\...\com.lveditor.draft）
        fps: 草稿帧率
    """
    if not segments:
        raise ValueError("没有保留段（keep）")
    if ratio is None:
        ratio = DEFAULT_RATIO
    canvas_w, canvas_h = CANVAS_SIZE[ratio]
    font_size = SUBTITLE_FONT_SIZE_BY_RATIO[ratio]

    if draft_dir is None:
        draft_dir = str(Path(os.environ["LOCALAPPDATA"]) / "JianyingPro" / "User Data" / "Projects" / "com.lveditor.draft")

    # 素材转码（去重）
    ffmpeg = find_ffmpeg()
    videos: Dict[str, Path] = {}
    for seg in segments:
        vp = seg.get("video")
        if not vp:
            raise ValueError("段缺少 video 字段")
        if vp not in videos:
            videos[vp] = to_mp4(Path(vp), ffmpeg)

    folder = draft.DraftFolder(draft_dir)
    script = folder.create_draft(name, canvas_w, canvas_h, fps=fps, allow_replace=True)
    script.append_tracks([
        draft.TrackSpec(draft.TrackType.video, "main_video"),
        draft.TrackSpec(draft.TrackType.text, "captions"),
    ])

    for seg in segments:
        vs = draft.VideoSegment(
            str(videos[seg["video"]]),
            trange(f"{seg['target_start']:.3f}s", f"{seg['duration']:.3f}s"),
            source_timerange=trange(f"{seg['source_start']:.3f}s", f"{seg['duration']:.3f}s"),
        )
        script.add_segment(vs, "main_video")

    if srt_path and Path(srt_path).exists():
        script.import_srt(str(srt_path), "captions", style_reference=make_style_reference(font_size))
    script.save()

    glow_count = inject_text_glow(draft_dir, name)

    target_duration = max((s["target_start"] + s["duration"] for s in segments), default=0.0)
    return {
        "ok": True,
        "draft_name": name,
        "draft_dir": draft_dir,
        "keep_segments": len(segments),
        "target_duration": round(target_duration, 3),
        "ratio": ratio,
        "canvas": f"{canvas_w}x{canvas_h}",
        "font_size": font_size,
        "glow_segments": glow_count,
    }
