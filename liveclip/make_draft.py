"""make_draft.py — 草稿生成兼容层（统一 v2 EDL 格式）。

旧库公开接口 `DraftOptions / build_target_srt / make_draft` 保持不变，
内部全部委托 liveclip.draft_builder + NormalizedEDL：

- `make_draft` 接受 **v2 格式**（{"keep": [...]} 的 dict / JSON 路径 / NormalizedEDL），
  自动归一化 → 目标时间轴 SRT → 委托 draft_builder 渲染草稿。
- 旧版 v1 EDL 对象（liveclip.edl.EDL）仍兼容，但不再对外产出。

注意行为（对齐用户确认的参数）：
- 画布比例：由 EDL 宽高推断（1920x1080->16:9，1080x1920->9:16，1920x1440->4:3）
- 字幕：后现代体 + 白字 + 红描边 + 红阴影 + 发光（传承样式）
- BGM 占位不再自动加（用户：BGM 不用管，最后自己加）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .draft_builder import build_draft_from_edl, ratio_from_size
from .edl import EDL, NormalizedEDL, KeepSegment
from .merge_words import Phrase, _fmt_ts


@dataclass
class DraftOptions:
    with_subtitle: bool = True
    srt_path: Optional[str] = None       # 直接用现成 SRT（目标时间轴）
    phrases: Optional[List[Phrase]] = None  # 或传源时间轴短语，自动映射
    bgm_path: Optional[str] = None       # 兼容字段：忽略（BGM 由用户在剪映添加）
    bgm_volume: float = 0.3
    draft_name: str = "live-clip"
    allow_replace: bool = True
    text_style: Optional[Dict[str, Any]] = None  # 预留：字幕样式覆盖


def build_target_srt(edl: NormalizedEDL, phrases: List[Phrase]) -> str:
    """源时间轴短语 → 目标时间轴 SRT（仅保留落在 keep 段的短语）。"""
    lines: List[str] = []
    idx = 1
    for p in phrases:
        mapped = edl.map_phrase_to_target(p.start, p.end)
        if mapped is None:
            continue
        t_start, t_end = mapped
        if t_end - t_start < 0.3:
            continue
        lines.append(str(idx))
        lines.append(f"{_fmt_ts(t_start)} --> {_fmt_ts(t_end)}")
        lines.append(p.text)
        lines.append("")
        idx += 1
    return "\n".join(lines)


def _write_srt_temp(srt_text: str) -> str:
    import tempfile
    fd, path = tempfile.mkstemp(suffix=".srt", prefix="liveclip_")
    with open(fd, "w", encoding="utf-8") as f:
        f.write(srt_text)
    return path


def _normalize(edl_or_data: Union[NormalizedEDL, EDL, Dict[str, Any], str]) -> NormalizedEDL:
    """任意 EDL 输入 → NormalizedEDL（v2 统一模型）。"""
    if isinstance(edl_or_data, NormalizedEDL):
        return edl_or_data
    if isinstance(edl_or_data, EDL):
        return NormalizedEDL.from_dict(edl_or_data.to_dict())
    if isinstance(edl_or_data, str):
        return NormalizedEDL.load(edl_or_data)
    if isinstance(edl_or_data, dict):
        return NormalizedEDL.from_dict(edl_or_data)
    raise TypeError(f"不支持的 EDL 输入类型: {type(edl_or_data)!r}")


def make_draft(
    edl: Union[NormalizedEDL, EDL, Dict[str, Any], str],
    draft_folder_path: str,
    options: Optional[DraftOptions] = None,
) -> str:
    """根据 EDL 生成剪映草稿，返回草稿目录路径。

    Args:
        edl: v2 格式 EDL（dict / JSON 路径 / NormalizedEDL）；旧版 EDL 对象兼容
        draft_folder_path: 剪映草稿目录（草稿会创建为其子目录）
        options: 字幕/配乐选项
    """
    opts = options or DraftOptions()
    normalized = _normalize(edl)
    errors = normalized.validate()
    if errors:
        raise ValueError("EDL 校验失败:\n" + "\n".join(errors))
    if normalized.target_duration_s <= 0:
        normalized.compute_target_timeline()

    # 字幕：优先现成 SRT，其次短语自动映射
    srt_file: Optional[str] = None
    if opts.with_subtitle:
        if opts.srt_path and Path(opts.srt_path).exists():
            srt_file = opts.srt_path
        elif opts.phrases:
            srt_text = build_target_srt(normalized, opts.phrases)
            srt_file = _write_srt_temp(srt_text)

    ratio = ratio_from_size(normalized.width, normalized.height)
    result = build_draft_from_edl(
        segments=normalized.to_segments(),
        srt_path=srt_file,
        ratio=ratio,
        name=opts.draft_name,
        draft_dir=draft_folder_path,
        fps=int(round(normalized.fps or 30)),
    )
    return str(Path(draft_folder_path) / opts.draft_name)
