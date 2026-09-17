"""EDL（编辑决策表）数据模型。

**对外唯一标准格式（新版，v2）**：

    {
      "keep": [ {"source_start", "source_end", "target_start", "duration",
                 "video"?, "reason"?, "is_golden"?, "text"?, "speaker_id"?, "dialogue_zoom"?} ],
      "cut": [ {"start", "end", "reason"?} ],          # 可选（被剪区间，供复查）
      "target_duration": 12.5,                         # 可选（缺失时自动重算）
      "source_path"?, "source_duration_s"?, "width"?, "height"?, "fps"?, "title"?,
      "speaker_switches"?
    }

所有消费方（build_draft / make_draft / run_pipeline / enrich_edl）统一使用该格式。
- `duration` 与 `source_end` 冗余：两者任给其一即可，`NormalizedEDL` 自动补齐。
- `video` 字段：多素材混排时指定该段素材；缺省用 EDL 顶层 `source_path`。

**旧版兼容**：历史产物（`{"source_path", "segments":[{"kind":"keep|cut", ...}]}`）仍可
通过 `NormalizedEDL.from_dict / load` 自动识别并归一化，但不再对外产出。

时间单位统一为秒（float）。

本模块同时保留 `EDL / EDLSegment`（旧 dataclass）作为内部时间轴映射/校验引擎，
**不对外承诺**——新代码一律使用 `NormalizedEDL`。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional


# ---------------------------------------------------------------------------
# 新版标准段（v2 keep 段）
# ---------------------------------------------------------------------------
@dataclass
class KeepSegment:
    source_start: float
    source_end: float
    target_start: float = 0.0
    duration: float = 0.0
    video: Optional[str] = None          # 多素材混排；None = 用 EDL.source_path
    reason: str = ""
    is_golden: bool = False
    text: str = ""
    speaker_id: Optional[int] = None     # 该段主要说话人（enrich_edl 填充）
    dialogue_zoom: bool = False          # 连麦对话段：谁说话放大谁

    def __post_init__(self) -> None:
        # duration / source_end 冗余互补：任给其一即可
        if self.duration <= 0 and self.source_end > self.source_start:
            self.duration = self.source_end - self.source_start
        if self.source_end <= self.source_start and self.duration > 0:
            self.source_end = self.source_start + self.duration

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return {k: v for k, v in d.items() if v not in (None, "", False, 0.0) or k in
                ("source_start", "source_end", "target_start")}


# ---------------------------------------------------------------------------
# 新版统一模型（唯一对外入口）
# ---------------------------------------------------------------------------
@dataclass
class NormalizedEDL:
    """统一 EDL 模型：keep 段列表 + 校验 / 时间轴映射 / 序列化（v2 格式）。"""

    keep: List[KeepSegment] = field(default_factory=list)
    cut: List[Dict[str, Any]] = field(default_factory=list)
    source_path: str = ""
    source_duration_s: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 30.0
    title: str = "live-clip"
    target_duration_s: float = 0.0
    speaker_switches: List[Dict[str, Any]] = field(default_factory=list)

    # ---------- 构建（自动识别新旧格式） ----------
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "NormalizedEDL":
        source_duration = float(d.get("source_duration_s") or d.get("source_duration") or 0.0)
        source_path = str(d.get("source_path") or "")

        if isinstance(d.get("keep"), list):
            # 新版 v2：{"keep": [...]}
            # 列表顺序 = 目标时间轴顺序（多素材混排时 target_start 递增）。
            # 仅单素材（无 video 或全部同素材）时按 source_start 排序防御乱序输入。
            keeps = [_keep_segment(k, source_path) for k in d["keep"] if isinstance(k, dict)]
            if len({k.video for k in keeps}) <= 1:
                keeps.sort(key=lambda k: (k.source_start, k.source_end))
            cut = list(d.get("cut") or [])
            target_duration = float(d.get("target_duration") or 0.0)
        else:
            # 旧版 v1：{"segments": [{"kind": "keep|cut", ...}]}
            keeps, cut = [], []
            for s in d.get("segments", []):
                if not isinstance(s, dict):
                    continue
                if s.get("kind") == "keep":
                    ks = KeepSegment(
                        source_start=float(s.get("source_start", 0.0)),
                        source_end=float(s.get("source_end", 0.0)),
                        target_start=float(s.get("target_start", 0.0)),
                        reason=str(s.get("reason") or ""),
                        is_golden=bool(s.get("is_golden", False)),
                        text=str(s.get("text") or ""),
                        speaker_id=s.get("speaker_id"),
                        dialogue_zoom=bool(s.get("dialogue_zoom", False)),
                    )
                    keeps.append(ks)
                elif s.get("kind") == "cut":
                    cut.append({"start": float(s.get("source_start", 0.0)),
                                "end": float(s.get("source_end", 0.0)),
                                "reason": str(s.get("reason") or "")})
            target_duration = float(d.get("target_duration") or d.get("target_duration_s") or 0.0)

        return cls(
            keep=keeps,
            cut=cut,
            source_path=source_path,
            source_duration_s=source_duration,
            width=int(d.get("width", 0) or 0),
            height=int(d.get("height", 0) or 0),
            fps=float(d.get("fps", 30.0) or 30.0),
            title=str(d.get("title") or "live-clip"),
            target_duration_s=target_duration,
            speaker_switches=list(d.get("speaker_switches") or []),
        )

    @classmethod
    def load(cls, path: str) -> "NormalizedEDL":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    # ---------- 校验 ----------
    def validate(self) -> List[str]:
        """校验 keep 段：按 video 分组检查源时间轴单调 / 不重叠 / 不越界。

        - 多素材混排：每组（同一 video）内 source_start 必须单调且不重叠，
          跨素材段之间没有源时间轴可比性，不检查。
        - 列表顺序 = 目标时间轴顺序（多素材时以 target_start 为准）。
        """
        errors: List[str] = []
        # 按 video 分组：空 video 视为同素材一组（单素材场景）
        groups: Dict[Any, List[KeepSegment]] = {}
        for s in self.keep:
            groups.setdefault(s.video, []).append(s)
        for video, segs in groups.items():
            ordered = sorted(segs, key=lambda k: (k.source_start, k.source_end))
            prev_end = 0.0
            for i, s in enumerate(ordered):
                if s.source_start < 0 or s.source_end <= s.source_start:
                    errors.append(f"段({video} #{i}): 时间非法 {s.source_start}~{s.source_end}")
                if self.source_duration_s and s.source_end > self.source_duration_s:
                    errors.append(f"段({video} #{i}): 终点越界 {s.source_end} > {self.source_duration_s}")
                if s.source_start < prev_end - 1e-6:
                    errors.append(f"段({video} #{i}): 与上一段重叠（{s.source_start} < {prev_end}）")
                prev_end = max(prev_end, s.source_end)
        if not self.keep:
            errors.append("没有保留段（keep）")
        return errors

    # ---------- 时间轴计算 ----------
    def compute_target_timeline(self) -> "NormalizedEDL":
        """按 keep 列表顺序重算 target_start/target_end。

        **列表顺序 = 目标时间轴顺序**（多素材混排按输入顺序拼接，不排序）；
        单素材（无 video 或全部同素材）时先按 source_start 排序防御乱序输入。
        """
        if len({k.video for k in self.keep}) <= 1:
            self.keep.sort(key=lambda k: (k.source_start, k.source_end))
        cursor = 0.0
        for s in self.keep:
            s.target_start = cursor
            s.duration = s.source_end - s.source_start
            cursor += s.duration
        self.target_duration_s = cursor
        return self

    # ---------- 源时间轴 → 目标时间轴映射 ----------
    def map_source_to_target(self, source_t: float) -> Optional[float]:
        """源时间轴时刻 → 目标时间轴；不在任何 keep 内返回 None。"""
        for s in self.keep:
            if s.source_start <= source_t <= s.source_end:
                return s.target_start + (source_t - s.source_start)
        return None

    def map_phrase_to_target(self, phrase_start: float, phrase_end: float) -> Optional[tuple]:
        """源时间轴短语区间 → 目标时间轴（只保留落在 keep 内的部分）。

        返回 (target_start, target_end)；完全落在 cut 内返回 None。
        """
        parts: List[tuple] = []
        for s in self.keep:
            ov_start = max(phrase_start, s.source_start)
            ov_end = min(phrase_end, s.source_end)
            if ov_end > ov_start:
                parts.append((
                    s.target_start + (ov_start - s.source_start),
                    s.target_start + (ov_end - s.source_start),
                ))
        if not parts:
            return None
        return parts[0][0], parts[-1][1]

    # ---------- 序列化（只输出 v2 格式） ----------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "keep": [s.to_dict() for s in self.keep],
            "cut": self.cut,
            "target_duration": round(self.target_duration_s, 3),
            "source_path": self.source_path,
            "source_duration_s": self.source_duration_s,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "title": self.title,
            "speaker_switches": self.speaker_switches,
        }

    def to_segments(self) -> List[Dict[str, Any]]:
        """→ draft_builder 段列表 [{video, source_start, duration, target_start}]。"""
        segs = []
        for s in self.keep:
            segs.append({
                "video": s.video or self.source_path,
                "source_start": s.source_start,
                "duration": s.duration,
                "target_start": s.target_start,
            })
        return segs

    def save(self, out_path: str) -> str:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return str(p)


def _keep_segment(k: Dict[str, Any], default_video: str) -> KeepSegment:
    """构造 KeepSegment：兼容 source_end/duration 任给其一。"""
    source_start = float(k.get("source_start", 0.0))
    source_end = float(k.get("source_end") or 0.0)
    duration = float(k.get("duration") or 0.0)
    if source_end <= source_start and duration > 0:
        source_end = source_start + duration
    elif duration <= 0 and source_end > source_start:
        duration = source_end - source_start
    return KeepSegment(
        source_start=source_start,
        source_end=source_end,
        target_start=float(k.get("target_start", 0.0) or 0.0),
        duration=duration,
        video=k.get("video"),
        reason=str(k.get("reason") or ""),
        is_golden=bool(k.get("is_golden", False)),
        text=str(k.get("text") or ""),
        speaker_id=k.get("speaker_id"),
        dialogue_zoom=bool(k.get("dialogue_zoom", False)),
    )


# ---------------------------------------------------------------------------
# 旧版兼容模型（deprecated，仅供内部映射与旧产物回读，不对外承诺）
# ---------------------------------------------------------------------------
@dataclass
class EDLSegment:
    kind: Literal["keep", "cut"]
    source_start: float
    source_end: float
    reason: str = ""
    is_golden: bool = False
    text: str = ""
    speaker_id: Optional[int] = None
    dialogue_zoom: bool = False
    target_start: float = 0.0
    target_end: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EDL:
    """旧版 EDL（v1，{"segments": [...]}）。已由 NormalizedEDL 取代，保留兼容。"""

    source_path: str
    width: int
    height: int
    source_duration_s: float
    segments: List[EDLSegment] = field(default_factory=list)
    title: str = "live-clip"
    fps: float = 30.0
    target_duration_s: float = 0.0
    speaker_switches: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EDL":
        segs = [EDLSegment(**{k: v for k, v in s.items() if k in EDLSegment.__dataclass_fields__})
                for s in d.get("segments", [])]
        return cls(
            source_path=d["source_path"],
            width=int(d.get("width", 0)),
            height=int(d.get("height", 0)),
            source_duration_s=float(d.get("source_duration_s", 0)),
            segments=segs,
            title=d.get("title", "live-clip"),
            fps=float(d.get("fps", 30.0)),
        )

    def validate(self) -> List[str]:
        errors: List[str] = []
        prev_end = 0.0
        has_keep = False
        for i, s in enumerate(self.segments):
            if s.source_start < 0 or s.source_end <= s.source_start:
                errors.append(f"段{i}: 时间非法 {s.source_start}~{s.source_end}")
            if s.source_start > self.source_duration_s:
                errors.append(f"段{i}: 起点越界 {s.source_start} > {self.source_duration_s}")
            if s.source_end > self.source_duration_s:
                errors.append(f"段{i}: 终点越界 {s.source_end} > {self.source_duration_s}")
            if s.source_start < prev_end - 1e-6:
                errors.append(f"段{i}: 与上一段重叠（{s.source_start} < {prev_end}）")
            if s.kind == "keep":
                has_keep = True
            prev_end = max(prev_end, s.source_end)
        if not has_keep:
            errors.append("没有保留段（keep）")
        return errors

    def compute_target_timeline(self) -> "EDL":
        cursor = 0.0
        for s in self.segments:
            if s.kind == "keep":
                s.target_start = cursor
                s.target_end = cursor + (s.source_end - s.source_start)
                cursor = s.target_end
            else:
                s.target_start = s.target_end = 0.0
        self.target_duration_s = cursor
        return self

    def map_source_to_target(self, source_t: float) -> Optional[float]:
        for s in self.segments:
            if s.kind != "keep":
                continue
            if s.source_start <= source_t <= s.source_end:
                return s.target_start + (source_t - s.source_start)
        return None

    def map_phrase_to_target(self, phrase_start: float, phrase_end: float) -> Optional[tuple]:
        parts: List[tuple] = []
        for s in self.segments:
            if s.kind != "keep":
                continue
            ov_start = max(phrase_start, s.source_start)
            ov_end = min(phrase_end, s.source_end)
            if ov_end > ov_start:
                parts.append((
                    s.target_start + (ov_start - s.source_start),
                    s.target_start + (ov_end - s.source_start),
                ))
        if not parts:
            return None
        return parts[0][0], parts[-1][1]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_path": self.source_path,
            "title": self.title,
            "width": self.width,
            "height": self.height,
            "source_duration_s": self.source_duration_s,
            "fps": self.fps,
            "target_duration_s": self.target_duration_s,
            "speaker_switches": self.speaker_switches,
            "segments": [s.to_dict() for s in self.segments],
        }

    def save(self, out_path: str) -> str:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return str(p)

    @classmethod
    def load(cls, path: str) -> "EDL":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
