"""EDL（编辑决策表）数据模型。

豆包（LLM 判断层）输出"源时间轴上的保留/删除区间 + 金句标记 + 理由"，
本模块负责：
- 承载与校验该决策（时间不重叠、不越界、顺序递增）
- 计算目标时间轴（keep 段顺序拼接后的 target_start/target_end）
- JSON 序列化（用户可手改、可复查）

时间单位统一为秒（float）。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional


@dataclass
class EDLSegment:
    kind: Literal["keep", "cut"]
    source_start: float
    source_end: float
    reason: str = ""                 # 判断理由（供用户复查）
    is_golden: bool = False          # 金句标记（M1 未启用花字，仅留元数据）
    text: str = ""                   # 该区间对应转写文本（供复查）
    # 说话人维度（由 enrich_edl 填充；判断层也可直接写）
    speaker_id: Optional[int] = None     # 该段主要说话人（音频 diarization）
    dialogue_zoom: bool = False          # 连麦对话段：谁说话放大谁（镜头切换预标记）
    # 以下由 compute_target_timeline 填充
    target_start: float = 0.0
    target_end: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EDL:
    source_path: str
    width: int
    height: int
    source_duration_s: float
    segments: List[EDLSegment] = field(default_factory=list)
    title: str = "live-clip"
    fps: float = 30.0
    target_duration_s: float = 0.0
    speaker_switches: List[Dict[str, Any]] = field(default_factory=list)

    # ---------- 构建与校验 ----------
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
        """校验 EDL，返回问题列表；空列表表示合法。"""
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

    # ---------- 时间轴计算 ----------
    def compute_target_timeline(self) -> "EDL":
        """按 keep 顺序拼接，填充每段 target_start/target_end。"""
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

    # ---------- 源时间轴 → 目标时间轴映射 ----------
    def map_source_to_target(self, source_t: float) -> Optional[float]:
        """把源时间轴上的时刻映射到目标时间轴；落在 cut 段则返回 None。"""
        for s in self.segments:
            if s.kind != "keep":
                continue
            if s.source_start <= source_t <= s.source_end:
                return s.target_start + (source_t - s.source_start)
        return None

    def map_phrase_to_target(self, phrase_start: float, phrase_end: float) -> Optional[tuple]:
        """把源时间轴上的短语区间映射到目标时间轴。

        短语可能跨 keep/cut 边界：只保留落在 keep 内的部分。
        返回 (target_start, target_end)；完全落在 cut 内返回 None。
        """
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

    # ---------- 序列化 ----------
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
