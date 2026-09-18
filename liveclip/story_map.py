"""story_map.py — 叙事简报合成器（Story Map 方案 Phase 1，零成本本地规则）。

把 5 路信号（词级转写/图文分镜幕/音频事件/说话人/静音）时间对齐成
"幕级叙事简报" story_map_raw.json，供判断层先通读全局、再在地图上选段：

- 幕骨架优先取 visual_timeline（布局变化=段边界），缺失时按说话人切换/长停顿切幕
- 每幕挂载：转写摘要（≤200 字）、活跃说话人、音频事件统计、静音、幕级 emotion 合成分
- 全局统计：低能量区（BGM 候选）、长静音、高情绪区、说话人占比

用法（CLI）：
    python liveclip/story_map.py --phrases outputs/work/phrases.json
        [--visual-timeline .../visual_timeline.json] [--audio-events .../audio_events.json]
        [--speakers .../speaker_timeline.json] [--silences .../silences.json] [--out ...]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

MAX_TEXT_CHARS = 200       # 每幕转写摘要上限
EMO_BURST_W = 0.4          # emotion = 0.4*burst密度 + 0.3*语速陡增 + 0.3*直听情绪
EMO_RATE_W = 0.3
EMO_LISTEN_W = 0.3
HIGH_EMOTION_TH = 0.3      # emotion ≥ 此值且 bursts ≥ 2 → 高情绪区（密度归一偏向短幕，需绝对事件数兜底）
HIGH_EMOTION_MIN_BURSTS = 2
LONG_SILENCE_S = 5.0       # 剪法规则 2：>5s 沉默
BGM_MIN_ACT_S = 5.0        # 低能量区单幕最短时长
BGM_DENSITY_RATIO = 0.6    # burst 密度 < 全局中位数×此值 → 低能量候选（P2 幻听区 0.05 vs 中位 0.09 需 0.6 才够）
SCENE_GAP_S = 2.5          # 无视觉时，说话人切换/停顿 ≥ 此值切幕

# 叙事幕整形（visual 幕粒度不可直接当叙事骨架）
SPLIT_MIN_ACT_S = 8.0      # 幕时长 ≥ 此值才做内部切分
SPLIT_PEAK_MARGIN_S = 2.0  # pre_silence_peak 距幕边界 ≥ 此值才切（反应后冷场=强边界）
SPLIT_AT_SILENCE_S = 2.5   # 幕内静音 ≥ 此值 → 在静音起点切
ACT_MAX_S = 60.0           # 单幕时长上限，超了在最长静音处强制切
MERGE_MIN_GAP_S = 2.0      # 相邻幕间隔 < 此值且布局兼容 → 合并
MERGE_MAX_SHORT_S = 4.0    # 短幕（≤此值）合并不设布局限制（VL 布局抖动=识别噪声）


def _overlap(s1: float, e1: float, s2: float, e2: float) -> float:
    return max(0.0, min(e1, e2) - max(s1, s2))


def _load_json(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"输入文件不存在: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _fmt(x: float) -> float:
    return round(x, 3)


# ---------------------------------------------------------------------------
# 输入解析
# ---------------------------------------------------------------------------

def load_phrases(path: str) -> List[Dict[str, Any]]:
    """words.json（{words:[...]}）或 phrases.json（[{text,start,end}]）都接受。"""
    raw = _load_json(path)
    if raw is None:
        raise ValueError(f"无法读取短语/词级转写: {path}")
    if isinstance(raw, list):
        out = [p for p in raw if isinstance(p, dict) and "text" in p]
        if not out:
            raise ValueError(f"phrases.json 缺少带 text 的条目: {path}")
        return out
    words = raw.get("words") or []
    # 词级 → 短语：按停顿边界合并（与 merge_words 同思路）
    from .merge_words import Phrase, merge_to_phrases
    from .transcribe import Word
    # 显式构造（勿用 Word(**w)）：词条可能带额外字段（如 ASR 后端的 conf），
    # **w 展开会对未知字段抛 TypeError，显式取字段可安全忽略。
    wl = [
        Word(
            word=str(w["word"]),
            start=float(w.get("start", 0.0)),
            end=float(w.get("end", 0.0)),
            probability=float(w.get("probability", 1.0)),
        )
        for w in words if isinstance(w, dict) and "word" in w
    ]
    if not wl:
        raise ValueError(f"words.json 缺少词条目: {path}")
    return [p.to_dict() for p in merge_to_phrases(wl)]


def load_acts_from_visual(tl: Dict[str, Any]) -> List[Dict[str, Any]]:
    """visual_timeline.json → 幕骨架（layout 变化即段边界，直接采用）。"""
    segs = tl.get("segments") or []
    acts: List[Dict[str, Any]] = []
    for i, s in enumerate(segs):
        acts.append({
            "id": i,
            "start": float(s.get("start", 0.0)),
            "end": float(s.get("end", 0.0)),
            "layout": s.get("layout", ""),
            "visual": s.get("content", [])[:3],
            "ui_text": [it.get("text") for it in (s.get("ui_text") or [])][:6],
            "danmaku": [{"t": it.get("t"), "text": it.get("text")}
                        for it in (s.get("danmaku") or [])][:8],
            "gifts": [{"t": it.get("t"), "text": it.get("text")}
                      for it in (s.get("gifts") or [])][:4],
            "frames": int(s.get("frames", 0)),
        })
    return acts


def _absorb_point_acts(acts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """0 时长"点幕"（VL 布局切换帧 start==end）吸收进后一幕，避免 0 时长碎幕。

    点幕并入后一幕：起点取最早，layout 冲突置 mixed，视觉信息补缺。
    末尾点幕并入最后一幕（扩展 end）。
    """
    out: List[Dict[str, Any]] = []
    pending: List[Dict[str, Any]] = []
    for a in acts:
        if a["end"] - a["start"] < 1e-6:
            pending.append(a)
            continue
        if pending:
            a["start"] = min(a["start"], pending[0]["start"])
            lays = {p.get("layout") for p in pending if p.get("layout")}
            lays.add(a.get("layout") or "")
            lays.discard("")
            if len(lays) > 1:
                a["layout"] = "mixed"
            if not a["visual"]:
                vis = [v for p in pending for v in (p.get("visual") or [])]
                a["visual"] = vis[:3]
            pending = []
        out.append(a)
    if pending and out:
        t = pending[-1]["start"]
        if t - out[-1]["end"] <= MERGE_MIN_GAP_S:  # 仅贴边并入，避免虚假覆盖大间隙
            out[-1]["end"] = max(out[-1]["end"], t)
    return out


def load_acts_rule_based(
    phrases: List[Dict[str, Any]],
    silences: Optional[List[Dict[str, Any]]],
    speakers: Optional[List[Dict[str, Any]]],
    duration: float,
) -> List[Dict[str, Any]]:
    """无视觉产物时回退：以说话人切换点 + 长停顿 + 静音边界切幕。"""
    bounds: List[float] = [0.0]
    for p in phrases:
        bounds.extend([float(p.get("start", 0)), float(p.get("end", 0))])
    for s in silences or []:
        st, en = float(s.get("start", 0)), float(s.get("end", 0))
        if en - st >= SCENE_GAP_S:
            bounds.extend([st, en])
    # 说话人切换点（前一段 speaker 与后一段不同，且间隔 ≥ SCENE_GAP_S）
    if speakers:
        ss = sorted(speakers, key=lambda x: float(x.get("start", 0)))
        for a, b in zip(ss, ss[1:]):
            a_spk, b_spk = a.get("speaker_id"), b.get("speaker_id")
            if a_spk != b_spk and (float(b.get("start", 0)) - float(a.get("end", 0))) >= SCENE_GAP_S:
                bounds.append(float(b.get("start", 0)))
    bounds = sorted(set(round(b, 3) for b in bounds if 0 < b < duration))
    acts: List[Dict[str, Any]] = []
    prev = 0.0
    for i, b in enumerate(bounds):
        acts.append({"id": i, "start": _fmt(prev), "end": _fmt(b), "layout": "",
                     "visual": [], "ui_text": [], "danmaku": [], "gifts": [], "frames": 0})
        prev = b
    acts.append({"id": len(acts), "start": _fmt(prev), "end": _fmt(duration),
                 "layout": "", "visual": [], "ui_text": [], "danmaku": [], "gifts": [], "frames": 0})
    return acts


# ---------------------------------------------------------------------------
# 叙事幕整形：分割（反应后冷场/长静音）+ 合并（碎幕归并）
# ---------------------------------------------------------------------------

def _split_acts(
    acts: List[Dict[str, Any]],
    events: Optional[List[Dict[str, Any]]],
    silences: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """长幕内按强叙事边界切分：
    1. pre_silence_peak（反应→冷场）处切
    2. 幕内 ≥ SPLIT_AT_SILENCE_S 的静音起点切
    3. 超过 ACT_MAX_S 在最长静音处强制切
    视觉布局不动；切分只产出更细的区间，后续挂载按区间重算。
    """
    out: List[Dict[str, Any]] = []
    peaks = [e for e in (events or []) if e.get("type") == "pre_silence_peak"]
    for a in acts:
        s0, s1 = a["start"], a["end"]
        dur = s1 - s0
        if dur < SPLIT_MIN_ACT_S:
            out.append(a)
            continue
        cuts: List[float] = []
        for p in peaks:
            t = float(p.get("t", 0))
            if s0 + SPLIT_PEAK_MARGIN_S <= t <= s1 - SPLIT_PEAK_MARGIN_S:
                cuts.append(t)
        if dur > ACT_MAX_S:
            # 幕内最长静音起点
            best_t, best_d = None, 0.0
            for s in silences or []:
                st, en = float(s.get("start", 0)), float(s.get("end", 0))
                ov = _overlap(s0, s1, st, en)
                if ov > best_d and s0 + 1.0 <= st <= s1 - 1.0:
                    best_t, best_d = st, ov
            if best_t is not None:
                cuts.append(best_t)
        cuts = sorted(c for c in cuts if s0 < c < s1)
        if not cuts:
            out.append(a)
            continue
        prev = s0
        for i, c in enumerate(cuts):
            seg = dict(a)
            seg["start"], seg["end"] = _fmt(prev), _fmt(c)
            seg["_hard"] = True  # 切分边界：不可被后续合并
            out.append(seg)
            prev = c
        if prev < s1:
            seg = dict(a)
            seg["start"], seg["end"] = _fmt(prev), _fmt(s1)
            seg["_hard"] = True
            out.append(seg)
    return out


def _merge_acts(acts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """相邻幕归并（去布局抖动噪声）：
    - 强边界（_hard，切分产生）不可合并
    - 布局相同（或一方为空）+ 间隔 < MERGE_MIN_GAP_S + 合并后 ≤ ACT_MAX_S → 合并
    - 任一方为短幕（≤ MERGE_MAX_SHORT_S）→ 布局不设限也合并（VL 布局高频抖动=识别噪声）
    视觉内容/text/音频统计在挂载阶段按最终区间重算，此处只合并骨架区间。
    """
    if not acts:
        return []
    merged: List[Dict[str, Any]] = []
    for a in acts:
        if a.get("_hard"):  # 切分强边界：不可合并
            a.pop("_hard", None)
            merged.append(dict(a))
            continue
        if not merged:
            merged.append(dict(a))
            continue
        last = merged[-1]
        same_layout = (not a.get("layout") or not last.get("layout")
                       or a.get("layout") == last.get("layout"))
        short_pair = (a["end"] - a["start"]) <= MERGE_MAX_SHORT_S \
            or (last["end"] - last["start"]) <= MERGE_MAX_SHORT_S
        gap = a["start"] - last["end"]
        if (same_layout or short_pair) and gap <= MERGE_MIN_GAP_S \
                and (a["end"] - last["start"]) <= ACT_MAX_S:
            # 布局冲突（短幕抖动合并）：整体标记为混合布局，避免继续吞并相邻长幕
            if a.get("layout") and last.get("layout") and a["layout"] != last["layout"]:
                last["layout"] = "mixed"
            last["end"] = a["end"]
            last["frames"] = int(last.get("frames", 0)) + int(a.get("frames", 0))
            # 视觉/文字等按时间合并（去重）
            for key in ("visual", "ui_text"):
                for v in a.get(key, []) or []:
                    if v not in last.get(key, []):
                        last.setdefault(key, []).append(v)
                last[key] = last[key][:6]
            for key in ("danmaku", "gifts"):
                last.setdefault(key, []).extend(a.get(key, []) or [])
                last[key] = last[key][:12]
        else:
            merged.append(dict(a))
    for i, a in enumerate(merged):
        a["id"] = i
        a["start"], a["end"] = _fmt(a["start"]), _fmt(a["end"])
    return merged


# ---------------------------------------------------------------------------
# 幕挂载
# ---------------------------------------------------------------------------

def _text_summary(phrases: List[Dict[str, Any]], s0: float, s1: float) -> Dict[str, Any]:
    hits = [p for p in phrases
            if _overlap(s0, s1, float(p.get("start", 0)), float(p.get("end", 0))) > 0]
    hits.sort(key=lambda p: float(p.get("start", 0)))
    text = "".join(str(p.get("text", "")).strip() for p in hits)
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS] + "…"
    return {"text": text, "phrase_count": len(hits)}


def _speaker_stats(
    speaker_segs: Optional[List[Dict[str, Any]]],
    s0: float, s1: float,
) -> Dict[str, Any]:
    if not speaker_segs:
        return {"speakers": [], "speaker_switches": 0}
    hits = [seg for seg in speaker_segs
            if _overlap(s0, s1, float(seg.get("start", 0)), float(seg.get("end", 0))) > 0]
    if not hits:
        return {"speakers": [], "speaker_switches": 0}
    hits.sort(key=lambda x: float(x.get("start", 0)))
    # 按重叠时长降序的活跃说话人
    dur_by_spk: Dict[Any, float] = {}
    for h in hits:
        spk = h.get("speaker_id")
        if spk is None:
            continue
        ov = _overlap(s0, s1, float(h.get("start", 0)), float(h.get("end", 0)))
        dur_by_spk[spk] = dur_by_spk.get(spk, 0.0) + ov
    ordered = sorted(dur_by_spk, key=dur_by_spk.get, reverse=True)  # type: ignore[arg-type]
    switches = sum(
        1 for a, b in zip(hits, hits[1:])
        if a.get("speaker_id") is not None and a.get("speaker_id") != b.get("speaker_id")
    )
    return {"speakers": [int(x) if isinstance(x, (int, float)) else x for x in ordered],
            "speaker_switches": switches}


def _audio_stats(
    events: Optional[List[Dict[str, Any]]],
    silences: Optional[List[Dict[str, Any]]],
    s0: float, s1: float,
) -> Dict[str, Any]:
    evs = events or []
    bursts = [e for e in evs if e.get("type") == "burst"
              and _overlap(s0, s1, float(e.get("t", 0)), float(e.get("end", 0))) > 0]
    up = [e for e in evs if e.get("type") == "speech_rate_jump"
          and e.get("detail", {}).get("direction") == "up"
          and _overlap(s0, s1, float(e.get("t", 0)), float(e.get("end", 0))) > 0]
    down = [e for e in evs if e.get("type") == "speech_rate_jump"
            and e.get("detail", {}).get("direction") == "down"
            and _overlap(s0, s1, float(e.get("t", 0)), float(e.get("end", 0))) > 0]
    peaks = [e for e in evs if e.get("type") == "pre_silence_peak"
             and s0 <= float(e.get("t", 0)) <= s1]
    longest_sil = 0.0
    for s in silences or []:
        ov = _overlap(s0, s1, float(s.get("start", 0)), float(s.get("end", 0)))
        longest_sil = max(longest_sil, ov)
    return {
        "bursts": len(bursts),
        "rate_jumps": {"up": len(up), "down": len(down)},
        "pre_silence_peaks": [
            {"t": _fmt(float(p.get("t", 0))), "silence_start": _fmt(float(p.get("detail", {}).get("silence_start", 0))),
             "silence_duration": _fmt(float(p.get("detail", {}).get("silence_duration", 0)))}
            for p in peaks
        ],
        "longest_silence": _fmt(longest_sil),
    }


def _listen_excitement(
    probes: Optional[Dict[str, Any]],
    s0: float, s1: float,
) -> Optional[float]:
    """幕内直听探针的情绪分均值（无探针返回 None）。"""
    if not probes:
        return None
    vals = [float(p.get("excitement", 0)) for p in (probes.get("probes") or [])
            if s0 <= float(p.get("t", 0)) <= s1 and p.get("excitement") is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


# ---------------------------------------------------------------------------
# emotion 合成（全局归一）
# ---------------------------------------------------------------------------

def _density(act: Dict[str, Any], key: str) -> float:
    dur = act["end"] - act["start"]
    if dur <= 0:
        return 0.0
    if key == "bursts":
        return act["audio"]["bursts"] / dur
    if key == "rate_up":
        return act["audio"]["rate_jumps"]["up"] / dur
    return 0.0


def _normalize(densities: List[float]) -> List[float]:
    mx = max(densities) if densities else 0.0
    if mx <= 0:
        return [0.0] * len(densities)
    return [d / mx for d in densities]


def synthesize_emotion(
    acts: List[Dict[str, Any]],
    probes: Optional[Dict[str, Any]],
) -> None:
    """按方案 §4：emotion = 0.4*burst密度归一 + 0.3*语速陡增密度归一 + 0.3*直听情绪。

    逐幕归一：有直听探针的幕分母 1.0；无探针的幕 0.3 权重按无信号处理、
    分母 0.7 重新归一（避免 listen 只覆盖部分幕时，未覆盖幕被系统性压低）。
    每幕还标记 cold_start（幕内 pre_silence_peak → 反应后冷场）。
    """
    bd = _normalize([_density(a, "bursts") for a in acts])
    rd = _normalize([_density(a, "rate_up") for a in acts])
    for i, a in enumerate(acts):
        exc = _listen_excitement(probes, a["start"], a["end"])
        a["listen_excitement"] = exc
        denom = EMO_BURST_W + EMO_RATE_W + (EMO_LISTEN_W if exc is not None else 0.0)
        e = (EMO_BURST_W * bd[i] + EMO_RATE_W * rd[i]
             + (EMO_LISTEN_W * exc if exc is not None else 0.0))
        a["emotion"] = round(min(1.0, e / denom), 3)
        peaks = a["audio"].get("pre_silence_peaks") or []
        a["cold_start"] = peaks[0] if peaks else None


# ---------------------------------------------------------------------------
# 全局统计
# ---------------------------------------------------------------------------

def _global_stats(
    acts: List[Dict[str, Any]],
    silences: Optional[List[Dict[str, Any]]],
    speaker_segs: Optional[List[Dict[str, Any]]],
    duration: float,
) -> Dict[str, Any]:
    stats: Dict[str, Any] = {
        "act_count": len(acts),
        "long_silences": [{"t": [_fmt(float(s.get("start", 0))), _fmt(float(s.get("end", 0)))],
                           "dur": _fmt(float(s.get("end", 0)) - float(s.get("start", 0)))}
                          for s in (silences or [])
                          if float(s.get("end", 0)) - float(s.get("start", 0)) >= LONG_SILENCE_S],
    }

    # 低能量区拆分：冷场（含 cold_start/长静音）≠ BGM（持续低能量，需 listen 确认）
    bds = [_density(a, "bursts") for a in acts]
    nz = [d for d in bds if d > 0]
    med = float(sorted(nz)[len(nz) // 2]) if nz else 0.0
    thr = med * BGM_DENSITY_RATIO
    lull_regions: List[Dict[str, Any]] = []
    bgm_regions: List[Dict[str, Any]] = []
    for a in acts:
        if _density(a, "bursts") >= thr or (a["end"] - a["start"]) < BGM_MIN_ACT_S:
            continue
        is_lull = a.get("cold_start") is not None or a["audio"].get("longest_silence", 0.0) >= 2.0
        bucket = lull_regions if is_lull else bgm_regions
        if bucket and bucket[-1]["t"][1] >= a["start"]:
            bucket[-1]["t"][1] = a["end"]
            bucket[-1]["act_ids"].append(a["id"])
        else:
            bucket.append({"t": [a["start"], a["end"]], "act_ids": [a["id"]]})
    stats["lull_regions"] = [
        {"t": [_fmt(r["t"][0]), _fmt(r["t"][1])], "act_ids": r["act_ids"],
         "note": "反应后冷场/长静音（pre_silence_peak 或静音≥2s）"}
        for r in lull_regions
    ]
    stats["bgm_regions"] = [
        {"t": [_fmt(r["t"][0]), _fmt(r["t"][1])], "act_ids": r["act_ids"],
         "note": "burst 密度低且非冷场，可能为 BGM/音乐段，需 listen 直听确认"}
        for r in bgm_regions
    ]

    # 高情绪区：emotion ≥ 阈值的连续幕（且幕内 burst ≥ 2，避免长静音幕混入）
    hi: List[Dict[str, Any]] = []
    cur = None
    for a in acts:
        if a["emotion"] >= HIGH_EMOTION_TH and a["audio"]["bursts"] >= HIGH_EMOTION_MIN_BURSTS:
            if cur is None:
                cur = {"t": [a["start"], a["end"]], "act_ids": [a["id"]],
                       "emotion_peak": a["emotion"]}
            else:
                cur["t"][1] = a["end"]
                cur["act_ids"].append(a["id"])
                cur["emotion_peak"] = max(cur["emotion_peak"], a["emotion"])
        else:
            if cur is not None:
                hi.append(cur)
                cur = None
    if cur is not None:
        hi.append(cur)
    stats["high_emotion_regions"] = [
        {"t": [_fmt(r["t"][0]), _fmt(r["t"][1])], "act_ids": r["act_ids"], "emotion_peak": r["emotion_peak"]}
        for r in hi
    ]

    # 说话人占比
    if speaker_segs:
        dur_by_spk: Dict[Any, float] = {}
        for seg in speaker_segs:
            spk = seg.get("speaker_id")
            if spk is None:
                continue
            d = float(seg.get("end", 0)) - float(seg.get("start", 0))
            dur_by_spk[spk] = dur_by_spk.get(spk, 0.0) + max(0.0, d)
        total = max(sum(dur_by_spk.values()), 1e-6)
        stats["speakers"] = {
            str(spk): {"share": round(d / total, 3)}
            for spk, d in sorted(dur_by_spk.items(), key=lambda kv: -kv[1])
        }
    return stats


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def build_story_map(
    phrases_path: str,
    *,
    visual_timeline_path: Optional[str] = None,
    audio_events_path: Optional[str] = None,
    speakers_path: Optional[str] = None,
    silences_path: Optional[str] = None,
    listen_probes_path: Optional[str] = None,
    duration_hint: Optional[float] = None,
) -> Dict[str, Any]:
    phrases = load_phrases(phrases_path)

    tl = _load_json(visual_timeline_path)
    ev = _load_json(audio_events_path)
    spk_raw = _load_json(speakers_path)
    sil_raw = _load_json(silences_path)
    probes = _load_json(listen_probes_path)

    # 说话人段：speaker_timeline.json 与 bound_speaker_timeline.json 结构兼容
    speaker_segs: Optional[List[Dict[str, Any]]] = None
    if spk_raw:
        speaker_segs = spk_raw.get("segments") or spk_raw.get("acts") or []
        if not isinstance(speaker_segs, list):
            speaker_segs = None

    silences: Optional[List[Dict[str, Any]]] = None
    if isinstance(sil_raw, list):
        silences = sil_raw
    elif sil_raw:
        silences = sil_raw.get("silences") or sil_raw.get("segments") or []

    events = (ev.get("events") or []) if ev else None

    # 素材时长：显式 --duration > 视觉时间轴 duration > 各信号覆盖的最大 end。
    # 纯音频模式不能只取转写 end——ASR 判静音会提前停，尾部静音区/BGM 候选会丢。
    sig_ends = [float(p.get("end", 0.0)) for p in phrases] if phrases else []
    if silences:
        sig_ends.extend(float(s.get("end", 0.0)) for s in silences)
    if speaker_segs:
        sig_ends.extend(float(s.get("end", 0.0)) for s in speaker_segs)
    if events:
        sig_ends.extend(float(e.get("end", e.get("t", 0.0))) for e in events)
    signal_max = max(sig_ends) if sig_ends else 0.0
    if duration_hint:
        duration = float(duration_hint)
    elif tl:
        duration = float(tl.get("duration") or signal_max)
    else:
        duration = signal_max

    if tl:
        acts = load_acts_from_visual(tl)
        acts = _absorb_point_acts(acts)
    else:
        acts = load_acts_rule_based(phrases, silences, speaker_segs, duration)

    # 叙事幕整形：长幕按反应后冷场/长静音切分，碎幕按布局+间隔归并
    acts = _merge_acts(_split_acts(acts, events, silences))

    for a in acts:
        a["text"] = _text_summary(phrases, a["start"], a["end"])
        a["speakers"] = _speaker_stats(speaker_segs, a["start"], a["end"])
        a["audio"] = _audio_stats(events, silences, a["start"], a["end"])

    synthesize_emotion(acts, probes)

    return {
        "video": (tl or {}).get("built_from") or Path(phrases_path).stem,
        "duration": _fmt(duration),
        "sources": {
            "phrases": phrases_path,
            "visual_timeline": visual_timeline_path,
            "audio_events": audio_events_path,
            "speakers": speakers_path,
            "silences": silences_path,
            "listen_probes": listen_probes_path,
        },
        "global": _global_stats(acts, silences, speaker_segs, duration),
        "acts": acts,
    }


def save_story_map(sm: Dict[str, Any], out_path: str) -> str:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(sm, ensure_ascii=False, indent=1), encoding="utf-8")
    return str(p)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(prog="story_map.py", description="叙事简报合成器（Story Map Phase 1）")
    ap.add_argument("--phrases", required=True, help="phrases.json 或 words.json")
    ap.add_argument("--visual-timeline", default=None, help="visual_timeline.json（幕骨架，推荐）")
    ap.add_argument("--audio-events", default=None, help="audio_events.json（可选）")
    ap.add_argument("--speakers", default=None, help="speaker_timeline.json / bound_speaker_timeline.json（可选）")
    ap.add_argument("--silences", default=None, help="silences.json（可选）")
    ap.add_argument("--listen", default=None, help="listen_probes.json（可选，提供幕级直听情绪）")
    ap.add_argument("--duration", type=float, default=None, help="素材时长（秒，无视觉且短语未覆盖结尾时用）")
    ap.add_argument("--out", default=None, help="输出 story_map_raw.json 路径")
    args = ap.parse_args()

    sm = build_story_map(
        args.phrases,
        visual_timeline_path=args.visual_timeline,
        audio_events_path=args.audio_events,
        speakers_path=args.speakers,
        silences_path=args.silences,
        listen_probes_path=args.listen,
        duration_hint=args.duration,
    )
    out = args.out or str(Path(args.phrases).with_name("story_map_raw.json"))
    save_story_map(sm, out)
    print(json.dumps({
        "ok": True,
        "out": out,
        "duration": sm["duration"],
        "acts": len(sm["acts"]),
        "lull_regions": len(sm["global"].get("lull_regions", [])),
        "bgm_regions": len(sm["global"].get("bgm_regions", [])),
        "high_emotion_regions": len(sm["global"].get("high_emotion_regions", [])),
        "long_silences": len(sm["global"].get("long_silences", [])),
    }, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
