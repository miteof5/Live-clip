"""storyboard.py — 图文分镜：内容感知抽帧 + 帧描述（判断层视觉输入）。

目标：把"画面"喂给判断层 LLM，弥补纯文字转写看不见画面的盲区。

抽帧原理（固定预算，内容感知，非均匀网格）：
1. 场景切换点（scdet）→ 最高价值锚点（画面内容变化处），始终优先
2. 冻结段起点（freezedetect）→ 次要锚点（画面恢复运动处）
3. 其余预算 → 最大最小距离（FPS）选帧：每次取"距已选帧最远"的点
   铺满时间轴，避免小预算时帧挤在开头（如固定布局连麦直播）。

预算与帧集关系：预算 B = 覆盖最优序列的前 B 个（嵌套性质），
因此"测试不同帧数"只需全量抽一次，各预算取前缀即可，成本 O(1)。

输出 storyboard.json（判断层输入）：
    {video, duration, fps, width, height, budget, sampling,
     frames: [{t, reason, frame, path, desc?}]}

可选：VL 模型（qwen3-vl-plus）逐帧生成中文描述 → 图文分镜；
再加一步 build_visual_timeline 把帧描述聚合成"视觉上下文时间轴"
（按幕聚合布局/弹幕/礼物/界面文字，判断层直接读）。

用法（CLI）：
    python -m liveclip.storyboard --video <素材.mp4> [--budget 120] [--out storyboard.json]
    python -m liveclip.storyboard --video <素材.mp4> --describe --timeline
    # --describe 生成帧描述（需 DASHSCOPE_API_KEY）；--timeline 生成视觉上下文时间轴
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

try:  # 模块方式
    from .config import find_ffmpeg, load_config, output_dir
    from .probe import probe_media
    from .visual_signal import extract_frozen_segments, extract_scene_changes
except ImportError:  # 直接运行（python liveclip/storyboard.py）
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from liveclip.config import find_ffmpeg, load_config, output_dir
    from liveclip.probe import probe_media
    from liveclip.visual_signal import extract_frozen_segments, extract_scene_changes


# ---------------------------------------------------------------------------
# 候选帧排序（内容感知）
# ---------------------------------------------------------------------------
@dataclass
class FrameCandidate:
    """一个候选抽帧点。tier 越低优先级越高：0=场景切换 1=冻结起点 2=均匀补帧。"""
    t: float
    tier: int
    score: float          # 同 tier 内排序分（场景分 / 空隙长度）
    reason: str
    anchor: bool = False  # 是否锚点（非补帧）

    def to_dict(self) -> Dict[str, Any]:
        return {"t": round(self.t, 3), "tier": self.tier,
                "score": round(self.score, 3), "reason": self.reason}


def rank_frame_candidates(
    duration: float,
    scene_changes: List[Dict[str, Any]],
    frozen_segments: List[Dict[str, Any]],
    *,
    merge_window: float = 0.8,
    max_candidates: int = 360,
    fill_grid_step: Optional[float] = None,
) -> List[FrameCandidate]:
    """按优先级返回候选帧列表（未截断前含锚点+补帧；返回已按 t 排序）。

    - 场景切换锚点：tier 0，score = scdet 切换分（越高越优先）
    - 冻结起点锚点：tier 1
    - 补帧：tier 2，score = 距最近锚点距离（空隙越大越优先）
    锚点间距离 ≤ merge_window 时只保留更优者（防重复帧）。
    """
    if duration <= 0:
        return []

    anchors: List[FrameCandidate] = []
    for sc in scene_changes or []:
        try:
            t = float(sc.get("t", -1.0))
            score = float(sc.get("score", 0.0))
        except (TypeError, ValueError):
            continue
        if 0.0 <= t <= duration:
            anchors.append(FrameCandidate(t, 0, score, f"scene:{score:.1f}", anchor=True))
    for fs in frozen_segments or []:
        try:
            t = float(fs.get("start", -1.0))
        except (TypeError, ValueError):
            continue
        if 0.0 <= t <= duration:
            anchors.append(FrameCandidate(t, 1, 0.0, "frozen", anchor=True))

    # 锚点去重（窗口内保留 tier 更小 / score 更高者）
    anchors.sort(key=lambda c: (c.t, c.tier, -c.score))
    dedup: List[FrameCandidate] = []
    for c in anchors:
        if dedup and abs(c.t - dedup[-1].t) <= merge_window:
            prev = dedup[-1]
            if c.tier < prev.tier or (c.tier == prev.tier and c.score > prev.score):
                dedup[-1] = c
            continue
        dedup.append(c)
    anchors = dedup

    # 补帧网格：步长随时长自适应（短片密、长片疏），避开锚点窗口
    step = fill_grid_step or max(1.0, min(5.0, duration / 600.0))
    fills: List[FrameCandidate] = []
    t = 0.0
    while t <= duration:
        near = min((abs(t - a.t) for a in anchors), default=1e9)
        if near > merge_window:
            fills.append(FrameCandidate(round(t, 3), 2, round(near, 3), "fill"))
        t += step
        if len(fills) > max_candidates * 4:  # 防死循环
            break

    # 优先级排序（tier 升序 → 同 tier score 降序），截断到 max_candidates
    ordered = sorted(anchors + fills, key=lambda c: (c.tier, -c.score))
    top = ordered[:max_candidates]
    top.sort(key=lambda c: c.t)
    return top


def frames_for_budget(ranked: List[FrameCandidate], budget: int) -> List[FrameCandidate]:
    """预算 B 的帧集 = 覆盖最优序列的前 B 个（嵌套：预算递增只增不减）。

    选择策略（max-min 最远点采样，保证任意预算都铺满时间轴）：
    1. 场景切换锚点先按 FPS 排序（种子 = 切换分最高的场景点）
       —— 场景少且价值高，预算再小也优先且铺开；
    2. 其余（冻结锚点 + 补帧）以已选集合为种子继续 FPS：
       每次取"距已选帧最远"的点，让帧均匀覆盖而不是挤在开头。
    """
    if budget <= 0 or not ranked:
        return []
    scenes = [c for c in ranked if c.tier == 0]
    rest = [c for c in ranked if c.tier != 0]

    # 1) 场景锚点 FPS 排序（种子 = 切换分最高者；同分取较早）
    scene_seq: List[FrameCandidate] = []
    if scenes:
        seed = max(scenes, key=lambda c: (c.score, -c.t))
        scene_seq.append(seed)
        remaining = [c for c in scenes if c is not seed]
        picked_ts = [seed.t]
        while remaining:
            best = max(remaining, key=lambda c: (min(abs(c.t - p) for p in picked_ts), c.score, -c.t))
            remaining.remove(best)
            scene_seq.append(best)
            picked_ts.append(best.t)

    seq: List[FrameCandidate] = list(scene_seq[:budget])
    if len(seq) >= budget:
        return sorted(seq, key=lambda c: c.t)

    # 2) 其余点 FPS 续接：种子 = 已选场景 + 已选冻结/补帧
    picked_ts = [c.t for c in seq]
    rest_pts = [(c, min((abs(c.t - p) for p in picked_ts), default=1e18)) for c in rest]
    while len(seq) < budget and rest_pts:
        # 距已选集合最远者优先；并列时冻结锚点(tier1)先于补帧(tier2)
        best, _d = max(rest_pts, key=lambda x: (x[1], -x[0].tier, -x[0].t))
        seq.append(best)
        # 关键：已选点必须从候选池移除，否则会被重复选中
        rest_pts = [(r, min(d, abs(r.t - best.t))) for r, d in rest_pts if r is not best]
        picked_ts.append(best.t)
    return sorted(seq, key=lambda c: c.t)


# ---------------------------------------------------------------------------
# 抽帧（单遍 ffmpeg select，按帧号精确取帧）
# ---------------------------------------------------------------------------
def extract_frames(
    video_path: str,
    candidates: List[FrameCandidate],
    out_dir: Path,
    fps: float,
    ffmpeg: Optional[str] = None,
    *,
    width: int = 0,
    height: int = 0,
    max_side: int = 960,
    jpeg_q: int = 2,
) -> List[str]:
    """把候选帧抽成 jpg，返回与 candidates 顺序一致的帧文件名列表。

    主路径：单遍 ffmpeg -vf "select=eq(n,N)+..."（帧号精确，解码一遍）。
    失败兜底：逐帧 -ss 快进抽取。
    """
    if not candidates:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    ff = ffmpeg or find_ffmpeg()
    if not ff:
        raise RuntimeError("未找到 ffmpeg")

    # 帧号去重（低 fps 时两个候选可能落在同一帧）
    seen: set = set()
    ns: List[Tuple[FrameCandidate, int]] = []
    for c in candidates:
        n = max(0, int(round(c.t * fps)))
        if n in seen:
            continue
        seen.add(n)
        ns.append((c, n))
    if not ns:
        return []

    # 缩放表达式：长边压到 max_side（保持宽高比）
    if width and height and height > width:
        scale_expr = f"scale=-2:'min({max_side},ih)'"
    else:
        scale_expr = f"scale='min({max_side},iw)':-2"

    sel = "+".join(f"eq(n\\,{n})" for _, n in ns)
    cmd = [
        ff, "-y", "-hide_banner", "-loglevel", "error",
        "-i", video_path,
        "-an", "-sn", "-dn",
        "-vf", f"select='{sel}',{scale_expr}",
        "-fps_mode", "vfr",
        "-q:v", str(jpeg_q),
        str(out_dir / "frame_%05d.jpg"),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")

    names: List[str] = []
    if proc.returncode == 0:
        for i in range(len(ns)):
            p = out_dir / f"frame_{i + 1:05d}.jpg"
            if p.exists():
                names.append(p.name)
            else:
                break  # 缺帧，走兜底
    if len(names) == len(ns):
        return names

    # 兜底：逐帧 -ss（快进到关键帧附近再精确定位）
    names = []
    for c, n in ns:
        t = c.t
        s1 = max(0.0, t - 1.0)
        s2 = t - s1
        out = out_dir / f"frame_{len(names) + 1:05d}.jpg"
        subprocess.run(
            [ff, "-y", "-hide_banner", "-loglevel", "error",
             "-ss", f"{s1:.3f}", "-i", video_path, "-ss", f"{s2:.3f}",
             "-frames:v", "1", "-vf", scale_expr, "-q:v", str(jpeg_q), str(out)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        names.append(out.name if out.exists() else "")
    return names


# ---------------------------------------------------------------------------
# 评估指标（纯函数，可单测）
# ---------------------------------------------------------------------------
def scene_recall(frame_ts: List[float], scene_ts: List[float], tol: float = 2.0) -> Optional[float]:
    """场景切换点被 ±tol 内帧覆盖的比例（内容感知抽帧的核心指标）。"""
    if not scene_ts:
        return None
    hits = sum(1 for s in scene_ts if any(abs(s - t) <= tol for t in frame_ts))
    return hits / len(scene_ts)


def moment_coverage(frame_ts: List[float], moments: List[Tuple[float, float]]) -> Optional[float]:
    """已知关键时间段 [s,e] 被帧覆盖的比例（帧落在段内或 ±1s 边缘）。"""
    if not moments:
        return None
    hits = 0
    for s, e in moments:
        if any((s - 1.0) <= t <= (e + 1.0) for t in frame_ts):
            hits += 1
    return hits / len(moments)


def avg_gap(frame_ts: List[float], duration: float) -> float:
    """时间分辨率代理：1s 探测网格上，最近帧距离的平均值（越小覆盖越密）。"""
    if not frame_ts or duration <= 0:
        return float("inf")
    total = 0.0
    n = 0
    t = 0.5
    while t < duration:
        total += min(abs(t - f) for f in frame_ts)
        n += 1
        t += 1.0
    return total / n if n else float("inf")


# ---------------------------------------------------------------------------
# VL 帧描述（qwen3-vl-flash，批量）
# ---------------------------------------------------------------------------
# 描述提示词 v2：OCR 优先 + 布局优先 + 强抗幻觉。
# 教训1：content 必须用 {"image": ...} 字典（字符串形式图片不会真正附加）。
# 教训2：宽松提示词 + 未见图时会输出看似合理的虚构描述（"主播在桌前写方案"）。
_DESCRIBE_PROMPT_V1 = (
    "以下是按时间顺序从直播录像中抽取的画面帧。请严格按顺序逐帧给出中文描述，"
    "每行格式：[序号] 一句话描述，序号从0开始，不要遗漏任何一帧。"
    "描述要点：画面主体人物/物体、动作、场景、画面内文字（弹幕/挂件/标题/价格等）。"
    "只输出描述行，不要总结。"
)

_DESCRIBE_PROMPT = (
    "以下图片是从直播录像中按时间顺序抽取的画面帧。请逐帧只描述【实际可见】的内容，"
    "严格按以下规则，输出格式每帧一行：[序号] 布局；文字：...；内容：...\n"
    "1. 布局：先说明画面整体结构（如：单画面 / 左右分屏 / 四宫格 / 电脑桌面截图等）。\n"
    "2. 文字：逐字转写画面中所有可读文字（弹幕、礼物记录、标题、挂件、按钮、数字），"
    "按出现位置分行列出；看不清就写'文字模糊'，绝对不要猜测文字内容。\n"
    "3. 内容：只描述确定可见的人物、动作、显著物体；看不到就写'不可见'。\n"
    "4. 严禁推测、脑补、编造场景/背景/服装/道具细节。\n"
    "5. 若某帧与上一帧基本相同，该行写'与上帧同'即可。\n"
    "不要输出总结。"
)

# 单帧提示词（逐帧请求用，无序号/无"与上帧同"要求）
_DESCRIBE_PROMPT_SINGLE = (
    "这是一张从直播录像中抽取的画面帧。请只描述【实际可见】的内容，不要推测、不要编造。\n"
    "输出格式：布局：...；文字：...；内容：...\n"
    "1. 布局：画面整体结构（如：单画面 / 左右分屏 / 四宫格 / 电脑桌面截图）。\n"
    "2. 文字：逐字转写画面中所有可读文字（弹幕、礼物记录、标题、挂件、按钮、数字），"
    "按出现位置列出；看不清写'文字模糊'，绝对不要猜测文字内容。\n"
    "3. 内容：只描述确定可见的人物、动作、显著物体；看不到写'不可见'。\n"
    "只输出上述格式内容，不要多余文字。"
)

_DESC_PREFIX_RE = re.compile(r"^\s*(?:\[?\d+\]?|[第]?\d+[).:：、])\s*")

_DESC_LINE_RE = re.compile(r"^\s*\[(\d+)\]\s*(.+?)\s*$")
# 宽容模式：兼容 "0." / "0:" / "0）" / "0、" / "第0帧" 等序号写法
_DESC_LINE_RE_LOOSE = re.compile(r"^\s*(?:第)?(\d+)\s*[).:：、]\s*(.+?)\s*$")


def resolve_vl_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """解析 VL 配置：config.json vl 段 → 环境变量 DASHSCOPE_API_KEY。"""
    v = cfg.get("vl") or {}
    api_key = str(v.get("api_key") or "")
    if not api_key:
        api_key = os.environ.get(str(v.get("api_key_env") or "DASHSCOPE_API_KEY"), "")
    if not api_key:
        api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    return {
        "model": str(v.get("model") or "qwen3-vl-flash"),
        "base_url": str(v.get("base_url")
                        or "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"),
        "api_key": api_key,
    }


def _parse_desc_response(text: str, n: int) -> List[Optional[str]]:
    """解析批量描述响应 → 按序 desc 列表（缺行置 None）。

    优先严格格式 "[i] ..."，失败则宽松匹配 "i. / i: / i） / i、/ 第i帧"。
    """
    out: List[Optional[str]] = [None] * n
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _DESC_LINE_RE.match(line) or _DESC_LINE_RE_LOOSE.match(line)
        if m:
            idx = int(m.group(1))
            if 0 <= idx < n and out[idx] is None:
                out[idx] = m.group(2).strip()
    return out


def describe_frames(
    frames: List[Dict[str, Any]],
    api_key: str,
    model: str,
    base_url: str,
    *,
    batch_size: int = 1,
    timeout: int = 180,
    max_tokens: int = 4096,
    workers: int = 4,
    retries: int = 2,
    prompt: Optional[str] = None,
) -> List[Optional[str]]:
    """对 frames（需含 path）逐帧调用 VL 生成描述，返回与 frames 等长的 desc 列表。

    协议说明：批量多图 + 编号行解析不可靠（模型编号会跳号/从1开始/合并相似帧），
    因此默认逐帧请求、整段输出即描述（batch_size=1 最稳）。
    batch_size>1 时保留编号行解析，供需要省请求数的场景使用。
    """
    n = len(frames)
    descs: List[Optional[str]] = [None] * n
    single = batch_size <= 1
    text_prompt = (prompt or (_DESCRIBE_PROMPT_SINGLE if single else _DESCRIBE_PROMPT))

    def _call(content: List[Dict[str, str]]) -> Optional[str]:
        payload = {
            "model": model,
            "input": {"messages": [{"role": "user", "content": content}]},
            "parameters": {"max_tokens": max_tokens},
        }
        headers = {"Authorization": f"Bearer {api_key}",
                   "Content-Type": "application/json"}
        last_err: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                resp = requests.post(base_url, headers=headers, json=payload, timeout=timeout)
            except requests.RequestException as e:  # 网络抖动重试
                last_err = e
                time.sleep(1.0 * (attempt + 1))
                continue
            if resp.status_code == 200:
                j = resp.json()
                try:
                    msg = j["output"]["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError):
                    raise RuntimeError(f"VL 响应结构异常: {json.dumps(j, ensure_ascii=False)[:400]}")
                return msg if isinstance(msg, str) else "".join(
                    x.get("text", "") for x in msg if isinstance(x, dict))
            if resp.status_code in (429, 500, 502, 503, 504):  # 限流/瞬时错误重试
                last_err = RuntimeError(f"HTTP {resp.status_code}")
                time.sleep(2.0 * (attempt + 1))
                continue
            raise RuntimeError(f"VL 描述请求失败 HTTP {resp.status_code}: {resp.text[:300]}")
        raise RuntimeError(f"VL 请求重试耗尽: {last_err}")

    def _img_data_uri(f: Dict[str, Any]) -> Optional[str]:
        p = Path(f["path"])
        if not p.exists():
            return None
        return "data:image/jpeg;base64," + base64.b64encode(p.read_bytes()).decode()

    if single:
        # 逐帧：整段输出即描述（去头尾空白；去掉模型自加的编号前缀）
        def _one(i: int) -> Tuple[int, Optional[str]]:
            f = frames[i]
            uri = _img_data_uri(f)
            if uri is None:
                return i, None
            text = (_call([{"image": uri}, {"text": text_prompt}]) or "").strip()
            return i, (_DESC_PREFIX_RE.sub("", text).strip() or None)

        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            for i, text in ex.map(_one, range(n)):
                descs[i] = text
        return descs

    # 批量多图 + 编号行解析（不可靠，仅省请求数时使用）
    for start in range(0, n, batch_size):
        batch = frames[start:start + batch_size]
        content: List[Dict[str, str]] = []
        for f in batch:
            uri = _img_data_uri(f)
            if uri is not None:
                content.append({"image": uri})
        if not content:
            continue
        content.append({"text": text_prompt})
        parsed = _parse_desc_response(_call(content) or "", len(batch))
        for i, d in enumerate(parsed):
            descs[start + i] = d
    # 缺帧单帧重试
    for i, d in enumerate(descs):
        if d is not None:
            continue
        uri = _img_data_uri(frames[i])
        if uri is None:
            continue
        text = (_call([{"image": uri}, {"text": text_prompt}]) or "").strip()
        descs[i] = text or None
    return descs


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def build_storyboard(
    video_path: str,
    *,
    budget: int = 120,
    out_dir: Optional[Path] = None,
    scene_changes: Optional[List[Dict[str, Any]]] = None,
    frozen_segments: Optional[List[Dict[str, Any]]] = None,
    describe: bool = False,
    cfg: Optional[Dict[str, Any]] = None,
    **overrides: Any,
) -> Dict[str, Any]:
    """构建图文分镜 JSON（判断层视觉输入）。"""
    cfg = cfg or load_config()
    sb_cfg = cfg.get("storyboard") or {}
    max_candidates = int(overrides.get("max_candidates", sb_cfg.get("max_candidates", 360)))
    merge_window = float(overrides.get("merge_window", sb_cfg.get("merge_window", 0.8)))
    max_side = int(overrides.get("max_side", sb_cfg.get("max_side", 960)))
    jpeg_q = int(overrides.get("jpeg_q", sb_cfg.get("jpeg_quality", 2)))

    info = probe_media(video_path, cfg)
    if scene_changes is None:
        scene_changes = extract_scene_changes(video_path)
    if frozen_segments is None:
        frozen_segments = extract_frozen_segments(video_path)

    ranked = rank_frame_candidates(
        info.duration_s, scene_changes, frozen_segments,
        merge_window=merge_window, max_candidates=max_candidates)
    cands = frames_for_budget(ranked, budget)

    out_root = Path(out_dir) if out_dir else output_dir(cfg) / "work" / "storyboard"
    safe = re.sub(r"[^\w\-]", "_", Path(video_path).stem)
    frames_dir = out_root / "frames" / safe
    names = extract_frames(video_path, cands, frames_dir, info.fps,
                           find_ffmpeg(cfg), width=info.width, height=info.height,
                           max_side=max_side, jpeg_q=jpeg_q)

    frames: List[Dict[str, Any]] = []
    for c, name in zip(cands, names):
        frames.append({
            "t": round(c.t, 3),
            "reason": c.reason,
            "anchor": c.anchor,
            "frame": name,
            "path": str(frames_dir / name) if name else "",
        })

    n_scene = sum(1 for c in ranked if c.tier == 0)
    n_frozen = sum(1 for c in ranked if c.tier == 1)
    n_fill = sum(1 for c in ranked if c.tier == 2)
    frame_ts = [f["t"] for f in frames]

    sb: Dict[str, Any] = {
        "video": str(video_path),
        "duration": round(info.duration_s, 3),
        "fps": info.fps,
        "width": info.width,
        "height": info.height,
        "budget": budget,
        "sampling": {
            "method": "content_aware",
            "scene_anchors": n_scene,
            "frozen_anchors": n_frozen,
            "fill": n_fill,
            "merge_window_s": merge_window,
            "frame_max_side": max_side,
        },
        "metrics": {
            "scene_recall_2s": round(scene_recall(frame_ts, [s["t"] for s in scene_changes]) or 0.0, 3),
            "avg_gap_s": round(avg_gap(frame_ts, info.duration_s), 3),
        },
        "frames": frames,
    }
    if describe:
        vlc = resolve_vl_config(cfg)
        if not vlc["api_key"]:
            raise RuntimeError("未配置 VL API Key（config.json vl 段或 DASHSCOPE_API_KEY）")
        descs = describe_frames(frames, vlc["api_key"], vlc["model"], vlc["base_url"],
                                batch_size=int(sb_cfg.get("describe_batch_size", 1)))
        for f, d in zip(frames, descs):
            f["desc"] = d
        sb["described_by"] = vlc["model"]
        sb["desc_missing"] = sum(1 for d in descs if d is None)
    return sb


def save_storyboard(sb: Dict[str, Any], out_path: str) -> str:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(sb, ensure_ascii=False, indent=1), encoding="utf-8")
    return str(p)


# ---------------------------------------------------------------------------
# 视觉上下文时间轴（判断层输入）
# ---------------------------------------------------------------------------
# 把 120 帧散装描述聚合成"按时间段的视觉证据"：
# 判断层不用读每一帧，读 segments[] 就知道"这段时间画面上在发生什么、
# 公屏弹幕在说什么、有没有打赏/礼物"——公屏反应正是金句/冷场/炸点判据。
_L_LAYOUT_RE = re.compile(r"布局[:：]\s*([^；;\n]+)")
_L_TEXT_RE = re.compile(r"文字[:：]\s*(.*?)(?=\n?\s*内容[:：]|$)", re.S)
_L_CONTENT_RE = re.compile(r"内容[:：]\s*(.+?)\s*$", re.S)

# 画面文字分类（启发式，辅助而非唯一依据；原始 desc 保留可复查）
_GIFT_RE = re.compile(r"送出|送给|点亮|为你闪|闪购|星光|啤酒|火箭|小心心|打赏|礼物|座驾")
_DANMAKU_RE = re.compile(r"来了|走了|点赞|评论|：|？|!|！|~|像|感觉|笑死|哈哈|666|好看|太|本|真的|一下|什么")
_UI_RE = re.compile(r"灯牌|粉丝团|直播中|关注|任务|推荐|标题|按钮|播放|进度|收藏|互动消息|大家都在|搜索|回放")
# 元描述引导行（VL 写的"弹幕内容：""部分模糊"等，不算实际内容）
_META_LINE_RE = re.compile(r"^(弹幕|礼物|评论|文字|其余|部分|更多|多条|后续|内容|以下|可见).{0,12}[:：模糊不可辨截断遮挡重叠]*$|文字模糊|不可辨|截断|遮挡|重复")
# 元描述尾注（行内混着真实文字 + "其余文字模糊不可辨"等尾巴时，只剥尾巴保留实内容）
_META_SUFFIX_RE = re.compile(r"(?:，|,|；|;|：|:)?(?:其余|部分|更多|多条|后续)?(?:文字)?(?:模糊|不可辨|截断|遮挡|重叠).*$")


def _norm_layout(layout: str) -> str:
    """把 VL 的布局自由描述归一化成标准类，避免措辞差异导致分段过碎。"""
    if not layout:
        return "unknown"
    if any(k in layout for k in ("四宫格", "2×2", "2x2", "四画面")):
        return "grid4"
    if any(k in layout for k in ("三宫格", "三部分", "三区域", "三画面", "左大", "左一", "上一下二")):
        return "grid3"
    if any(k in layout for k in ("桌面截图", "电脑桌面", "多个窗口", "多层窗口", "多层嵌套")):
        return "desktop"
    if any(k in layout for k in ("左右分屏", "左右分", "左小窗右主", "分屏")):
        return "split"
    if any(k in layout for k in ("单画面", "单人", "单屏", "居中", "主画面为一人", "主画面为中央",
                                 "人物特写", "人物半身", "一人近景", "居中大图", "左下角小窗",
                                 "小窗画面", "小窗嵌入", "一人半身")):
        return "single"
    return layout[:6]


def _split_text_lines(text: str) -> List[str]:
    """把 desc 的文字段按分隔符切成单条（弹幕/挂件/标题各自成行）。"""
    lines: List[str] = []
    for chunk in re.split(r"[\n；;]", text):
        chunk = chunk.strip().strip("'\"“”‘’")
        if chunk:
            lines.append(chunk)
    return lines


def _classify_text(line: str) -> str:
    """gift=礼物/打赏记录；danmaku=观众发言/反应；ui=挂件/标题/按钮等界面文字；other。"""
    if _GIFT_RE.search(line):
        return "gift"
    if _DANMAKU_RE.search(line):
        return "danmaku"
    if _UI_RE.search(line):
        return "ui"
    return "other"


def build_visual_timeline(sb: Dict[str, Any], *, min_segment_gap: float = 20.0) -> Dict[str, Any]:
    """从（含 desc 的）storyboard 构建视觉上下文时间轴。

    分段规则：
    - 归一化布局变化 → 新幕（如 单画面→四宫格）
    - 帧间距 > min_segment_gap → 新幕（长时间无采样，保险起见分段）
    - 其余并入当前幕（含"与上帧同"）
    每幕聚合：布局、弹幕/礼物/界面文字（**每条带出现时刻 t**，便于定位）、
    内容摘要、幕内帧数。
    """
    frames = sorted(sb.get("frames", []), key=lambda f: f["t"])
    if not frames:
        return {"segments": [], "built_from": None}

    segments: List[Dict[str, Any]] = []
    cur: Optional[Dict[str, Any]] = None

    def _open_seg(f: Dict[str, Any]) -> Dict[str, Any]:
        m = _L_LAYOUT_RE.search(f.get("desc") or "")
        return {
            "start": f["t"], "end": f["t"],
            "layout": _norm_layout(m.group(1).strip() if m else ""),
            "frames": 0, "danmaku": [], "gifts": [], "ui_text": [],
            "content": [],
        }

    def _add_text(cur_seg: Dict[str, Any], line: str, t: float) -> None:
        kind = _classify_text(line)
        bucket = {"gift": "gifts", "danmaku": "danmaku", "ui": "ui_text"}.get(kind, "ui_text")
        items = cur_seg[bucket]
        if not any(it["text"] == line for it in items):
            items.append({"t": round(t, 3), "text": line})

    for f in frames:
        desc = f.get("desc") or ""
        t = f["t"]
        if "与上帧同" in desc and cur is not None:
            cur["end"] = t
            cur["frames"] += 1
            continue
        m = _L_LAYOUT_RE.search(desc)
        layout = _norm_layout(m.group(1).strip() if m else "")
        if cur is not None and (t - cur["end"]) <= min_segment_gap \
                and (not layout or layout == cur["layout"]):
            # 并入当前幕
            cur["end"] = t
            cur["frames"] += 1
        else:
            if cur is not None:
                segments.append(cur)
            cur = _open_seg(f)
            cur["frames"] += 1
        # 收集文字/内容
        m = _L_TEXT_RE.search(desc)
        if m:
            for raw in _split_text_lines(m.group(1)):
                line = _META_SUFFIX_RE.sub("", raw).strip("，,；;：: ")
                if not line or _META_LINE_RE.search(line):
                    continue
                _add_text(cur, line, t)
        m = _L_CONTENT_RE.search(desc)
        if m:
            c = m.group(1).strip()
            if c and c not in cur["content"]:
                cur["content"].append(c)

    if cur is not None:
        segments.append(cur)

    # 收尾：弹幕/礼物按 t 排序；内容摘要截断
    for s in segments:
        s["end"] = round(s["end"], 3)
        s["start"] = round(s["start"], 3)
        for key in ("gifts", "danmaku", "ui_text"):
            s[key].sort(key=lambda it: it["t"])
        s["content"] = s["content"][:3]
    return {
        "built_from": sb.get("described_by") or "frames",
        "duration": sb.get("duration"),
        "budget": sb.get("budget"),
        "segment_count": len(segments),
        "segments": segments,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(prog="storyboard.py", description="图文分镜：内容感知抽帧 + 帧描述")
    ap.add_argument("--video", required=True, help="素材视频路径")
    ap.add_argument("--budget", type=int, default=120, help="抽帧预算（固定帧数）")
    ap.add_argument("--describe", action="store_true", help="调用 VL 生成帧描述（需 DASHSCOPE_API_KEY）")
    ap.add_argument("--timeline", action="store_true", help="同时生成视觉上下文时间轴 visual_timeline.json（需已描述）")
    ap.add_argument("--max-candidates", type=int, default=360, help="候选帧池上限")
    ap.add_argument("--out", default=None, help="输出 JSON 路径（默认 outputs/work/storyboard/<name>.json）")
    args = ap.parse_args()

    cfg = load_config()
    out = args.out
    if out is None:
        safe = re.sub(r"[^\w\-]", "_", Path(args.video).stem)
        out = str(output_dir(cfg) / "work" / "storyboard" / f"{safe}_b{args.budget}.json")

    sb = build_storyboard(args.video, budget=args.budget, describe=args.describe,
                          cfg=cfg, max_candidates=args.max_candidates)
    save_storyboard(sb, out)
    timeline_out = None
    if args.timeline:
        if not (sb.get("frames") and sb["frames"][0].get("desc")):
            raise SystemExit("--timeline 需要已描述的 storyboard（请加 --describe 或先跑 --describe）")
        tl = build_visual_timeline(sb)
        timeline_out = str(Path(out).with_name(Path(out).stem + "_timeline.json"))
        save_storyboard(tl, timeline_out)
    frames = sb["frames"]
    print(json.dumps({
        "ok": True,
        "video": args.video,
        "duration": sb["duration"],
        "budget": args.budget,
        "frames": len(frames),
        "scene_anchors": sb["sampling"]["scene_anchors"],
        "frozen_anchors": sb["sampling"]["frozen_anchors"],
        "fill": sb["sampling"]["fill"],
        "scene_recall_2s": sb["metrics"]["scene_recall_2s"],
        "avg_gap_s": sb["metrics"]["avg_gap_s"],
        "described": bool(sb.get("frames") and sb["frames"][0].get("desc")),
        "timeline": timeline_out,
        "out": out,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
