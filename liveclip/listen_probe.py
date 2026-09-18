"""listen_probe.py — 候选段"直听"（判断层按需调用，Omni 音频理解）。

规则层（audio_events.py）先零成本筛出线索；判断层对"犹豫段"
（长留白 / 金句候选 / BGM 区）切 3-10s 片段，用 qwen3.5-omni 直听确认：
"这段有笑声 / 掌声 / 惊呼 / 情绪高潮吗"——服务金句/炸点与留白该不该留两类判断。

输出 listen_probes.json：
    {audio, model, probes: [{t, end, laughs, applause, shouts, music, speech,
                             excitement, description, model}]}

用法（CLI）：
    python -m liveclip.listen_probe --audio <audio.wav> --from 30.0 --to 40.0
    python -m liveclip.listen_probe --audio <audio.wav> --events <audio_events.json> --max 8
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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

try:  # 模块方式
    from .config import find_ffmpeg, load_config, output_dir
except ImportError:  # 直接运行
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from liveclip.config import find_ffmpeg, load_config, output_dir

_PROBE_PROMPT = (
    "你听一段直播录音片段。只报告听到的事实，不要推测、不要编造。\n"
    '只输出一个 JSON 对象，格式：\n'
    '{"laughs":bool,"applause":bool,"shouts":bool,"music":bool,"speech":bool,'
    '"excitement":0.0,"description":"一句话说明听到了什么声音"}\n'
    "判断标准：\n"
    "laughs=有明显笑声（观众或主播），语气词不算；\n"
    "applause=掌声/欢呼/口哨；\n"
    "shouts=惊呼/喊叫/尖叫声；\n"
    "music=有背景音乐/BGM/旋律/歌声；\n"
    "speech=有人说话；\n"
    "excitement=情绪强度，0=平淡，1=非常激动。\n"
    "description 用中文一句话概括这段声音（如'主播说话+观众笑声'），不要多余文字。"
)

_JSON_RE = re.compile(r"\{.*\}", re.S)


def slice_audio(
    audio_path: str,
    start: float,
    end: float,
    out_wav: str,
    ffmpeg: Optional[str] = None,
    sample_rate: int = 16000,
) -> str:
    """切 [start, end) 音频段为 16kHz 单声道 wav；返回路径。"""
    ff = ffmpeg or find_ffmpeg()
    if not ff:
        raise RuntimeError("未找到 ffmpeg")
    out = Path(out_wav)
    out.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [ff, "-y", "-hide_banner", "-loglevel", "error",
         "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", audio_path,
         "-vn", "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le",
         str(out)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if proc.returncode != 0 or not out.exists():
        raise RuntimeError(f"切音频失败: {proc.stderr.strip()}")
    return str(out)


def _call_omni(
    audio_uri: str,
    prompt: str,
    api_key: str,
    model: str,
    base_url: str,
    *,
    timeout: int = 120,
    max_tokens: int = 512,
    retries: int = 2,
) -> str:
    """调 DashScope multimodal-generation（音频理解）。"""
    payload = {
        "model": model,
        "input": {"messages": [{"role": "user", "content": [
            {"audio": audio_uri}, {"text": prompt},
        ]}]},
        "parameters": {"max_tokens": max_tokens},
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(base_url, headers=headers, json=payload, timeout=timeout)
        except requests.RequestException as e:
            last_err = e
            time.sleep(1.0 * (attempt + 1))
            continue
        if resp.status_code == 200:
            j = resp.json()
            try:
                msg = j["output"]["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                raise RuntimeError(f"Omni 响应结构异常: {json.dumps(j, ensure_ascii=False)[:400]}")
            return msg if isinstance(msg, str) else "".join(
                x.get("text", "") for x in msg if isinstance(x, dict))
        if resp.status_code in (429, 500, 502, 503, 504):
            last_err = RuntimeError(f"HTTP {resp.status_code}")
            time.sleep(2.0 * (attempt + 1))
            continue
        raise RuntimeError(f"Omni 直听失败 HTTP {resp.status_code}: {resp.text[:300]}")
    raise RuntimeError(f"Omni 请求重试耗尽: {last_err}")


def parse_probe_json(text: str) -> Dict[str, Any]:
    """从模型输出中提取 JSON（容忍代码块/前后文字）。"""
    m = _JSON_RE.search(text or "")
    if not m:
        raise ValueError(f"Omni 输出无 JSON: {text[:200]!r}")
    data = json.loads(m.group(0))
    if not isinstance(data, dict):
        raise ValueError("Omni 输出不是对象")
    return data


def probe_segment(
    audio_path: str,
    start: float,
    end: float,
    *,
    api_key: str,
    model: str,
    base_url: str,
    prompt: Optional[str] = None,
    ffmpeg: Optional[str] = None,
    tmp_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """直听一段音频 → 结构化判断。

    临时切片 wav 用完即删（无论成功失败），不残留堆积。
    """
    tmp = (tmp_dir or Path(audio_path).parent)
    seg = str(tmp / f"_probe_{start:.3f}_{end:.3f}.wav")
    slice_audio(audio_path, start, end, seg, ffmpeg=ffmpeg)
    try:
        uri = "data:;base64," + base64.b64encode(Path(seg).read_bytes()).decode()
        text = _call_omni(uri, prompt or _PROBE_PROMPT, api_key, model, base_url)
        out = parse_probe_json(text)
        out["t"] = round(float(start), 3)
        out["end"] = round(float(end), 3)
        out["model"] = model
        return out
    finally:
        try:
            Path(seg).unlink(missing_ok=True)
        except OSError:
            pass


def resolve_listen_config(cfg: Dict[str, Any]) -> Dict[str, str]:
    """解析 listen 配置：config.json listen 段 → 环境变量。

    注意：DashScope 无裸 `qwen3.5-omni` 模型名，正确名称为
    `qwen3.5-omni-flash` / `qwen3.5-omni-plus`（见 SKILL.md 踩坑记录）。
    """
    l = cfg.get("listen") or {}
    api_key = str(l.get("api_key") or "")
    if not api_key:
        api_key = os.environ.get(str(l.get("api_key_env") or "DASHSCOPE_API_KEY"), "")
    return {
        "model": str(l.get("model") or "qwen3.5-omni-flash"),
        "base_url": str(l.get("base_url")
                        or "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"),
        "api_key": api_key,
    }


def probe_segments(
    audio_path: str,
    segments: List[Tuple[float, float]],
    *,
    cfg: Optional[Dict[str, Any]] = None,
    max_seconds: float = 10.0,
) -> Dict[str, Any]:
    """对多段顺序直听（每段一次 LLM 调用）。

    单段失败不中断：降级为带 `error` 字段的条目，已直听成功的段保留。
    """
    cfg = cfg or load_config()
    lc = resolve_listen_config(cfg)
    if not lc["api_key"]:
        raise RuntimeError("未配置 Omni API Key（config.json listen 段或 DASHSCOPE_API_KEY）")
    probes: List[Dict[str, Any]] = []
    failed = 0
    for start, end in segments:
        if end - start > max_seconds:
            end = start + max_seconds
        try:
            probes.append(probe_segment(audio_path, start, end, **lc))
        except Exception as e:  # noqa: BLE001 —— 单段失败降级，不中断后续段
            failed += 1
            probes.append({
                "t": round(float(start), 3),
                "end": round(float(end), 3),
                "model": lc["model"],
                "error": str(e),
            })
    return {"audio": str(audio_path), "model": lc["model"],
            "probes": probes, "failed": failed}


def main() -> None:
    ap = argparse.ArgumentParser(prog="listen_probe.py", description="候选段直听（Omni 音频理解）")
    ap.add_argument("--audio", required=True, help="音频 wav（16kHz 单声道）")
    ap.add_argument("--from", dest="t0", type=float, default=None, help="直听起点（秒）")
    ap.add_argument("--to", dest="t1", type=float, default=None, help="直听终点（秒）")
    ap.add_argument("--events", default=None, help="audio_events.json：自动取 top N 事件段直听")
    ap.add_argument("--max", type=int, default=8, help="--events 模式最多直听段数")
    ap.add_argument("--out", default=None, help="输出路径（默认 outputs/work/<名>_listen_probes.json）")
    args = ap.parse_args()

    cfg = load_config()
    segments: List[Tuple[float, float]] = []
    if args.t0 is not None:
        if args.t1 is None:
            ap.error("--from 需配 --to")
        segments.append((args.t0, args.t1))
    elif args.events:
        ev = json.loads(Path(args.events).read_text(encoding="utf-8"))
        # 取 top N 高分会话事件段：每事件 t 为中心 6s
        events = sorted(ev.get("events", []), key=lambda e: e.get("score", 0), reverse=True)[: args.max]
        for e in events:
            t = float(e.get("t", 0))
            segments.append((max(0.0, t - 2.0), t + 4.0))
    else:
        ap.error("需要 --from/--to 或 --events")

    res = probe_segments(args.audio, segments, cfg=cfg)
    safe = re.sub(r"[^\w\-]", "_", Path(args.audio).stem)
    out = args.out or str(output_dir(cfg) / "work" / f"{safe}_listen_probes.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"ok": True, "probes": len(res["probes"]), "out": out}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
