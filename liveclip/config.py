"""doubao-live-clip 配置模块。

加载项目根目录下的 config.json；未配置时提供默认值与自动探测（ffmpeg 等）。
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

# 项目根目录（config.json 所在处）
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 剪映草稿目录的常见位置（Windows）
JIANYING_DRAFT_CANDIDATES = [
    Path(os.environ.get("LOCALAPPDATA", "")) / "JianyingPro" / "User Data" / "Projects" / "com.lveditor.draft",
    Path(os.environ.get("LOCALAPPDATA", "")) / "JianyingPro" / "User Data" / "Projects" / "com.lveditor.draft" / "draft_content.json",
    Path.home() / "AppData" / "Local" / "JianyingPro" / "User Data" / "Projects" / "com.lveditor.draft",
]

# 剪映安装目录（各版本）
JIANYING_APP_DIRS = [
    Path(os.environ.get("LOCALAPPDATA", "")) / "JianyingPro" / "Apps",
    Path("C:/MyApp/剪映JianyingPro 免V1P"),
]

DEFAULT_CONFIG: Dict[str, Any] = {
    "siliconflow_api_key": "",           # 硅基流动 API Key（用户自填）
    "whisper_model": "FunAudioLLM/Whisper-large-v3-turbo",
    "whisper_base_url": "https://api.siliconflow.cn/v1",
    "ffmpeg_path": "",                   # 留空则自动探测
    "ffprobe_path": "",
    "draft_folder": "",                  # 留空则自动探测剪映草稿目录
    "output_dir": "outputs",             # 中间产物目录（相对项目根）
    "silence_noise_db": -30.0,           # 静音检测阈值 dB
    "silence_min_duration": 0.8,         # 最短静音时长 s
    "chunk_max_seconds": 600,            # 转写单块最长秒数（whisper 文件大小限制）
}


class ConfigError(RuntimeError):
    pass


def load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """加载配置；缺失文件时给出明确提示。"""
    cfg_path = path or (PROJECT_ROOT / "config.json")
    if not cfg_path.exists():
        example = PROJECT_ROOT / "config.json.example"
        raise ConfigError(
            f"未找到配置文件 {cfg_path}。\n"
            f"请复制 {example} 为 config.json 并填入 siliconflow_api_key。"
        )
    with open(cfg_path, "r", encoding="utf-8") as f:
        user_cfg = json.load(f)
    merged = dict(DEFAULT_CONFIG)
    merged.update(user_cfg)
    merged["_config_path"] = str(cfg_path)
    return merged


def find_ffmpeg(cfg: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """探测 ffmpeg 可执行文件路径。

    优先级：config 显式指定 → PATH → 剪映安装目录（取版本号最大者）→ 绿色版剪映。
    未传 cfg 时自动加载 config.json（保证所有调用点都能命中显式配置）。
    """
    if cfg is None:
        try:
            cfg = load_config()
        except ConfigError:
            cfg = None
    if cfg and cfg.get("ffmpeg_path"):
        p = cfg["ffmpeg_path"]
        if Path(p).exists():
            return p
    found = shutil.which("ffmpeg")
    if found:
        return found
    best: Optional[tuple] = None
    for base in JIANYING_APP_DIRS:
        if not base.exists():
            continue
        for ver_dir in base.iterdir():
            if not ver_dir.is_dir():
                continue
            exe = ver_dir / "ffmpeg.exe"
            if exe.exists():
                # 版本目录形如 "8.9.0.13361"，取最大版本
                try:
                    key = tuple(int(x) for x in ver_dir.name.split("."))
                except ValueError:
                    key = (0,)
                if best is None or key > best[0]:
                    best = (key, str(exe))
    return best[1] if best else None


def find_ffprobe(cfg: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """探测 ffprobe；剪映不自带 ffprobe，找不到返回 None。"""
    if cfg is None:
        try:
            cfg = load_config()
        except ConfigError:
            cfg = None
    if cfg and cfg.get("ffprobe_path"):
        p = cfg["ffprobe_path"]
        if Path(p).exists():
            return p
    return shutil.which("ffprobe")


def find_draft_folder(cfg: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """探测剪映草稿目录；config 显式指定优先。"""
    if cfg is None:
        try:
            cfg = load_config()
        except ConfigError:
            cfg = None
    if cfg and cfg.get("draft_folder"):
        p = Path(cfg["draft_folder"])
        if p.exists():
            return str(p)
    for cand in JIANYING_DRAFT_CANDIDATES:
        if cand.exists() and cand.is_dir():
            return str(cand)
    return None


def output_dir(cfg: Dict[str, Any]) -> Path:
    """解析输出目录（绝对路径，自动创建）。"""
    d = Path(cfg.get("output_dir", "outputs"))
    if not d.is_absolute():
        d = PROJECT_ROOT / d
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# ASR 多后端（换模型/换平台不碰代码，只改 config.json 或环境变量）
# ---------------------------------------------------------------------------

# api_key 解析链：config.json 显式值 → 显式指定的环境变量名 → 通用环境变量链
ASR_ENV_KEYS = ["DASHSCOPE_API_KEY", "SILICONFLOW_API_KEY", "API_KEY"]

# 各平台 OpenAI 兼容模式 base_url
DEFAULT_BASE_URLS = {
    "bailian": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "bailian_native": "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation",
    "siliconflow": "https://api.siliconflow.cn/v1",
    "openai": "https://api.openai.com/v1",
}


def _first_env(names: List[str]) -> Optional[str]:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return None


def resolve_api_key(backend: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    """解析后端 api_key：config 显式值 → 环境变量。

    - backend["api_key"] 非空则直接使用；
    - 若指定了 backend["api_key_env"]，只从该环境变量读取（防止
      DASHSCOPE_API_KEY 被误用到硅基后端）；
    - 均未指定时才走通用链 DASHSCOPE_API_KEY → SILICONFLOW_API_KEY → API_KEY，
      再回退旧字段 cfg["siliconflow_api_key"]。
    """
    v = str(backend.get("api_key") or "")
    if v:
        return v
    if backend.get("api_key_env"):
        return _first_env([str(backend["api_key_env"])]) or ""
    v = _first_env(ASR_ENV_KEYS) or ""
    if not v:
        v = str(cfg.get("siliconflow_api_key") or "")
    return v


def normalize_backend(b: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """规范化单个后端：补齐默认值、解析 api_key 与 base_url。

    api_mode 决定转写调用方式：
      - "native"：百炼原生多模态端点（能拿词级时间戳，base_url 为完整端点）
      - "openai"：OpenAI 兼容 /audio/transcriptions（硅基 whisper 等）
    provider=bailian 且未显式指定时，默认走 native。
    """
    model = str(b.get("model") or "").strip()
    name = str(b.get("name") or model or "asr-backend")
    provider = str(b.get("provider") or "").lower()
    api_mode = str(b.get("api_mode") or "").lower()
    if not api_mode:
        api_mode = "native" if provider == "bailian" else "openai"
    base_url = str(b.get("base_url") or "").strip()
    if not base_url:
        if api_mode == "native":
            base_url = DEFAULT_BASE_URLS["bailian_native"]
        else:
            base_url = DEFAULT_BASE_URLS.get(provider, DEFAULT_BASE_URLS["bailian"])
    return {
        "name": name,
        "provider": provider,
        "base_url": base_url,
        "api_key": resolve_api_key(b, cfg),
        "model": model,
        "max_seconds": float(b.get("max_seconds", 300.0)),   # 单次请求最长秒数
        "max_mb": float(b.get("max_mb", 10.0)),              # 单次请求最大文件 MB
        "language": str(b.get("language") or "zh"),
        "api_mode": api_mode,
    }


def load_asr_backends(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """按优先级返回 ASR 后端列表（从差到好，用于逐个消耗额度/兜底）。

    优先读 cfg["asr"]：
      {
        "backends": [ {name, provider, base_url, api_key, api_key_env, model,
                       max_seconds, max_mb, language}, ... ],
        "active":   ["差模型名", "好模型名", ...]   # 使用顺序；缺省按 backends 声明顺序
      }
    未配置 asr 时回退旧字段（siliconflow_api_key / whisper_model /
    whisper_base_url / chunk_max_seconds）构造单后端，保证旧 config.json 仍可用。
    """
    asr_cfg = cfg.get("asr")
    if isinstance(asr_cfg, dict) and asr_cfg.get("backends"):
        raw_list = [x for x in asr_cfg["backends"] if isinstance(x, dict)]
        by_name: Dict[str, Dict[str, Any]] = {}
        for b in raw_list:
            nb = normalize_backend(b, cfg)
            if nb["model"]:
                by_name[nb["name"]] = nb
        active = [str(n) for n in asr_cfg.get("active", [])]
        ordered = [by_name[n] for n in active if n in by_name]
        ordered += [b for name, b in by_name.items() if name not in set(active)]
        return ordered
    # 旧字段回退：单后端硅基 whisper
    api_key = cfg.get("siliconflow_api_key") or _first_env(ASR_ENV_KEYS) or ""
    return [{
        "name": "siliconflow-whisper",
        "provider": "siliconflow",
        "base_url": str(cfg.get("whisper_base_url") or DEFAULT_BASE_URLS["siliconflow"]),
        "api_key": api_key,
        "model": str(cfg.get("whisper_model") or "FunAudioLLM/Whisper-large-v3-turbo"),
        "max_seconds": float(cfg.get("chunk_max_seconds", 600.0)),
        "max_mb": 50.0,
        "language": "zh",
        "api_mode": "openai",
    }]
