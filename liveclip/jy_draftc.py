"""jy_draftc.py — 剪映草稿解密/读取封装。

依赖外部工具 jy-draftc（https://github.com/wenshui330/jy-draftc）：
  jy-draftc --dec <encrypted-json> [output]   # 解密（加载剪映 videoeditor.dll）
  jy-draftc --enc <plaintext-json> [output]   # 加密（带回环验证）

用途：用户在剪映里手动精修草稿后，剪映以加密格式保存；本模块解密读回，
用于"学习"用户的样式/剪辑修改（样式传承、剪法沉淀）。

exe 获取方式（二选一）：
  1. 网络恢复后下载 jy-draftc release 的预编译 exe（或 jy-draft-port 的 JYDraftPort.exe 内嵌）
  2. 本机安装 MSYS2 MinGW-w64 后用 scripts/build-native.ps1 编译
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def find_exe() -> Optional[str]:
    """查找 jy-draftc 可执行文件：环境变量 → PATH → 常见位置。"""
    p = os.environ.get("JY_DRAFTC_PATH")
    if p and Path(p).is_file():
        return p
    p = shutil.which("jy-draftc")
    if p:
        return p
    for cand in [
        Path(os.environ.get("LOCALAPPDATA", "")) / "JYDraftPort" / "jy-draftc.exe",
        Path(os.environ.get("USERPROFILE", "")) / ".local" / "bin" / "jy-draftc.exe",
    ]:
        if cand.is_file():
            return str(cand)
    return None


def find_install_dir() -> Optional[str]:
    """找含 videoeditor.dll 的剪映版本目录（JY_INSTALL_DIR 或自动探测）。"""
    env = os.environ.get("JY_INSTALL_DIR")
    if env and (Path(env) / "videoeditor.dll").is_file():
        return env
    base = Path(os.environ.get("LOCALAPPDATA", "")) / "JianyingPro" / "Apps"
    if base.is_dir():
        vers = sorted([d for d in base.iterdir() if d.is_dir() and (d / "videoeditor.dll").is_file()])
        if vers:
            return str(vers[-1])  # 最新版本
    return None


def decrypt_file(encrypted_path: str, output_path: Optional[str] = None) -> str:
    """解密剪映草稿 JSON，返回明文 JSON 文本。"""
    exe = find_exe()
    if not exe:
        raise FileNotFoundError(
            "未找到 jy-draftc.exe。请设置 JY_DRAFTC_PATH，或网络恢复后下载 "
            "https://github.com/wenshui330/jy-draftc 的 release / 用 MSYS2 编译。"
        )
    install = find_install_dir()
    if not install:
        raise FileNotFoundError("未找到含 videoeditor.dll 的剪映安装目录")
    env = dict(os.environ, JY_INSTALL_DIR=install)
    cmd = [exe, "--dec", encrypted_path]
    if output_path:
        cmd.append(output_path)
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(f"jy-draftc 解密失败: {r.stderr[-500:] or r.stdout[-500:]}")
    if output_path:
        return Path(output_path).read_text(encoding="utf-8")
    return r.stdout


def encrypt_file(plaintext_path: str, output_path: Optional[str] = None) -> str:
    """加密明文草稿 JSON（写入剪映前使用），返回密文文本。"""
    exe = find_exe()
    if not exe:
        raise FileNotFoundError("未找到 jy-draftc.exe")
    install = find_install_dir()
    if not install:
        raise FileNotFoundError("未找到含 videoeditor.dll 的剪映安装目录")
    env = dict(os.environ, JY_INSTALL_DIR=install)
    cmd = [exe, "--enc", plaintext_path]
    if output_path:
        cmd.append(output_path)
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(f"jy-draftc 加密失败: {r.stderr[-500:] or r.stdout[-500:]}")
    if output_path:
        return Path(output_path).read_text(encoding="utf-8")
    return r.stdout


def load_draft_content(draft_dir: str) -> Dict[str, Any]:
    """读取草稿内容：明文直接读；加密则尝试 jy-draftc 解密。"""
    p = Path(draft_dir) / "draft_content.json"
    raw = p.read_bytes()
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        text = decrypt_file(str(p))
        return json.loads(text)


# ---------------- 样式提取（从用户精修后的草稿学习） ----------------

def extract_text_style(content: Dict[str, Any]) -> Dict[str, Any]:
    """从解密草稿提取字幕样式（供 build_draft.py 的 style_reference 复用）。

    返回：{border, shadow, size, position_y, font}，仅含用户实际设置的项。
    """
    texts = content.get("materials", {}).get("texts", [])
    if not texts:
        return {}
    t = texts[0]
    c = t.get("content", {})
    styles = (c.get("styles") or [{}])[0]
    clip = None
    for seg in content.get("tracks", []):
        if seg.get("type") == "text" and seg.get("segments"):
            clip = seg["segments"][0].get("clip", {})
            break
    out: Dict[str, Any] = {}
    for s in styles.get("strokes", []) or []:
        solid = (s.get("content") or {}).get("solid", {})
        out["border"] = {
            "alpha": s.get("alpha", 1.0),
            "color": tuple(solid.get("color", [0, 0, 0])),
            "width": s.get("width", 0.08) * 100 / 0.2,  # 反向换算回 0-100
        }
        break
    for s in styles.get("shadows", []) or []:
        solid = (s.get("content") or {}).get("solid", {})
        out["shadow"] = {
            "alpha": s.get("alpha", 1.0),
            "color": tuple(solid.get("color", [0, 0, 0])),
            "diffuse": s.get("diffuse", 0.025) * 100 * 6,  # 反向换算回 0-100
            "distance": s.get("distance", 5.0),
            "angle": s.get("angle", -45.0),
        }
        break
    if clip:
        tr = clip.get("transform", {})
        if "y" in tr:
            out["position_y"] = tr["y"]
        scale = clip.get("scale", {})
        if scale.get("y") and scale["y"] != 1.0:
            out["size_scale"] = scale["y"]
    return out
