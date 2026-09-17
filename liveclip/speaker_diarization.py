"""speaker_diarization.py — 说话人分离（音频主）。

基于百炼 paraformer-v2 录音文件识别（异步任务接口）：
1. ffmpeg 转 16kHz 单声道 wav（说话人分离仅支持单声道）
2. 百炼临时存储上传 → 获得 oss:// URL（有效期 48h）
3. 异步任务提交（diarization_enabled + timestamp_alignment_enabled）
4. 轮询任务状态，成功取 transcription_url 下载结果
5. 解析 sentences[] → SpeakerTimeline（start/end/text/speaker_id/words[]）

参考（已查证）：
- RESTful API: https://help.aliyun.com/zh/model-studio/paraformer-recorded-speech-recognition-restful-api
- 临时 URL 上传: https://help.aliyun.com/zh/model-studio/get-temporary-file-url

用法（CLI）：
    python liveclip/speaker_diarization.py --video <素材.mp4> [--speakers 2] [--out speaker_timeline.json]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

try:  # 模块方式（from liveclip.speaker_diarization import ...）
    from .config import find_ffmpeg, load_config
    from .transcribe import Word, TranscribeError
except ImportError:  # 直接运行（python liveclip/speaker_diarization.py）
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
    from liveclip.config import find_ffmpeg, load_config
    from liveclip.transcribe import Word, TranscribeError

# 公共（非 WorkspaceId）服务端点
SUBMIT_URL = "https://dashscope.aliyuncs.com/api/v1/services/audio/asr/transcription"
TASK_URL = "https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"
UPLOAD_POLICY_URL = "https://dashscope.aliyuncs.com/api/v1/uploads"


@dataclass
class SpeakerSegment:
    """一个语音片段（句级），带说话人编号与词级时间戳。"""
    start: float
    end: float
    text: str
    speaker_id: Optional[int]   # 相对身份（本素材内一致），未知为 None
    words: List[Word] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text,
            "speaker_id": self.speaker_id,
            "words": [w.to_dict() for w in self.words],
        }


@dataclass
class SpeakerTimeline:
    video: str
    duration: float
    speakers_total: Optional[int]
    segments: List[SpeakerSegment]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video": self.video,
            "duration": round(self.duration, 3),
            "speakers_total": self.speakers_total,
            "segments": [s.to_dict() for s in self.segments],
        }


# ---------------------------------------------------------------------------
# 1. 单声道预处理
# ---------------------------------------------------------------------------
def ensure_mono_wav(
    video_path: str,
    out_dir: Path,
    *,
    sample_rate: int = 16000,
    ffmpeg: Optional[str] = None,
) -> Path:
    """转 16kHz 单声道 wav；文件名改纯 ASCII（OSS URL 含中文/空格会下载失败）。

    返回产物路径（视频/已 mono 的 wav 都会产出新文件）。
    """
    ff = ffmpeg or find_ffmpeg()
    if not ff:
        raise RuntimeError("未找到 ffmpeg")
    src = Path(video_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"mono_{src.stem[:40] or 'audio'}.wav"
    # 清掉非 ASCII 字符（保留字母数字下划线连字符点），避免 OSS URL 编码问题
    safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in out.name)
    out = out_dir / safe_name

    proc = subprocess.run(
        [ff, "-y", "-hide_banner", "-loglevel", "error",
         "-i", str(src),
         "-vn", "-ac", "1", "-ar", str(sample_rate),
         "-c:a", "pcm_s16le", str(out)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if proc.returncode != 0 or not out.exists():
        raise TranscribeError(f"转单声道失败: {proc.stderr.strip()}")
    return out


# ---------------------------------------------------------------------------
# 2. 临时存储上传 → oss:// URL
# ---------------------------------------------------------------------------
def upload_file_to_oss(
    api_key: str,
    model: str,
    file_path: str,
    *,
    timeout: int = 120,
) -> str:
    """getPolicy 拿上传凭证 → OSS 表单上传 → 返回 oss://<key>（48h 有效）。

    注意：上传时指定的 model 必须与调用时一致（paraformer-v2）。
    """
    resp = requests.get(
        UPLOAD_POLICY_URL,
        params={"action": "getPolicy", "model": model},
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        timeout=timeout,
    )
    if resp.status_code != 200:
        raise TranscribeError(f"获取上传凭证失败 HTTP {resp.status_code}: {resp.text[:300]}")
    policy = resp.json().get("data", {})
    if not policy:
        raise TranscribeError(f"上传凭证响应缺少 data: {resp.text[:300]}")

    fname = Path(file_path).name
    key = f"{policy['upload_dir']}/{fname}"
    with open(file_path, "rb") as f:
        files = {
            "OSSAccessKeyId": (None, policy["oss_access_key_id"]),
            "Signature": (None, policy["signature"]),
            "policy": (None, policy["policy"]),
            "x-oss-object-acl": (None, policy["x_oss_object_acl"]),
            "x-oss-forbid-overwrite": (None, policy["x_oss_forbid_overwrite"]),
            "key": (None, key),
            "success_action_status": (None, "200"),
            "file": (fname, f),
        }
        up = requests.post(policy["upload_host"], files=files, timeout=timeout)
    if up.status_code != 200:
        raise TranscribeError(f"OSS 上传失败 HTTP {up.status_code}: {up.text[:300]}")
    return f"oss://{key}"


# ---------------------------------------------------------------------------
# 3. 异步任务：提交 + 轮询 + 取结果
# ---------------------------------------------------------------------------
def submit_task(
    api_key: str,
    file_url: str,
    *,
    model: str = "paraformer-v2",
    speaker_count: Optional[int] = None,
    language_hints: Optional[List[str]] = None,
    timeout: int = 60,
) -> str:
    """提交录音文件转写任务，返回 task_id。"""
    body: Dict[str, Any] = {
        "model": model,
        "input": {
            "file_urls": [file_url],
            "language_hints": language_hints or ["zh"],
        },
        "parameters": {
            # 说话人分离：每句带 speaker_id（仅单声道；放 parameters 层级才生效）
            "diarization_enabled": True,
            # 时间戳校准：words[] 毫秒词级时间戳
            "timestamp_alignment_enabled": True,
        },
    }
    if speaker_count is not None:
        body["parameters"]["speaker_count"] = int(speaker_count)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-DashScope-Async": "enable",
        # oss:// 临时 URL 必须带此头才能被解析
        "X-DashScope-OssResourceResolve": "enable",
    }
    resp = requests.post(SUBMIT_URL, headers=headers, json=body, timeout=timeout)
    if resp.status_code != 200:
        raise TranscribeError(
            f"提交转写任务失败 HTTP {resp.status_code}: {resp.text[:400]}"
        )
    j = resp.json()
    task_id = j.get("output", {}).get("task_id")
    if not task_id:
        raise TranscribeError(f"提交响应缺少 task_id: {json.dumps(j, ensure_ascii=False)[:400]}")
    return task_id


def poll_task(
    api_key: str,
    task_id: str,
    *,
    timeout: float = 1800.0,
    poll_interval: float = 5.0,
) -> List[Dict[str, Any]]:
    """轮询任务直到 SUCCEEDED/FAILED，返回 output.results[]。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = requests.post(
            TASK_URL.format(task_id=task_id),
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json",
                     "X-DashScope-Async": "enable"},
            timeout=60,
        )
        if resp.status_code != 200:
            raise TranscribeError(f"查询任务失败 HTTP {resp.status_code}: {resp.text[:300]}")
        out = resp.json().get("output", {})
        status = out.get("task_status", "UNKNOWN")
        if status == "SUCCEEDED":
            return out.get("results", [])
        if status in ("FAILED", "CANCELED"):
            raise TranscribeError(f"任务失败 status={status}: {json.dumps(out, ensure_ascii=False)[:400]}")
        time.sleep(poll_interval)
    raise TranscribeError(f"任务轮询超时（{timeout}s）: {task_id}")


def fetch_transcription(transcription_url: str, timeout: int = 120) -> Dict[str, Any]:
    """下载识别结果 JSON（transcription_url 24h 有效）。"""
    resp = requests.get(transcription_url, timeout=timeout)
    if resp.status_code != 200:
        raise TranscribeError(f"下载识别结果失败 HTTP {resp.status_code}: {resp.text[:300]}")
    return resp.json()


# ---------------------------------------------------------------------------
# 4. 结果解析
# ---------------------------------------------------------------------------
def parse_result(payload: Dict[str, Any], offset: float = 0.0) -> List[SpeakerSegment]:
    """解析识别结果 JSON → SpeakerSegment[]。

    sentences[].words[] 为毫秒词级时间戳；speaker_id 缺失时记 None（兜底）。
    """
    segs: List[SpeakerSegment] = []
    for tr in payload.get("transcripts", []):
        for s in tr.get("sentences", []):
            text = str(s.get("text", "") or "")
            try:
                start = float(s.get("begin_time", 0)) / 1000.0 + offset
                end = float(s.get("end_time", 0)) / 1000.0 + offset
            except (TypeError, ValueError):
                continue
            spk = s.get("speaker_id")
            speaker_id = int(spk) if spk is not None else None
            words: List[Word] = []
            for w in s.get("words", []):
                try:
                    t = str(w.get("text", "") or "").strip()
                    if not t:
                        continue
                    wb = float(w.get("begin_time", 0)) / 1000.0 + offset
                    we = float(w.get("end_time", 0)) / 1000.0 + offset
                    punct = str(w.get("punctuation", "") or "")
                    words.append(Word(word=t + punct, start=wb, end=we))
                except (TypeError, ValueError):
                    continue
            words.sort(key=lambda x: x.start)
            segs.append(SpeakerSegment(start=start, end=end, text=text,
                                       speaker_id=speaker_id, words=words))
    segs.sort(key=lambda x: x.start)
    return segs


# ---------------------------------------------------------------------------
# 5. 主流程
# ---------------------------------------------------------------------------
def run_diarization(
    video_path: str,
    api_key: str,
    *,
    model: str = "paraformer-v2",
    speaker_count: Optional[int] = None,
    work_dir: Optional[Path] = None,
    timeout: float = 1800.0,
    poll_interval: float = 5.0,
) -> SpeakerTimeline:
    """完整说话人分离流程：转 mono → 上传 → 提交 → 轮询 → 解析。

    speaker_count：连麦传 2，四人 PK 传 4；None 表示自动判断。
    """
    wav = ensure_mono_wav(video_path, work_dir or Path.cwd() / "outputs" / "work")
    oss_url = upload_file_to_oss(api_key, model, str(wav))
    task_id = submit_task(api_key, oss_url, model=model, speaker_count=speaker_count)
    results = poll_task(api_key, task_id, timeout=timeout, poll_interval=poll_interval)

    if not results:
        raise TranscribeError("任务成功但无 results")
    ok = [r for r in results if r.get("subtask_status") == "SUCCEEDED"]
    if not ok:
        raise TranscribeError(f"全部子任务失败: {json.dumps(results, ensure_ascii=False)[:400]}")
    payload = fetch_transcription(ok[0]["transcription_url"])

    segs = parse_result(payload)
    duration = payload.get("properties", {}).get("original_duration_in_milliseconds", 0)
    speakers = sorted({s.speaker_id for s in segs if s.speaker_id is not None})
    return SpeakerTimeline(
        video=video_path,
        duration=round(duration / 1000.0, 3),
        speakers_total=len(speakers) if speakers else None,
        segments=segs,
    )


def save_timeline(timeline: SpeakerTimeline, out_path: str) -> str:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(timeline.to_dict(), ensure_ascii=False, indent=2),
                 encoding="utf-8")
    return str(p)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(prog="speaker_diarization.py", description="说话人分离（paraformer-v2）")
    ap.add_argument("--video", required=True, help="素材视频/音频路径")
    ap.add_argument("--speakers", type=int, default=None, help="说话人数（连麦 2 / PK 4；默认自动）")
    ap.add_argument("--model", default="paraformer-v2", help="百炼模型名")
    ap.add_argument("--out", default=None, help="输出 JSON 路径（默认 outputs/work/speaker_timeline.json）")
    args = ap.parse_args()

    cfg = load_config()
    api_key = cfg.get("asr", {}).get("backends", [{}])[0].get("api_key", "") or ""
    import os as _os
    if not api_key:
        api_key = _os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        print("未配置 DASHSCOPE_API_KEY（环境变量或 config.json）")
        raise SystemExit(1)

    work = Path(cfg.get("output_dir", "outputs")) / "work"
    out = args.out or str(work / "speaker_timeline.json")
    timeline = run_diarization(args.video, api_key, model=args.model,
                               speaker_count=args.speakers, work_dir=work)
    save_timeline(timeline, out)
    print(json.dumps({
        "ok": True,
        "video": args.video,
        "duration": timeline.duration,
        "speakers_total": timeline.speakers_total,
        "segments": len(timeline.segments),
        "out": out,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
