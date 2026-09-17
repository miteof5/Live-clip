"""硅基流动 whisper 词级转写。

OpenAI 兼容接口：POST {base_url}/audio/transcriptions
- response_format=verbose_json，timestamp_granularities[]=word → 词级时间戳
- 长音频按静音中点切块（whisper API 有文件大小限制），逐块转写后按偏移合并
"""
from __future__ import annotations

import base64
import json
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from .config import find_ffmpeg
from .preprocess import Silence, find_chunk_boundaries


@dataclass
class Word:
    word: str
    start: float
    end: float
    probability: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Transcript:
    text: str
    words: List[Word]
    language: str = "zh"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "language": self.language,
            "words": [w.to_dict() for w in self.words],
        }


class TranscribeError(RuntimeError):
    pass


def _split_audio_chunks(
    wav_path: str,
    boundaries: List[float],
    out_dir: Path,
    ffmpeg: Optional[str] = None,
) -> List[Path]:
    """按边界切 wav 为多块（块间不重叠）。"""
    ff = ffmpeg or find_ffmpeg()
    if not ff:
        raise RuntimeError("未找到 ffmpeg")
    chunks: List[Path] = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        dur = boundaries[i + 1] - start
        if dur <= 0.5:
            continue
        out = out_dir / f"chunk_{i:03d}.wav"
        proc = subprocess.run(
            [ff, "-y", "-hide_banner", "-loglevel", "error",
             "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", wav_path,
             "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(out)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        if proc.returncode != 0 or not out.exists():
            raise TranscribeError(f"切块失败 [{i}]: {proc.stderr.strip()}")
        chunks.append(out)
    return chunks


def _transcribe_one(
    audio_path: Path,
    api_key: str,
    model: str,
    base_url: str,
    language: str,
    timeout: int = 300,
) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/audio/transcriptions"
    with open(audio_path, "rb") as f:
        files = {"file": (audio_path.name, f, "audio/wav")}
        data = {
            "model": model,
            "response_format": "verbose_json",
            "timestamp_granularities[]": "word",
            "language": language,
        }
        headers = {"Authorization": f"Bearer {api_key}"}
        resp = requests.post(url, headers=headers, files=files, data=data, timeout=timeout)
    if resp.status_code != 200:
        raise TranscribeError(
            f"转写请求失败 HTTP {resp.status_code}: {resp.text[:300]}"
        )
    return resp.json()


def _transcribe_one_bailian(
    audio_path: Path,
    api_key: str,
    model: str,
    endpoint: str,
    language: str,
    timeout: int = 300,
) -> List[Dict[str, Any]]:
    """百炼原生同步端点转写（qwen-audio-3.0-asr-flash / qwen3-asr-flash）。

    POST {endpoint}（如 https://dashscope.aliyuncs.com/api/v1/services/
    aigc/multimodal-generation/generation），音频以 base64 内联提交。
    返回句子列表（含 sentence.words[] 字级时间戳，毫秒）。
    注意：OpenAI 兼容接口（/audio/transcriptions）对千问3-ASR 不返回时间戳，
    必须走本原生端点才能拿到词级时间戳。
    """
    with open(audio_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    payload: Dict[str, Any] = {
        "model": model,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": "data:audio/wav;base64," + b64,
                            },
                        }
                    ],
                }
            ]
        },
        "parameters": {"format": "wav", "sample_rate": "16000"},
    }
    if language:
        payload["parameters"]["language_hints"] = [language]
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-DashScope-SSE": "disable",
    }
    resp = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
    if resp.status_code != 200:
        raise TranscribeError(
            f"百炼转写请求失败 HTTP {resp.status_code}: {resp.text[:300]}"
        )
    j = resp.json()
    out = j.get("output", {}).get("output", {})
    sentences = out.get("sentences")
    if sentences is None and out.get("sentence"):
        sentences = [out["sentence"]]
    if not isinstance(sentences, list):
        # 兼容 chat.completion 风格响应（qwen3-asr-flash 同步版）：无词级时间戳
        choices = j.get("output", {}).get("choices", [])
        if choices:
            raise TranscribeError(
                f"模型 {model} 同步接口不返回词级时间戳（响应为 chat.completion 纯文本）；"
                "请改用 qwen-audio-3.0-asr-flash 或 filetrans"
            )
        raise TranscribeError(
            f"百炼响应结构异常: {json.dumps(j, ensure_ascii=False)[:300]}"
        )
    return sentences


def _parse_bailian(sentences: List[Dict[str, Any]], offset: float) -> Transcript:
    """解析百炼原生响应：sentence.words[]（毫秒）→ Word（秒）+ 偏移。

    word 可能是单字或词片段，text 为字词，punctuation 为附着的标点，
    合并为带标点的词条（如 "红"+"，" → "红，"）。
    """
    words: List[Word] = []
    parts: List[str] = []
    for s in sentences:
        text = str(s.get("text", "") or "")
        if text:
            parts.append(text)
        for w in s.get("words", []):
            try:
                t = str(w.get("text", "") or "").strip()
                if not t:
                    continue
                begin = float(w.get("begin_time", 0)) / 1000.0 + offset
                end = float(w.get("end_time", 0)) / 1000.0 + offset
                punct = str(w.get("punctuation", "") or "")
                words.append(Word(word=t + punct, start=begin, end=end))
            except (TypeError, ValueError):
                continue
    words.sort(key=lambda x: x.start)
    return Transcript(text="".join(parts), words=words, language="zh")


def transcribe_audio(
    wav_path: str,
    api_key: str,
    model: str = "FunAudioLLM/Whisper-large-v3-turbo",
    base_url: str = "https://api.siliconflow.cn/v1",
    language: str = "zh",
    silences: Optional[List[Silence]] = None,
    chunk_max_seconds: float = 600.0,
    chunk_max_mb: float = 50.0,
    work_dir: Optional[Path] = None,
    retries: int = 1,
    api_mode: str = "openai",
) -> Transcript:
    """对 wav 执行词级转写。

    - api_mode="openai"：OpenAI 兼容 POST {base_url}/audio/transcriptions
      （硅基 whisper，verbose_json + word 时间戳）
    - api_mode="native"：百炼原生多模态端点（base_url 为完整端点 URL，
      音频 base64 内联，sentence.words[] 毫秒时间戳）
    - 短音频（≤ 后端时长/大小双上限）直接转写
    - 长音频按静音中点切块（切块统一 16kHz 单声道 wav，控制体积），
      逐块转写，词级时间戳加偏移合并
    - chunk_max_mb 与 chunk_max_seconds 共同决定单块上限
    """
    wav = Path(wav_path)
    if not wav.exists():
        raise FileNotFoundError(f"音频不存在: {wav_path}")

    work = work_dir or wav.parent / "chunks"
    work.mkdir(parents=True, exist_ok=True)

    # 双约束：时长 + 文件大小（16kHz/16bit/mono wav ≈ 32000 B/s）
    max_dur_by_size = chunk_max_mb * 1024 * 1024 / 32000.0 * 0.9
    chunk_limit = min(chunk_max_seconds, max_dur_by_size)

    # 判断是否切块：按时长与文件大小
    need_split = False
    file_size_mb = wav.stat().st_size / (1024 * 1024)
    # 用 pymediainfo/ffmpeg 估时长（简化：wav 时长 = 文件字节 / 采样率 / 2）
    # 16000Hz * 2B = 32000 B/s
    approx_dur = wav.stat().st_size / 32000.0
    if approx_dur > chunk_limit or file_size_mb > chunk_max_mb:
        need_split = True

    def _transcribe_chunk(chunk: Path, offset: float) -> Transcript:
        if api_mode == "native":
            sentences = _transcribe_one_bailian(
                chunk, api_key, model, base_url, language
            )
            return _parse_bailian(sentences, offset=offset)
        raw = _transcribe_one(chunk, api_key, model, base_url, language)
        return _parse_verbose_json(raw, offset=offset)

    if not need_split:
        return _transcribe_chunk(wav, 0.0)

    # 长音频：切块
    boundaries = find_chunk_boundaries(silences or [], chunk_limit)
    if len(boundaries) < 2:
        boundaries = [0.0, approx_dur]
    chunks = _split_audio_chunks(str(wav), boundaries, work, ffmpeg=find_ffmpeg())

    all_words: List[Word] = []
    all_text: List[str] = []
    offset = 0.0
    for i, chunk in enumerate(chunks):
        last_error: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                parsed = _transcribe_chunk(chunk, offset)
                last_error = None
                break
            except Exception as e:  # noqa: BLE001
                last_error = e
                if attempt < retries:
                    time.sleep(2 * (attempt + 1))
        if last_error is not None:
            raise TranscribeError(f"切块 {i} 转写失败: {last_error}")
        all_words.extend(parsed.words)
        if parsed.text:
            all_text.append(parsed.text)
        # 下一块偏移 = 本块起点 + 本块实际转写尾（取块内最后一个词）
        if parsed.words:
            offset = parsed.words[-1].end
        else:
            offset = boundaries[i + 1]
        chunk.unlink(missing_ok=True)

    # 按时间排序并合并相邻词
    all_words.sort(key=lambda w: w.start)
    return Transcript(text="".join(all_text), words=all_words, language=language)


def _parse_verbose_json(raw: Dict[str, Any], offset: float) -> Transcript:
    """解析 verbose_json 响应，提取词级时间戳并加偏移。"""
    words: List[Word] = []
    for w in raw.get("words", []):
        try:
            words.append(Word(
                word=str(w.get("word", "")).strip(),
                start=float(w.get("start", 0)) + offset,
                end=float(w.get("end", 0)) + offset,
                probability=float(w.get("probability", 1.0)),
            ))
        except (TypeError, ValueError):
            continue
    words = [w for w in words if w.word]
    text = str(raw.get("text", "") or "")
    return Transcript(text=text, words=words, language=raw.get("language", "zh"))


def save_transcript(transcript: Transcript, out_path: str) -> str:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(transcript.to_dict(), f, ensure_ascii=False, indent=2)
    return str(p)


def transcribe_with_backends(
    wav_path: str,
    backends: List[Dict[str, Any]],
    silences: Optional[List[Silence]] = None,
    work_dir: Optional[Path] = None,
    retries: int = 1,
) -> tuple[Transcript, str]:
    """按优先级依次尝试各 ASR 后端，返回 (Transcript, 实际使用后端名)。

    用于"从差到好消耗额度/失败兜底"：前面的后端失败（额度用尽、
    4xx、网络错误等）时自动切换到下一个后端；全部失败抛 TranscribeError，
    错误信息列出每个后端的失败原因。
    """
    errors: List[str] = []
    for b in backends:
        if not b.get("api_key"):
            errors.append(f"{b['name']}: 未配置 api_key")
            continue
        try:
            tr = transcribe_audio(
                wav_path,
                api_key=b["api_key"],
                model=b["model"],
                base_url=b["base_url"],
                language=b.get("language", "zh"),
                silences=silences,
                chunk_max_seconds=float(b.get("max_seconds", 300.0)),
                chunk_max_mb=float(b.get("max_mb", 10.0)),
                work_dir=work_dir,
                retries=retries,
                api_mode=str(b.get("api_mode", "openai")),
            )
            return tr, b["name"]
        except Exception as e:  # noqa: BLE001
            errors.append(f"{b['name']} ({b['model']}): {e}")
            continue
    raise TranscribeError(
        "所有 ASR 后端均失败:\n" + "\n".join(errors)
    )
