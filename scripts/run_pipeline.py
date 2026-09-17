"""doubao-live-clip 命令行入口。

用法示例：
  python scripts/run_pipeline.py probe --video input.mp4
  python scripts/run_pipeline.py preprocess --video input.mp4
  python scripts/run_pipeline.py transcribe --audio outputs/audio.wav
  python scripts/run_pipeline.py merge --words outputs/words.json
  python scripts/run_pipeline.py draft --edl outputs/edl.json --draft-folder "<剪映草稿目录>"

每个子命令输出 JSON（到 stdout），便于豆包 skill 或脚本解析。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 允许从项目根直接运行 scripts/run_pipeline.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liveclip import config as cfg  # noqa: E402
from liveclip.edl import EDL  # noqa: E402
from liveclip.make_draft import DraftOptions, make_draft  # noqa: E402
from liveclip.merge_words import merge_to_phrases, save_srt  # noqa: E402
from liveclip.preprocess import Silence, detect_silence, extract_audio  # noqa: E402
from liveclip.probe import probe_media  # noqa: E402
from liveclip.transcribe import save_transcript, transcribe_audio, transcribe_with_backends  # noqa: E402


def _load_cfg(args) -> dict:
    try:
        return cfg.load_config(Path(args.config) if args.config else None)
    except cfg.ConfigError as e:
        print(json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False))
        sys.exit(1)


def cmd_probe(args) -> int:
    conf = _load_cfg(args)
    try:
        info = probe_media(args.video, conf)
        print(json.dumps({"ok": True, "media": info.to_dict()}, ensure_ascii=False))
        return 0
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False))
        return 1


def cmd_preprocess(args) -> int:
    conf = _load_cfg(args)
    outdir = cfg.output_dir(conf) / "work"
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        info = probe_media(args.video, conf)
        wav = extract_audio(args.video, str(outdir / "audio.wav"), ffmpeg=cfg.find_ffmpeg(conf))
        silences = detect_silence(
            wav,
            noise_db=conf.get("silence_noise_db", -30.0),
            min_duration=conf.get("silence_min_duration", 0.8),
            ffmpeg=cfg.find_ffmpeg(conf),
        )
        sil_path = outdir / "silences.json"
        sil_path.write_text(
            json.dumps([s.to_dict() for s in silences], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        total_silence = sum(s.duration for s in silences)
        print(json.dumps({
            "ok": True,
            "media": info.to_dict(),
            "audio_wav": wav,
            "silences_json": str(sil_path),
            "silence_count": len(silences),
            "total_silence_s": round(total_silence, 2),
        }, ensure_ascii=False))
        return 0
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False))
        return 1


def cmd_transcribe(args) -> int:
    conf = _load_cfg(args)
    backends = cfg.load_asr_backends(conf)
    if not backends:
        print(json.dumps({"ok": False, "error": "未配置任何 ASR 后端"},
                         ensure_ascii=False))
        return 1
    # 可选 --backend 指定单一后端；否则按配置优先级（差→好）依次尝试
    if args.backend:
        picked = [b for b in backends if b["name"] == args.backend]
        if not picked:
            print(json.dumps({"ok": False,
                              "error": f"未找到后端 {args.backend}，可用: "
                                       f"{[b['name'] for b in backends]}"},
                             ensure_ascii=False))
            return 1
        backends = picked
    if not any(b.get("api_key") for b in backends):
        print(json.dumps({
            "ok": False,
            "error": ("所有后端均未配置 api_key。请在 config.json 的 asr.backends "
                      "中填写，或设置环境变量 DASHSCOPE_API_KEY / SILICONFLOW_API_KEY")
        }, ensure_ascii=False))
        return 1
    outdir = cfg.output_dir(conf) / "work"
    outdir.mkdir(parents=True, exist_ok=True)
    silences: list[Silence] = []
    if args.silences:
        raw = json.loads(Path(args.silences).read_text(encoding="utf-8"))
        silences = [Silence(**s) for s in raw]
    try:
        tr, used_backend = transcribe_with_backends(
            args.audio,
            backends,
            silences=silences or None,
            work_dir=outdir / "chunks",
        )
        out = args.out or str(outdir / "words.json")
        save_transcript(tr, out)
        print(json.dumps({
            "ok": True,
            "words_json": out,
            "backend_used": used_backend,
            "word_count": len(tr.words),
            "text_len": len(tr.text),
            "duration_hint": (tr.words[-1].end if tr.words else 0.0),
        }, ensure_ascii=False))
        return 0
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False))
        return 1


def cmd_merge(args) -> int:
    raw = json.loads(Path(args.words).read_text(encoding="utf-8"))
    from liveclip.transcribe import Word
    words = [Word(**w) for w in raw.get("words", [])]
    phrases = merge_to_phrases(words, max_gap=args.max_gap, max_chars=args.max_chars)
    outdir = Path(args.words).parent
    phrases_path = outdir / "phrases.json"
    phrases_path.write_text(
        json.dumps([p.to_dict() for p in phrases], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    srt_path = save_srt(phrases, str(outdir / "source.srt"))
    print(json.dumps({
        "ok": True,
        "phrases_json": str(phrases_path),
        "source_srt": srt_path,
        "phrase_count": len(phrases),
    }, ensure_ascii=False))
    return 0


def cmd_draft(args) -> int:
    edl = EDL.load(args.edl)
    errors = edl.validate()
    if errors:
        print(json.dumps({"ok": False, "errors": errors}, ensure_ascii=False))
        return 1
    edl.compute_target_timeline()
    opts = DraftOptions(
        draft_name=args.name or edl.title,
        allow_replace=args.replace,
        with_subtitle=not args.no_subtitle,
        bgm_path=args.bgm,
        bgm_volume=args.bgm_volume,
    )
    if args.subtitle:
        opts.srt_path = args.subtitle
    try:
        draft_dir = make_draft(edl, args.draft_folder, opts)
        print(json.dumps({
            "ok": True,
            "draft_folder": draft_dir,
            "draft_name": opts.draft_name,
            "target_duration_s": round(edl.target_duration_s, 3),
            "keep_segments": sum(1 for s in edl.segments if s.kind == "keep"),
        }, ensure_ascii=False))
        return 0
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False))
        return 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="doubao-live-clip", description="直播切片自动剪辑工具层")
    parser.add_argument("--config", default=None, help="config.json 路径（默认项目根）")
    sub = parser.add_subparsers(dest="command", required=True)

    p_probe = sub.add_parser("probe", help="探测视频信息")
    p_probe.add_argument("--video", required=True)
    p_probe.set_defaults(func=cmd_probe)

    p_pre = sub.add_parser("preprocess", help="抽音频 + 静音检测")
    p_pre.add_argument("--video", required=True)
    p_pre.set_defaults(func=cmd_preprocess)

    p_tr = sub.add_parser("transcribe", help="词级转写（多后端按优先级尝试）")
    p_tr.add_argument("--audio", required=True)
    p_tr.add_argument("--silences", default=None, help="silences.json（长音频切块用）")
    p_tr.add_argument("--out", default=None, help="输出 words.json 路径")
    p_tr.add_argument("--backend", default=None, help="指定单一后端名（默认按配置顺序尝试）")
    p_tr.set_defaults(func=cmd_transcribe)

    p_mg = sub.add_parser("merge", help="词级 → 短语合并 + 源时间轴 SRT")
    p_mg.add_argument("--words", required=True)
    p_mg.add_argument("--max-gap", type=float, default=0.3)
    p_mg.add_argument("--max-chars", type=int, default=24)
    p_mg.set_defaults(func=cmd_merge)

    p_dr = sub.add_parser("draft", help="EDL → 剪映草稿")
    p_dr.add_argument("--edl", required=True)
    p_dr.add_argument("--draft-folder", required=True, help="剪映草稿目录")
    p_dr.add_argument("--name", default=None, help="草稿名（默认取 EDL.title）")
    p_dr.add_argument("--subtitle", default=None, help="目标时间轴 SRT（默认由 phrases 自动映射）")
    p_dr.add_argument("--no-subtitle", action="store_true")
    p_dr.add_argument("--bgm", default=None, help="铺底配乐路径（可选）")
    p_dr.add_argument("--bgm-volume", type=float, default=0.3)
    p_dr.add_argument("--no-replace", dest="replace", action="store_false", default=True)
    p_dr.set_defaults(func=cmd_draft)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
