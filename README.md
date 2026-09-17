# doubao-live-clip

直播切片自动剪辑工具层：输入 5–60min 直播录像，输出可直接在剪映打开的半成品草稿
（已剪气口、去废话、带字幕），由用户精修收尾。

## 架构

```
用户（导演） ←→ 豆包（剪辑师：LLM 判断） ←→ 本工具层（执行）
```

- **判断层（LLM，不在本仓库代码内）**：读词级转写文本，识别废话/金句，编排节奏，
  产出 `edl.json`（编辑决策表：源时间轴上的 keep/cut 区间 + 理由）。
- **工具层（本仓库）**：确定性执行——探测、抽音频、静音检测、词级转写、短语合并、落剪映草稿。

## 工作流

```
1. probe        python scripts/run_pipeline.py probe --video input.mp4
2. preprocess   python scripts/run_pipeline.py preprocess --video input.mp4
                → outputs/work/audio.wav + silences.json
3. transcribe   python scripts/run_pipeline.py transcribe --audio outputs/work/audio.wav
                → outputs/work/words.json（词级时间戳，百炼/硅基流动 whisper）
4. merge        python scripts/run_pipeline.py merge --words outputs/work/words.json
                → outputs/work/phrases.json + source.srt
5. （豆包/用户）读 words.json/phrases.json，产出 edl.json（v2 格式，见下）
6. draft        python scripts/run_pipeline.py draft --edl outputs/work/edl.json \
                  --draft-folder "<剪映草稿目录>"
                → 剪映草稿文件夹（明文，不撞加密护栏）
```

推荐草稿入口：`scripts/build_draft.py`（EDL + SRT → 草稿，支持多素材混排，见 AGENTS.md）。

## 依赖

- Python 3.10+（推荐 3.11）
- ffmpeg（自动探测：PATH → 剪映安装目录；也可在 config.json 指定）
- ASR API Key（任意一个即可，建议都配用于兜底）：
  - 阿里百炼：环境变量 `DASHSCOPE_API_KEY`（模型 `qwen3-asr-flash-*`，词级时间戳，≤5min/10MB）
  - 硅基流动：环境变量 `SILICONFLOW_API_KEY`（`FunAudioLLM/Whisper-large-v3-turbo`）
- pyJianYingDraft（草稿生成）

## 配置

复制 `config.json.example` 为 `config.json`（或直接使用现成 config.json）：

```json
{
  "asr": {
    "active": ["bailian-qwen3-flash-2025-09-08", "bailian-qwen3-flash-2026-02-10", "siliconflow-whisper"],
    "backends": [ { "name": "...", "provider": "bailian", "base_url": "...", "api_key": "", "api_key_env": "DASHSCOPE_API_KEY", "model": "qwen3-asr-flash-2026-02-10", "max_seconds": 300, "max_mb": 10 }, ... ]
  },
  "draft_folder": "C:\\Users\\<你>\\AppData\\Local\\JianyingPro\\User Data\\Projects\\com.lveditor.draft"
}
```

要点：
- **换模型/换平台只改 config.json，不碰代码**。`active` 列表 = 转写时从差到好的尝试顺序；
  前面的后端失败（额度用尽等）自动切下一个，`transcribe` 输出 `backend_used` 标识实际使用。
- `api_key` 留空时按 `api_key_env` → `DASHSCOPE_API_KEY` → `SILICONFLOW_API_KEY` → `API_KEY` 顺序读环境变量。
- `max_seconds` / `max_mb` 是该后端单次请求上限（百炼 flash 同步版 = 300s/10MB）。
- 旧格式（`siliconflow_api_key` / `whisper_model` / `whisper_base_url`）仍兼容，自动回退为单后端。

## EDL 格式（唯一标准 v2，判断层输出）

```json
{
  "keep": [
    {"source_start": 0.0, "source_end": 12.5, "target_start": 0.0,
     "duration": 12.5, "video": "D:/input.mp4", "reason": "开场金句",
     "is_golden": true, "text": "...", "speaker_id": 1, "dialogue_zoom": false},
    {"source_start": 12.5, "source_end": 20.0, "target_start": 12.5, "duration": 7.5}
  ],
  "cut": [{"start": 20.0, "end": 22.0, "reason": "气口停顿"}],
  "target_duration": 20.0,
  "source_path": "D:/input.mp4", "width": 1920, "height": 1080,
  "source_duration_s": 3600.0, "fps": 30, "title": "切片标题"
}
```

要点：
- 只有 **keep 段列表**是必需；`target_duration` 缺失会自动重算；`duration` 与 `source_end` 任给其一
- `video`：多素材混排时指定该段素材，缺省用顶层 `source_path`
- 旧版 `{"segments": [...]}` 仍可读入（`NormalizedEDL` 自动识别），但不再产出
- 说话人字段（`speaker_id`/`dialogue_zoom`/`speaker_switches`）由 `liveclip/enrich_edl.py` 填充

## 测试

```bash
.venv\Scripts\python.exe -m pytest tests\ -q
# 或逐个手动跑：python tests\test_core.py
```

## 里程碑

- [x] M1 工具层骨架（探测/预处理/转写/合并/EDL/落草稿）
- [ ] M2 真实素材最小闭环（豆包判断 → 草稿 → 剪映打开）
- [ ] M3 判断规则打磨（废话/金句/节奏）
- [ ] M4 转独立 agent 项目（README/打包/发布）
