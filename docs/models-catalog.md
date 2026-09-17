# 百炼模型选型清单（doubao-live-clip 专用）

> 数据来源：2026-09-12 拉取 `https://dashscope.aliyuncs.com/compatible-mode/v1/models`，共 249 个。
> 标注：✅=实测可用 | 🟡=待验证 | ⚠️=有坑 | ❌=不适合本场景

## 一、核心：ASR 词级转写（剪切片的基础）

| 模型 | 状态 | 词级时间戳 | 上限 | 说明 |
|---|---|---|---|---|
| **fun-asr-flash-2026-06-15** | ✅ 已实测 | ✅ `sentence.words[]` 毫秒级 | 5min/10MB | FunASR 系，识别质量好，**可作从差到好第一档** |
| **qwen-audio-3.0-asr-flash** | ✅ 已实测 | ✅ `sentence.words[]` 毫秒级 | 5min/10MB | **主力**，直播口语识别准 |
| qwen3-asr-flash-2026-02-10 | ❌ | ❌ 无（chat.completion 纯文本） | 5min/10MB | 同步版不给时间戳，官方要求用 Filetrans |
| qwen3-asr-flash-2025-09-08 | ❌ | ❌ 同上 | 5min/10MB | 同上（且 compatible 列表未挂载） |
| qwen3-asr-flash-realtime 系列 ×3 | ❌ | 流式实时，需 WebSocket | 不限 | 不适合离线文件转写 |
| qwen3-asr-flash-filetrans（原生 API） | 🟡 | ✅ 有（句级+字级） | 12h/2GB | 需公网 URL，本地文件不可直传 |

## 二、判断层备选（当前由豆包直接判断，不调 API；转 agent 时可用）

| 模型 | 定位 | 说明 |
|---|---|---|
| deepseek-v4-flash / deepseek-v4-pro | 推理/判断主力 | 你已有 DeepSeek API，百炼内也挂了同源模型 |
| qwen3.8-flash | 通用轻量 | 最新一代，成本低 |
| qwen3.8-max | 最强通用 | 需要高判断质量时 |
| kimi-k3 | 长上下文 | 超长转写文本分析可用 |
| qwen3.5-397b-a17b | MoE 大杯 | 强推理 |

## 三、扩展能力（以后可能用到）

### 3.1 多模态/视频理解（画面打分、镜头判断，当前默认关闭）
- qwen3.5-omni-flash / plus：音频+视频理解
- qwen3-vl-flash / plus：纯视觉
- qwen-vl-ocr：画面文字识别（检测直播里出现的关键词/弹幕）

> ⚠️ **实测警告**：Omni 系列虽能"输出带时间戳的转写 JSON"，但时间戳是模型按字数编造的
> （实测 qwen3-omni-flash 每字固定 0.1s 均匀步进），**无真实声学对齐**，不可用于剪辑切点。

### 3.2 TTS 配音（如果以后给切片加解说）
- qwen3-tts-flash：轻量配音
- MiniMax/speech-2.8-hd：高质量配音

### 3.3 Embedding（素材库语义检索）
- qwen3.7-text-embedding-flash

### 3.4 翻译（直播切片出海）
- qwen3.5-livetranslate-flash：实时翻译
- qwen-mt-flash：文本翻译

## 四、用不上的
- 图像生成（qwen-image / wan2.7-image）：与剪辑无关
- 老代 qwen1.x/qwen2.x/qwen3-2507：被新代取代
- rerank（qwen3.7-text-rerank）：素材检索可选项

## 五、结论

**现在只需要 2 个模型干活**（都带词级时间戳，各 36K 秒额度 ≈ 10 小时音频，从差到好消耗）：
1. `fun-asr-flash-2026-06-15`（先用）
2. `qwen-audio-3.0-asr-flash`（主力）
**未来转 agent 时**再加 1 个判断 LLM（推荐 deepseek-v4-flash 或 qwen3.8-flash）。
