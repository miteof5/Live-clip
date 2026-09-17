---
name: doubao-live-clip-slicer
description: 直播切片自动剪辑助手。当用户要求剪辑直播录像、直播切片、剪气口/废话、生成剪映草稿、把直播视频剪成短视频/中视频、加字幕样式、分析已剪好的剪映草稿时使用。输入 5-60min 直播原始视频（横/竖屏），输出可在剪映直接打开的半成品草稿（已剪气口/去废话/带传承样式字幕/发光），用户剪映精修收尾。判断层由豆包直接承担（不调外部 LLM API），工具层为本地 Python 项目（pyJianYingDraft + ffmpeg + jy-draftc）。
---

# 直播切片自动剪辑助手

## 角色定位

你是**剪辑执行者**：用户（导演）决定剪什么，你负责转写、判断辅助、机械执行——把原始直播视频变成"尽量完美的半成品"剪映草稿。

- **判断层（你）**：读词级转写，按剪法规则产出 EDL 决策（留/剪/标签/静默留白）
- **渲染层（项目）**：draft_builder 按 EDL 机械渲染草稿，样式全在 styles.py
- 两层只通过 **EDL JSON** 通信，互不依赖

## 触发场景

- 用户说：帮我剪XX的直播 / 切片 / 切气口 / 剪掉废话 / 生成剪映草稿 / 我改好了草稿你看看
- 用户提供：本地视频文件（mp4/ts/其他）、剪映草稿名
- 用户要求：加字幕 / 标题 / 花字 / 调整式

## 项目环境（实测）

| 项 | 值 |
|---|---|
| 项目根 | `D:\myproj\doubao-live-clip` |
| Python | `.venv\Scripts\python.exe`（必须用它） |
| ffmpeg | `tools\ffmpeg\ffmpeg.exe`（完整版） |
| jy-draftc | `tools\jy-draftc.exe`（配 `tools\.env`，UTF-8 无 BOM） |
| 剪映草稿目录 | `C:\Users\Chinese\AppData\Local\JianyingPro\User Data\Projects\com.lveditor.draft` |
| ASR Key | 环境变量 `DASHSCOPE_API_KEY`（已配） |

## 工作流

### Step 1 探测与预处理

```powershell
cd D:\myproj\doubao-live-clip
.venv\Scripts\python.exe scripts\run_pipeline.py probe --video "<素材路径>"
.venv\Scripts\python.exe scripts\run_pipeline.py preprocess --video "<素材路径>"
# → outputs\work\audio.wav + silences.json
```

多素材：把产物隔离到子目录（如 `outputs\work\rongyiming\`），不要覆盖 outputs\work\ 根。

### Step 2 转写（词级时间戳）

```powershell
.venv\Scripts\python.exe scripts\run_pipeline.py transcribe --audio outputs\work\audio.wav
# → outputs\work\words.json（words[]: {word, start, end} 毫秒级）
```

ASR 选型（**踩坑记录，勿用错模型**）：
- ✅ `fun-asr-flash-2026-06-15` / `qwen-audio-3.0-asr-flash`（唯二有真实词级时间戳）
- ❌ `qwen3-asr-flash` 全系（无时间戳）；❌ Omni 系列（时间戳编造）；❌ words.probability（假字段）
- 额度各 36K 秒/月；config.json `asr.active` 从差到好自动 fallback

### Step 3 词→短语

```powershell
.venv\Scripts\python.exe scripts\run_pipeline.py merge --words outputs\work\words.json
# → outputs\work\phrases.json
```

### Step 4 判断层（你干活的地方）

读 `words.json`/`phrases.json` 的词级时间戳 + 静音信息，按**剪法规则**做决策：

1. **主题相关性 > 搞笑性**：与主题无关的观众互动/操作吐槽，再好笑也不留
2. **无语境长留白不留**：>5s 沉默删除（直播间冷场切片观众看不懂）
3. **原片音乐段是资产**：低音量 BGM 上转出的"歌词"是真实歌声非幻听，可留作结尾
4. **开场 3 秒上最炸的梗**：金句密集区直接开场
5. **结尾用音乐段收尾**（情绪升华）
6. **短句字幕**：4-10 字跟读式（不要整句长字幕）
7. **BGM 不用管**（用户自己加）；**标题每期必有**（主题标题贯穿全程）
8. 花字（可选）：`(吐槽)` 弹幕式，替观众说话

产出 EDL（**唯一标准 v2 格式**）：
```json
{"keep": [{"source_start", "source_end", "target_start", "duration", "video"?,
           "reason"?, "is_golden"?, "speaker_id"?, "dialogue_zoom"?}],
 "cut": [{"start", "end", "reason"?}], "target_duration": N, ...}
```
- `duration` 与 `source_end` 任给其一；`target_duration` 缺失自动重算
- `video`：多素材混排指定段素材，缺省用顶层 `source_path`
- 旧版 `{"segments": [...]}` 已废弃（仅可读入，不再产出）

**用判断表驱动时**：编辑 `scripts\judge_clips.py` 的 `P1_JUDGE`/`P2_JUDGE`：
```python
# (源起点, 源终点, keep?, reason, silent?)
(74.20, 81.80, True, "看AI手艺人", False),    # 保留
(29.91, 43.26, True, "留白冷场", True),        # 保留无字幕
(12.33, 14.21, False, "片头静默", False),      # 剪
```

### Step 5 生成草稿（推荐 CLI）

```powershell
.venv\Scripts\python.exe scripts\build_draft.py --edl "<merged_edl.json>" --srt "<merged.srt>" --ratio 4:3 --name "<草稿名>"
```

- `--ratio`：9:16 / 4:3 / 16:9（默认 4:3；字号自动映射 15/8/5）
- 自动完成：TS→MP4 转码、画布按比例、传承样式字幕、发光注入、多素材混排

### Step 6 交付与验证

1. 告诉用户**草稿名**（剪映里打开该名字）
2. 如需验证结构：读 `"<剪映草稿目录>\<草稿名>\draft_content.json"`（明文可读）检查画布/轨道/字幕数
3. 测试草稿用完即删；用户草稿勿动

## 读取用户精修成品（学习）

用户剪映保存后草稿**加密**（draft_content.json 二进制，头非 `{`）：

```powershell
& "D:\myproj\doubao-live-clip\tools\jy-draftc.exe" --dec "<草稿目录>\draft_content.json" "<输出>.json"
```

或代码：`from liveclip.jy_draftc import load_draft_content`。
分析成品可反哺：结构（段数/时长/顺序）、字幕（短句/字号/样式）、花字、BGM、标题、变速等 → 更新 styles.py 与剪法规则。

## 样式确定参数（styles.py 唯一真源）

| 参数 | 值 |
|---|---|
| 画布 | 默认 4:3（1920×1440） |
| 字幕字号 | 9:16→15 / 4:3→8 / 16:9→5 |
| 字体 | 后现代体（id 6740435494053614093） |
| 颜色 | 白字、居中 |
| 描边 | 红 #ff0023（GUI 0.0666 / pyJianYingDraft 传 33.3） |
| 阴影 | 红、alpha 0.3976 |
| 发光 | 轮廓光 bloom #d70000 / 0.55 / 0.66 |
| 标题 | 字号 12，与字幕同族 |
| 花字(可选) | 喜鹊古字典体 + 字号 8 + 清新粉色发光灯箱感花字（id 6896138122774531335） |

## 常见坑（全部踩过）

1. 剪映 6.0+ 草稿保存即加密 → 读前先 jy-draftc 解密（tools\.env 必须 UTF-8 无 BOM 写中文路径）
2. TS 素材先转 MP4（build_draft 自动）；ffmpeg 用 tools\ 完整版（系统版可能缺编码器）
3. 转写用词级时间戳模型（§工作流 Step 2），勿用 Omni/qwen3-asr
4. run_pipeline draft 走兼容层（样式正确）；新开发直接用 build_draft.py
5. 多素材产物隔离子目录；不要覆盖用户草稿

## 交付约定

- 回复用中文；不生成图片/视频内容（用户偏好）
- 完成时给出：草稿名、画布比例/时长、字幕数、保留/剪掉的段数概览
- 用户精修后回来：解密 → 对比分析 → 反哺规则，而不是直接覆盖他的草稿
