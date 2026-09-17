# AGENTS.md — 给 Agent 的操作手册

> 本文件面向**任何 AI Agent**（豆包/Claude/其他），说明如何操作本"直播切片自动剪辑工具"。
> 读完本文件即可独立完成：转写 → 判断 → 生成剪映草稿 → 读取用户精修成品。
> 所有命令均已在本机实测。最后更新：2026-09-15。

---

## 0. 这是什么

输入 **5-60min 直播原始视频**（横屏/竖屏均可）→ 输出**剪映草稿**（半成品：已剪气口/去废话/带样式字幕），用户用剪映打开精修收尾。

**架构铁律（勿破坏）**：
- **判断层（豆包/人工）与渲染层（draft_builder）只通过 EDL JSON 通信**
- 样式只改 `liveclip/styles.py` 一处
- 所有草稿生成入口共用 `liveclip/draft_builder.py` 唯一核心

## 1. 环境（本机实测）

| 项 | 路径/值 |
|---|---|
| Python | `D:\myproj\doubao-live-clip\.venv\Scripts\python.exe`（必须用它） |
| ffmpeg | `D:\myproj\doubao-live-clip\tools\ffmpeg\ffmpeg.exe`（完整版 9.0.1，有 libx264/libmp3lame） |
| jy-draftc（解密） | `D:\myproj\doubao-live-clip\tools\jy-draftc.exe` |
| 剪映草稿目录 | `C:\Users\Chinese\AppData\Local\JianyingPro\User Data\Projects\com.lveditor.draft` |
| 剪映版本 | `C:\MyApp\剪映JianyingPro 免V1P\6.0.1.11779`（唯一保留版，AppData 官方版已删） |
| ASR Key | 环境变量 `DASHSCOPE_API_KEY`（已配好，别问用户要） |

**tools\.env 内容（UTF-8 无 BOM，勿用 ASCII 重写否则中文变 ???）**：
```
JY_INSTALL_DIR=C:\MyApp\剪映JianyingPro 免V1P\6.0.1.11779
```

## 2. 标准工作流（5 步）

```powershell
cd D:\myproj\doubao-live-clip

# 1. 探测视频
.venv\Scripts\python.exe scripts\run_pipeline.py probe --video <视频路径>
# 2. 预处理：抽音频 + 静音检测
.venv\Scripts\python.exe scripts\run_pipeline.py preprocess --video <视频路径>
#    → outputs\work\audio.wav + silences.json
# 3. 转写：词级时间戳（自动 fallback，见 §4）
.venv\Scripts\python.exe scripts\run_pipeline.py transcribe --audio outputs\work\audio.wav
#    → outputs\work\words.json
# 4. 词→短语合并
.venv\Scripts\python.exe scripts\run_pipeline.py merge --words outputs\work\words.json
#    → outputs\work\phrases.json
# 5. （判断层）产出 EDL → 生成草稿（见 §3）
```

> 注意：`preprocess`/`transcribe` 的单素材工作流会覆盖 `outputs\work\` 下的同名文件。**多素材**（如 P1/P2）请把产物放到独立子目录（参考 `outputs\work\rongyiming\`：P1_words.json / P2_words.json …）。

## 3. 判断层 + 生成草稿（两种路径）

### 路径 A：判断表驱动（推荐，可精确控制）

1. 打开 `scripts\judge_clips.py`，编辑 `P1_JUDGE` / `P2_JUDGE` 判断表：
   ```python
   # (源起点, 源终点, keep?, reason, silent?)
   (74.20, 81.80, True, "看AI手艺人", False),     # 保留
   (29.91, 43.26, True, "⭐留白:冷场", True),      # 保留但无字幕（silent=True）
   (12.33, 14.21, False, "片头静默", False),       # 剪掉
   ```
2. 生成判断结果 + EDL + 字幕：
   ```powershell
   .venv\Scripts\python.exe scripts\judge_clips.py
   .venv\Scripts\python.exe -c "import json; ..."  # 由 judged 清单组装 EDL + mapped.srt
   ```
3. 生成草稿（推荐用 CLI，支持多素材混排）：
   ```powershell
   .venv\Scripts\python.exe scripts\build_draft.py --edl <merged_edl.json> --srt <merged.srt> --ratio 4:3 --name "<草稿名>"
   ```

### 路径 B：自动剪静音（无判断，先跑通用）

```powershell
.venv\Scripts\python.exe scripts\build_edl.py --phrases outputs\work\phrases.json --duration <原视频秒数>
# → outputs\work\edl.json + mapped.srt（按静音自动剪）
.venv\Scripts\python.exe scripts\build_draft.py --edl outputs\work\edl.json --srt outputs\work\mapped.srt --video <原视频> --name "<草稿名>"
```

### EDL 唯一标准格式（v2，勿再产出旧版）

- **唯一标准**：`{"keep": [{"source_start","source_end","target_start","duration","video"?}], "cut": [...], "target_duration": N}`
- 旧版 `{"source_path","segments":[{"kind":"keep|cut"}]}` **已废弃**：只能被自动识别读入，不再产出
- 说话人字段由 `python -m liveclip.enrich_edl --edl <v2 edl>.json --speaker bound_speaker_timeline.json` 填充

## 4. ASR 选型（重要，踩坑记录）

**只有这 2 个模型返回真实词级时间戳（sentence.words[] 毫秒级）**：
- `fun-asr-flash-2026-06-15`（差→好的第 1 档）
- `qwen-audio-3.0-asr-flash`（主力）

**不可用**：
- `qwen3-asr-flash` 全系 → 返回纯文本无时间戳 ❌
- Omni 系列（17 个）→ 时间戳每字 0.1s 编造 ❌
- fun-asr 的 `words[].probability` 全是 1.0（假字段），置信度过滤不可用 ❌

额度：每个 36K 秒/月 ≈ 10h 音频。`config.json` 的 `asr.active` = 从差到好的自动 fallback 顺序；`transcribe` 输出 `backend_used` 标识实际使用。

## 5. 剪法规则（用户成品反推，判断层核心）

1. **主题相关性 > 搞笑性**：与主题无关的观众互动/操作吐槽，再好笑也不留
2. **无语境长留白不留**：>5s 的沉默删除（直播间冷场切片观众看不懂）
3. **原片音乐段是资产**：低音量 BGM 上 ASR 转出的"歌词"是真实歌声，不是幻听，可留作结尾
4. **开场 3 秒上最炸的梗**（金句密集区直接开场）
5. **结尾用音乐段收尾**（情绪升华）+ 可选 1s 收尾素材
6. **短句字幕**：4-10 字跟读式（不要整句长字幕）
7. **BGM 不用管**：用户自己加
8. **标题每期必有**：主题标题全程贯穿（如"考拉P图大赛"）
9. 花字注释（可选）：`(吐槽)` 弹幕式，替观众说话

## 6. 样式确定参数（liveclip/styles.py，唯一真源）

| 参数 | 值 |
|---|---|
| 画布 | 默认 **4:3**（1920×1440）；可选 9:16 / 16:9 |
| 字幕字号 | **9:16→15 / 4:3→8 / 16:9→5**（按画布比例映射） |
| 字体 | 后现代体（id 6740435494053614093） |
| 颜色 | 白字、居中 |
| 描边 | 红 #ff0023（GUI 0.0666 / pyJianYingDraft 传 33.3 自动归一化） |
| 阴影 | 红、alpha 0.3976 |
| 发光 | 轮廓光 bloom：#d70000 / strength 0.55 / range 0.66 |
| 标题 | 字号 12，与字幕同族 |
| 花字(可选) | 喜鹊古字典体 + 字号 8 + 清新粉色发光灯箱感花字（id 6896138122774531335） |

## 7. 读取用户精修成品（学习样式/剪法）

剪映 6.0+ 保存草稿后 `draft_content.json` **加密**（明文 JSON 读不了）：

```powershell
# 解密（需 tools\.env 的 JY_INSTALL_DIR 正确，UTF-8 无 BOM）
& "D:\myproj\doubao-live-clip\tools\jy-draftc.exe" --dec "<草稿目录>\draft_content.json" "<输出路径>.json"
```

或代码：`from liveclip.jy_draftc import load_draft_content`。

## 8. 常见坑（全部踩过）

1. **草稿加密**：剪映保存后 draft_content.json 是二进制（`dhg1Yzt1...` 开头），必须先 jy-draftc 解密；判断是否加密看文件头是否以 `{` 开头
2. **TS 素材**：必须先转码 MP4（build_draft 自动处理，copy 失败会重编码）
3. **ffmpeg 需完整版**：系统 PATH 的 ffmpeg 可能缺编码器，用 `tools\ffmpeg\`（有 libx264/libx265/libmp3lame）
4. **tools\.env 编码**：必须 UTF-8 无 BOM 写中文路径，ASCII 会变 `???` 导致 jy-draftc 报 `JY_INSTALL_DIR does not exist`
5. **run_pipeline draft**：走 make_draft 兼容层（已委托 draft_builder，样式正确），新开发请直接用 `scripts\build_draft.py`
6. **草稿清理**：验证用草稿用完即删（用户草稿如"切片测试-全保留"勿动）
7. **多素材产物隔离**：不要覆盖 outputs\work\ 根目录文件，多素材用子目录

## 9. 目录地图

```
doubao-live-clip/
├── liveclip/            # 核心库（唯一逻辑）
│   ├── config.py        #   配置加载（config.json）+ find_ffmpeg 唯一实现
│   ├── probe.py         #   视频探测
│   ├── preprocess.py    #   抽音频+静音检测
│   ├── transcribe.py    #   ASR 多后端（词级时间戳）
│   ├── merge_words.py   #   词→短语（含 SRT 时间戳格式化唯一实现 _fmt_ts）
│   ├── edl.py           #   ⭐ EDL 统一模型（NormalizedEDL，v2 keep 格式唯一标准）
│   ├── draft_builder.py #   ⭐ 草稿生成唯一核心
│   ├── make_draft.py    #   兼容层（统一委托 draft_builder + NormalizedEDL）
│   ├── jy_draftc.py     #   草稿解密封装（自动探测 tools\jy-draftc.exe + tools\.env）
│   ├── styles.py        #   ⭐ 确定参数唯一真源
│   ├── visual_signal.py        #   视觉信号层：人脸+跟踪+五维评分+区域（opencv）
│   ├── speaker_diarization.py  #   说话人分离（paraformer-v2 异步）
│   ├── speaker_binding.py      #   融合层：说话人↔画面位置绑定
│   └── enrich_edl.py           #   EDL 说话人富化（speaker_id/dialogue_zoom）
├── scripts/             # CLI 入口（正式）
│   ├── run_pipeline.py  #   主入口（probe/preprocess/transcribe/merge/draft）
│   ├── build_edl.py     #   自动剪静音 → v2 EDL + mapped.srt
│   ├── judge_clips.py   #   判断表驱动（编辑 P1_JUDGE/P2_JUDGE）
│   └── build_draft.py   #   ⭐ EDL+SRT → 草稿（推荐）
├── scripts/scratch/     # 历史一次性调试脚本（_*.py，仅供回溯，勿当正式入口）
├── tools/
│   ├── ffmpeg/          #   完整版 ffmpeg + ffprobe（不入 git，本地保留）
│   ├── jy-draftc.exe    #   剪映草稿解密/加密（不入 git，本地保留）
│   └── .env             #   JY_INSTALL_DIR（UTF-8 无 BOM，jy_draftc 自动读取）
├── outputs/
│   ├── work/            #   中间产物（words/phrases/judged/edl/srt）
│   └── decrypted/       #   解密后的成品草稿 JSON
├── tests/               # pytest（tests\test_*.py，运行：.venv\Scripts\python.exe -m pytest tests\ -q）
├── config.json          #   ASR 后端/阈值/路径配置
└── docs/ARCHITECTURE.md #   架构设计稿
```

## 10. 说话人维度（音频主 + 视觉辅，2026-09-16 新增）

让判断层回答"这句话是 A/B/C/D 谁说的"。**三层**：

```powershell
# ① 音频主：说话人分离（百炼 paraformer-v2 异步任务）
.venv\Scripts\python.exe -m liveclip.speaker_diarization --video <素材.mp4> [--speakers 2]
#   → outputs\work\speaker_timeline.json
#   （speakers：连麦传 2 / PK 传 4；不传自动判断）

# ② 视觉辅：画面主体 + 分屏区域（Haar 人脸 + IoU 跟踪 + 五维评分）
.venv\Scripts\python.exe liveclip\visual_signal.py --video <素材.mp4> --region-mode lr
#   → outputs\work\visual_events.json
#   （region-mode：lr=左右连麦默认 / quad=四象限PK / auto=按人数）

# ③ 融合：说话人 ↔ 画面位置绑定
.venv\Scripts\python.exe -m liveclip.speaker_binding --audio speaker_timeline.json --visual visual_events.json
#   → outputs\work\bound_speaker_timeline.json

# ④ 富化：EDL/keep 列表补 speaker_id + dialogue_zoom（判断层输入）
.venv\Scripts\python.exe -m liveclip.enrich_edl --edl <judged|edl>.json --speaker bound_speaker_timeline.json
```

**数据流**：speaker_timeline（谁在说，词级时间戳）⊕ visual_events（脸在哪个区域）
→ bound（speaker↔region 映射 + 基率 + 冲突标记）→ EDL 的 `speaker_id` / `dialogue_zoom`。

**已实现**（P2 实测）：
- diarization 能把 **BGM 歌声从人声中分离**（P2 中 spk=2 全是歌词——"资产"判断可用）
- speaker_id 全程一致（相对身份，不跨素材）
- dialogue_zoom 只标真实切换（切换两侧语音 ≥0.3s 才计，冷场+BGM 歌词不误标）

**待验证**（需连麦素材）：双人连麦 speaker↔左右区域绑定准确率；四人 PK。

**坑**：
- diarization 仅支持**单声道**（模块自动转）；异步任务需**公网 URL**（模块自动走百炼临时存储上传，48h 有效）
- 参数必须放 `parameters` 层级（放 input 不生效，speaker_id 不返回）
- speaker_id 是相对身份：同一素材内一致，**不跨素材对应**

## 11. 与用户协作约定

- 用户是"导演"：他决定剪什么/留什么；你负责转写、判断辅助、机械执行
- 生成草稿后告诉用户**草稿名**（剪映里打开该名字）
- 用户精修后草稿会加密——想学习样式就解密分析，但**不要覆盖用户的草稿**
- 用户偏好：不要生成图片/视频内容；回复用中文
