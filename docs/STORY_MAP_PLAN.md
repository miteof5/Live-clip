# 全局叙事理解（Story Map）—— 判断层优化方案 v0.1

> 目标：把判断层从"逐句扫哪句好笑"（局部贪心）升级为"先通读全局、产出故事地图、在地图上选段"（全局规划），让切片带"故事感"。
> 核心改动：**不换架构、不改 EDL v2 唯一格式**，加一个本地零成本的"叙事简报合成器"+ 判断层先读全局再挑片的工作流动作。
> 状态：方案待评审。

---

## 1. 问题再诊断

现有链路：直播 → ASR 词级转写 → 逐句扫"哪句好笑" → EDL。

"文字稿"是对一场表演的**有损压缩**：语调节奏、表情反应、前因后果全丢了。人剪片带着完整上下文理解——"他俩为什么突然这样、这个铺垫在等哪个包袱、这一嗓子是被谁逗的、观众在笑什么"——逐句挑金句挑不出故事感。

**缺的不是数据，是"先读全局、再挑片"这个动作。** 判断层现在直接跳到挑句子，中间少了一步：先产出一张故事地图（这是个什么局 / 包袱链是哪句铺垫→哪句抖 / 情绪在哪几个点爆 / 哪里是冷场留白），然后在这张地图上选段。

## 2. 现有原料盘点（全部已实测/已实施，零新增采集）

| 信号 | 模块 | 产物 | 状态 |
|---|---|---|---|
| 词级转写 | transcribe.py | `words.json` / `phrases.json` | ✅ 词级真实时间戳 |
| 说话人分离 | speaker_diarization.py（paraformer-v2） | `speaker_timeline.json`（句级 speaker_id） | ✅ P2 实测（3 speaker，含 BGM 歌声分离） |
| 画面绑定 | speaker_binding.py | `bound_speaker_timeline.json`（speaker↔region） | ✅ 投票+基率校验 |
| 图文分镜 | storyboard.py | `visual_timeline.json`（幕级 layout/content/ui_text） | ✅ P1 15 幕 / P2 34 幕 / 9月15日 9 幕 |
| 音频事件 | audio_events.py | `audio_events.json`（burst/语速/静音前峰） | ✅ P2 冷场精确命中 |
| 候选段直听 | listen_probe.py | `listen_probes.json`（laughs/music/excitement） | ✅ qwen3.5-omni-flash 实测 |
| EDL 增强 | enrich_edl.py | EDL v2 带 speaker_id/dialogue_zoom/speaker_switches | ✅ 已实施 |

**结论：原料齐了。缺的是把它们合成"一次能通读的叙事简报"+ 判断层先读全局的动作。**

## 3. 方案总览

```
                ┌─────────────────────────────────────────────────┐
                │ 工具层（零成本，本地规则合成）                       │
                │  story_map.py：5 路信号时间对齐 → 幕级叙事简报       │
                │  story_map_raw.json（每幕：视觉+文字摘要+说话人+音频统计）│
                └──────────────┬──────────────────────────────────┘
                               ▼
                ┌─────────────────────────────────────────────────┐
                │ 判断层（豆包，不调外部 LLM）                        │
                │  Step 4.0 通读叙事简报                              │
                │  Step 4.1 产出故事地图（局/人物/包袱链/情绪曲线/留白）   │
                │  Step 4.2 在地图上选弧 → 弧内用剪法 8 条细剪          │
                │  Step 4.3 输出 EDL v2（keep 段 reason 标故事角色）    │
                └──────────────────────────────┬───────────────────┘
                                               ▼
                                          EDL v2 → 渲染层（不变）
```

两层只通过 **story_map_raw.json / story_map.json / EDL v2** 通信，互不依赖；任一层失败可回退（简报缺视觉→纯文字+音频；故事地图缺→回退散装输入逐句）。

## 4. 新产物 A：`story_map_raw.json`（叙事简报，工具层合成）

**合成器**：`liveclip/story_map.py`（新增，纯本地规则，无 API 调用）。

**时间对齐原则**：以 `visual_timeline.json` 幕为**骨架**（布局变化=段边界，已有）；词/短语、说话人段、音频事件、静音全部按时间挂载到幕上。无视觉产物时回退：以说话人切换点+静音+长停顿切幕。

```json
{
  "video": "P2.mp4", "duration": 194.8,
  "global": {
    "act_count": 34,
    "bgm_regions": [{"t": [140.3, 186.6], "evidence": "burst 稀疏+listen music=true"}],
    "long_silences": [{"t": [29.5, 31.6], "dur": 2.1}],
    "high_emotion_regions": [{"t": [95.0, 101.0], "evidence": "burst+listen excitement=1.0"}],
    "speakers": {"0": "占 82% 时长", "2": "BGM 歌声段"}
  },
  "acts": [
    {
      "id": 0, "t": [0.0, 12.3], "layout": "single",
      "visual": "主播单人画面，桌面录屏",              // 幕内 VL content 精选 ≤3 条
      "ui_text": ["加我粉丝团"],
      "text": "开场……（幕窗内 phrases 合并，去停顿词，≤200 字）",  // 关键：不能堆原始词
      "speakers": [0], "speaker_switches": 0,
      "audio": {"bursts": 1, "rate_jumps": 0, "pre_silence_peaks": 1, "longest_silence": 2.1},
      "emotion": 0.3                                    // 规则合成分 0-1
    }
  ]
}
```

**幕级 emotion 合成规则**（本地，可复算）：
`emotion = 0.4*burst密度归一 + 0.3*语速陡增密度归一 + 0.3*listen.excitement（无则 0）`，另：幕内存在 pre_silence_peak → 幕尾标记 `"cold_start": t`（反应后冷场）。

**成本**：零（纯本地）。**量级**：P2 34 幕 × 每幕 ~300 字 ≈ 10K token，强模型一次通读无压力。

## 5. 新产物 B：`story_map.json`（故事地图，判断层产出）

判断层通读简报后，**先回答四个问题再动手**，产出（可存盘供回看/审计/后续复用，也可只在判断内部使用）：

```json
{
  "story": {
    "setup": "主播在直播间展示 AI 生成的图片并逐张点评",
    "premise": "观众通过弹幕吐槽（'像蜘蛛侠''和本人一样'），主播找补",
    "persons": [{"id": 0, "role": "主播", "traits": "爱找补、语气夸张"}],
    "arc": "展示→吐槽→找补→冷场→金句收束"
  },
  "beats": [
    {"id": "b1", "t": [12.3, 32.1], "role": "setup", "label": "开场展示", "speakers": [0]},
    {"id": "b3", "t": [74.2, 81.8], "role": "punchline", "label": "看AI手艺人", "speakers": [0]},
    {"id": "b5", "t": [95.9, 102.9], "role": "reaction", "label": "漂亮/太舒服了", "speakers": [0]},
    {"id": "b6", "t": [103.0, 112.4], "role": "lull", "label": "大停顿", "speakers": []}
  ],
  "punchline_chain": [
    {"setup_beat": "b1", "punchline_beat": "b3", "at": 74.2, "why": "铺垫展示→抖包袱"}
  ],
  "emotion_curve": [
    {"t": [74.2, 81.8], "level": "high", "driver": "burst+语速+excitement"},
    {"t": [140.3, 186.6], "level": "low", "driver": "BGM 区"}
  ],
  "lulls": [{"t": [29.9, 43.3], "kind": "画面型冷场", "evidence": {"audio": "pre_silence_peak@29.5", "visual": "弹幕热议"}}],
  "recommended_arcs": [
    {"beats": ["b3", "b4", "b5"], "t": [74.2, 135.7], "why": "完整包袱链：铺垫→抖→反应"}
  ]
}
```

**关键区别**：现在的 keep 是"这句好笑所以留"；地图选段是"这是包袱链 b3→b5 的完整弧所以整段留，弧内再细剪"。

## 6. 判断层工作流改造（SKILL.md Step 4 重写为）

1. **Step 4.0 通读全局**：读 `story_map_raw.json`（有 story_map.json 则直接读故事地图）。无简报 → 先按 §4 规则在脑内做时间对齐再读，不得跳过"先全局后局部"。
2. **Step 4.1 先答四问**：这是什么局？谁对谁（连麦用 speaker_id + 布局）？包袱链哪句铺垫→哪句抖？情绪在哪爆/哪冷？→ 形成故事地图。
3. **Step 4.2 地图选弧**：圈 1-3 个叙事弧（recommended_arcs 或自选），弧内用剪法 8 条细剪（金句开场/留白判断/音乐收尾）。
4. **Step 4.3 输出 EDL v2**：keep 段 `reason` 标注故事角色（如 `"b3 包袱：铺垫展示→抖包袱"`）；`speaker_id`/`dialogue_zoom` 由 enrich_edl 照旧补。

**与剪法 8 条的关系**：不冲突。8 条是**弧内细剪规则**（怎么剪得干净）；故事地图是**弧级选段规则**（剪哪段）。先全局选弧、再局部细剪。

## 7. 验证方案（数据驱动，不拍脑袋）

| 验证 | 方法 | 指标 |
|---|---|---|
| 回归 P1/P2 | 用现有 judged 表作 ground truth，人工跑一遍"地图选段"流程 | (a) keep 集覆盖率；(b) 边界偏差 ±1s 内；(c) **新增质量**：选出段是否含完整包袱链（非孤立金句），P1/P2 各评 3-5 个弧 |
| 冷场/留白 | pre_silence_peak 命中（P2 已有 29.5/59.95 精确命中） | 地图 lulls 与 judged 留白段一致 |
| 连麦端到端 | 9月15日.mp4：`speaker_diarization --speakers 2` + story_map 合成 + 地图选段 | 对话弧（谁对谁）可回答；补 GT_MOMENTS 真实时刻表（当前 n/a） |
| 长视频 | >30min 素材按幕表粗筛候选段再补帧（storyboard 已有约定） | 简报幕数上限控制（如 >60 幕先按 emotion 合并相邻同类幕） |

## 8. 成本

- 叙事简报合成：**零成本**（本地规则）
- 故事地图：判断层豆包内部完成，**零新增 API**；如需存盘审计可选强模型一次通读（qwen3.5-omni-plus，~10K token，几分钱）
- 直听：按需触发（已有，一次几分钱）

## 9. 分阶段落地

| 阶段 | 内容 | 验收 |
|---|---|---|
| 一 | `liveclip/story_map.py` 合成器（时间对齐+幕挂载+emotion 规则）+ CLI + 单测 | P1/P2 产出 story_map_raw.json，字段齐 |
| 二 | SKILL.md Step 4 重写（先全局后局部）+ P1/P2 人工回归 | 地图选段 vs judged 差异报告（含包袱链质量评定） |
| 三 | 9月15日 连麦端到端（说话人分离跑通 + 对话弧 + GT 补全） | 连麦素材可回答"谁对谁" |
| 四（可选） | 强模型通读产 story_map.json 存盘 / 幕合并优化 | 长视频简报可控 |

## 10. 风险与边界

1. **幕太细**（P2 34 幕）：简报仍可控（~10K token）；>60 幕先按 emotion+speaker 合并相邻同类幕。
2. **speaker_id 是相对身份，不跨素材**：简报内局部一致即可，不跨视频对比。
3. **故事地图质量难自动量化**：用 judged 回归 + 用户精修回读反哺（既有闭环）。
4. **画面型冷场无音频证据**（P2 已实证）：地图 lulls 必须同时看 visual（弹幕热议）与 audio（pre_silence_peak），一路缺失不构成反证。
5. **不回退兼容性**：无视觉/无说话人产物时，简报降级为"文字+音频"或"纯文字"骨架，判断层按降级形态通读，不中断流水线。
