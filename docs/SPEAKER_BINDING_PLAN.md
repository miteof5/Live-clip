# 说话人分离与画面绑定 —— 技术方案 v0.1

> 目标：让判断层能回答 "这句话是 A/B/C/D 谁说的"，为连麦场景 "谁说话放大谁"（dialogue_zoom）提供数据基础。
> 架构原则：
>
> **音频主（说话人分离）+ 视觉辅（位置绑定）**
>
> ，不新增常驻大模型、不装重识别模型。
> 状态：**v0.1 已实施（2026-09-16）**，P2 实测通过；连麦/PK 待真实素材验证。

---

## 0. 实施记录（2026-09-16）

| 阶段 | 模块 | 状态 | 备注 |
|---|---|---|---|
| 一 音频层 | `liveclip/speaker_diarization.py` | ✅ | P2 分离出 3 speaker（BGM 歌声单独成 speaker=2，见下） |
| 二 视觉层 | `liveclip/visual_signal.py` 加 region | ✅ | region_mode：lr 默认 / quad / auto |
| 三 融合层 | `liveclip/speaker_binding.py` | ✅ | 投票绑定 + 基率 + region_conflict |
| 四 判断层 | `liveclip/enrich_edl.py` + EDL 新字段 | ✅ | speaker_id / dialogue_zoom / speaker_switches |
| 五 连麦实测 | — | ⏳ 待素材 | P2 单人只验证了接口与机制 |

**P2 实测结论**：
- paraformer-v2 说话人分离**真实有效**：把 BGM 歌声（spk=2，全歌词）从荣一鸣人声（spk=0）中分出来
- 参数必须放 `parameters` 层级（放 `input` 不生效，speaker_id 不返回）
- dialogue_zoom 误标已修：切换两侧语音重叠 <0.3s 不算切换（冷场段+BGM 歌词不再误标）
- region_mode=auto 在误检下区域名漂移（left/lt 混用）→ 默认改 lr

**坑（已踩）**：
- 异步接口要公网 URL → 走百炼临时存储上传（getPolicy + OSS POST，48h 有效，已封装）
- OSS 文件名含中文/空格会 InvalidFile.DownloadFailed → 模块自动改 ASCII 文件名
- 说话人分离仅单声道 → 模块自动 ffmpeg `-ac 1` 转 16k mono

---

## 1. 背景与问题

现有链路（M1 粗剪闭环）只有 "谁在说话" 的文本，没有 "说话人是谁"。

画面里 2-4 人连麦 / PK 时，判断层无法回答：



* 这段台词是 A 说的还是 B 说的？

* 现在是 A 和 B 在对话，还是 B 和 C？

* 谁说话放大谁（镜头切换）的依据是什么？

纯视觉（OpenCV）只能跟踪 "第几张脸"，脸一转头 / 出画 track\_id 就断，无法稳定区分 A/B/C/D。

**结论：说话人归属本质是音频问题，视觉只负责 "画面里第几个位置是谁"。**



***

## 2. 总体架构



```mermaid
flowchart LR
    subgraph 素材
        V[视频文件 mp4]
    end

    subgraph 音频主
        V --> M1[ffmpeg 转单声道]
        M1 --> M2[Paraformer-v2 异步转写<br/>diarization_enabled=true]
        M2 --> M3[sentences[]<br/>begin/end/text/speaker_id/words]
    end

    subgraph 视觉辅
        V --> V1[visual_signal.py<br/>Haar人脸+跟踪+五维评分]
        V1 --> V2[face_events<br/>每采样帧 faces+区域]
    end

    subgraph 融合层
        M3 --> B1[speaker_binding.py<br/>时间对齐+投票绑定]
        V2 --> B1
        B1 --> B2[bound_speaker_timeline.json<br/>start/end/speaker/region/置信度]
    end

    subgraph 判断层
        B2 --> J1[judge_clips.py<br/>EDL 增加 speaker 标签]
        J1 --> J2[build_draft.py<br/>dialogue_zoom 预标记]
    end

    J2 --> D[剪映草稿]
```



***

## 3. 各层详细设计

### 3.1 音频层（主）—— 说话人分离

**模型**：`paraformer-v2`（百炼录音文件识别，异步任务接口）

**已查证能力**（官方文档）：



| 参数                            | 值          | 作用                                             |
| ----------------------------- | ---------- | ---------------------------------------------- |
| `diarization_enabled`         | `true`     | 每个 sentence 带 `speaker_id`（0/1/2/3… 相对身份，全程一致） |
| `timestamp_alignment_enabled` | `true`     | 词级时间戳 `words[]`（毫秒）—— 字幕链路可复用                  |
| `speaker_count`               | 2\~100（可选） | 提示算法 "有几个人"，连麦传 2、PK 传 4                       |
| `language_hints`              | `["zh"]`   | 中文优先                                           |

**预处理**：说话人分离仅支持单声道 → `ffmpeg -i in.mp4 -ac 1 mono.wav`

**输入限制**：异步接口要求公网 URL（不支持本地路径）

→ 方案：用 DashScope 文件上传能力取 URL（**待验证**），或本地起临时上传。

**输出**：



```
{ "begin\_time": 100, "end\_time": 3820, "text": "你好，我们今天讨论项目进度。",

&#x20; "speaker\_id": 0,

&#x20; "words": \[ { "begin\_time": 100, "end\_time": 596, "text": "你好" } ] }
```

**计费**：按语音内容时长计费（非语音不计费）；预计在用户 36K 额度内（**待实测确认**）。

### 3.2 视觉层（辅）—— 画面区域 + 主体

**已有**：`visual_signal.py`（Haar 多级联人脸检测 + IoU 跟踪 + 五维评分，已移植实测）。

**增量（本方案新增）**：分屏区域划分



* 双人连麦：画面按中分线划分 `left / right`

* 四人 PK：四象限 `tl / tr / bl / br`

* 单人：`whole`

* 每采样帧：每个 face 的框中心点 → 落区域；输出 `region` 字段



```
{ "t": 32.0, "n\_faces": 2,

&#x20; "subjects": \[

&#x20;   { "track\_id": 8, "region": "left",  "score": 0.55, "motion": 0.00 },

&#x20;   { "track\_id": 20, "region": "right", "score": 0.44, "motion": 0.12 } ],

&#x20; "dominant": { "track\_id": 8, "region": "left" } }
```

### 3.3 融合层（核心新模块 `speaker_binding.py`）

**输入**：speaker\_timeline（音频）+ face\_events（视觉）

**输出**：`bound_speaker_timeline.json`

**算法（三步）**：



1. **时间对齐**：每个 speech segment `[s, e]`，取窗口内所有视觉采样帧，统计：

* 各 region 出现次数、mean motion、mean size、dominant 次数

* 该 segment 的 **dominant region** = 综合得分最高区域

1. **投票绑定**：每个 speaker\_id 累计其全部 segment 的 dominant region 票 →

   **speaker → region 映射表**（置信度 = 该映射得票占比）。

* 例：speaker 0 说话时画面脸 87% 在 left → `speaker0 ↔ left`，置信度 0.87

1. **反向修正 + 基率校验**（借鉴 VideoHighlighter `signal_combinations.py`）：

* segment 内 dominant region 与映射表冲突 → 标记 `low_confidence`（切换瞬间 / 遮挡 / 抢话）

* 基率校验：若该视频 "大多数时间该 region 都有脸在动"，绑定置信度要打折 ——

  否则 "左边一直有人" 会被误当成 "speaker0 总在左边"

* **兜底**（借鉴 video-auto-edit-agent）：任何一步失败 → 该段标注 `speaker=null`，判断层按 "未知说话人" 处理，不中断流水线

**输出格式**：



```
{ "video": "...", "speakers\_total": 2,

&#x20; "binding\_map": { "0": {"region": "left", "confidence": 0.87},

&#x20;                  "1": {"region": "right", "confidence": 0.82} },

&#x20; "segments": \[

&#x20;   { "start": 10.2, "end": 14.8, "speaker\_id": 0, "text": "...",

&#x20;     "region": "left", "bind\_confidence": 0.87, "words": \[...] },

&#x20;   { "start": 15.1, "end": 18.0, "speaker\_id": null,

&#x20;     "region": null, "bind\_confidence": 0.0, "words": \[...] } ] }
```

### 3.4 判断层升级



* EDL 增加可选 `speaker_id` 字段：台词归属可见

* 连麦段标记 `dialogue_zoom` 候选：speaker 切换点 = 镜头切换候选点

* 判断 prompt 增加上下文：speaker 变化点 = 对话轮次切换



***

## 4. 借鉴点映射表



| 来源项目                                      | 借鉴内容                               | 用在                   | 状态    |
| ----------------------------------------- | ---------------------------------- | -------------------- | ----- |
| VideoHighlighter `signal_combinations.py` | **基率对比**：组合是否真的罕见（避免 "一直如此" 被当发现）  | 绑定置信度校验              | 本方案引入 |
| VideoHighlighter `SmartActionDetector`    | 五维主体评分公式                           | 视觉 dominant（已移植）     | ✅ 已实现 |
| VideoHighlighter `PersonTracker`          | IoU 跨帧跟踪                           | 视觉 track\_id（已移植）    | ✅ 已实现 |
| video-auto-edit-agent `vision_analyst`    | 结构化 JSON Schema 输出                 | speaker\_timeline 格式 | 本方案引入 |
| video-auto-edit-agent                     | `_sanitize` + baseline fallback 容错 | 融合层兜底                | 本方案引入 |
| video-auto-edit-agent `frame_extractor`   | 事件驱动抽帧参数                           | 说话人切换点抽帧复核（可选）       | 备用    |
| 百炼官方文档                                    | paraformer-v2 说话人分离参数              | 音频层                  | 本方案引入 |



***

## 5. 实施步骤（评审通过后）



| 阶段 | 内容                                                   | 验收标准                                |
| -- | ---------------------------------------------------- | ----------------------------------- |
| 一  | 音频层：接入 paraformer-v2（单声道预处理 + 异步任务 + speaker\_id 解析） | P2 跑通，返回 speaker\_id（预期单 speaker=0） |
| 二  | 视觉层：区域划分                                             | face\_events 带 region 字段            |
| 三  | 融合层：speaker\_binding.py（对齐 + 投票 + 基率校验 + 兜底）         | 双人素材 speaker↔region 映射正确            |
| 四  | 判断层：EDL 加 speaker\_id、dialogue\_zoom 预标记             | 连麦段标签正确                             |
| 五  | 实测：陈泽连麦素材（speaker\_count=2）                          | "谁说话放大谁" 半成品草稿                      |



***

## 6. 风险与待验证



1. **异步接口公网 URL**：paraformer-v2 要 file\_urls（公网）。DashScope 文件上传通道可行性待验证；不行则需 OSS 或其他中转（**首个 blocker，阶段一先测**）

2. **说话人分离鲁棒性**：直播音频压缩重、BGM 混响，speaker\_id 准确率待实测

3. **speaker\_id 是相对身份**：同一素材内一致，但**不跨素材**（每段素材重新编号）

4. **四人 PK**：speaker\_count=4 的分离准确率待实测；画面四象限绑定复杂度高

5. **额度**：paraformer-v2 是否计入用户 36K 额度，待实测确认

6. **抢话 / 重叠说话**：两人同时说话时分离易混，绑定置信度会低 —— 设计上允许 `speaker=null` 兜底