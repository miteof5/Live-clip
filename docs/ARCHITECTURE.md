# doubao-live-clip 架构设计

> 状态：设计定稿（v1）
> 目标：直播切片自动剪辑系统 —— 输入 5~60min 原始视频，输出 30s~10min "尽量完美的半成品"剪映草稿，用户剪映精修收尾。
> 判断层由豆包承担（不调外部 LLM API），工具层为本地 Python 项目（pyJianYingDraft 生成剪映草稿）。

---

## 1. 设计原则

1. **单入口，一条命令跑完**：`run_pipeline.py --video X --rule live_clip`，中间产物全部落盘可复查。
2. **分层清晰，判断与渲染分离**：
   - 判断层（rules/ + 豆包）→ 产出 EDL 决策（留/剪/标签/说话人）
   - 渲染层（draft_builder）→ 按 EDL 标签机械渲染草稿
   - 判断层与渲染层之间**只通过 EDL JSON 通信**，互不依赖。
3. **段标签驱动**：EDL 决策不只是"留/剪"，每段携带 `mode` + `effects` + `speakers` 等标签，渲染器按标签组合效果。新操作 = 新标签 + 新渲染逻辑，主流程不动。
4. **一切可配置**：模型、样式、剪法参数、镜头参数全进 `config.json`，换风格不改代码。
5. **判断层可替换**：说话人时间轴/金句判断可以是规则、豆包、或用户手动标注，接口统一。

---

## 2. 总体架构

```
                       ┌─────────────────────────────────────────┐
                       │             判断层 (decision)            │
                       │  rules/ 剪法规则集  +  豆包 (LLM 判断)    │
                       │  输入: 词级时间戳/短语/静音/画面信息       │
                       │  输出: EDL (编辑决策表, JSON)             │
                       └───────────────────┬─────────────────────┘
                                           │ EDL JSON（唯一接口）
                       ┌───────────────────▼─────────────────────┐
                       │             渲染层 (render)               │
                       │  draft_builder: 按 EDL 标签渲染           │
                       │  - 视频段截取 / 变速                       │
                       │  - 字幕导入 + 样式 (字号/描边/阴影/发光)    │
                       │  - 镜头效果 (缩放/位置关键帧)              │
                       │  - 转场 / 淡入淡出 / BGM 音量关键帧        │
                       └───────────────────┬─────────────────────┘
                                           ▼
                              剪映草稿 (draft_content.json)
                              用户剪映打开 → 精修 → 导出

输入视频 ──► probe ──► preprocess ──► transcribe ──► merge ──► (判断层) ──► (渲染层)
             探测     抽音频+查静音   ASR词级时间戳   词→短语     EDL 决策      落草稿
```

---

## 3. 目录结构与模块职责

```
doubao-live-clip/
├── liveclip/                    # 核心库（纯逻辑，无 argparse，可被 skill 直接 import）
│   ├── config.py                # 配置加载/合并/校验（config.json + 环境变量 + 参数覆盖）
│   ├── probe.py                 # 视频探测：时长/分辨率/帧率/码率/方向（横竖屏）
│   ├── preprocess.py            # 音频提取 + 静音段检测（Silence 列表）
│   ├── transcribe.py            # ASR 多后端（词级时间戳 words.json），失败自动 fallback
│   ├── merge_words.py           # 词级 → 短语（Phrase 列表）+ source.srt
│   ├── edl.py                   # ⭐ EDL 数据模型：校验/目标时间轴/源↔目标映射（已实现）
│   ├── rules/                   # ⭐ 剪法规则集（可插拔，判断层入口）
│   │   ├── base.py              #   规则接口：RuleContext → EDL
│   │   ├── minimal.py           #   极简：全保留+剪静音（当前 M1 用的）
│   │   └── live_clip.py         #   直播切片规则（M2）：金句/气口/废话/连麦段标签
│   ├── judge.py                 # ⭐ 豆包判断层适配器：读中间产物 → 调用豆包 → 回填 EDL 标签
│   ├── styles.py                # ⭐ 字幕样式（传承样式集中管理：字号规则/描边/阴影/发光）
│   ├── draft_builder.py         # ⭐ 草稿渲染器（合并旧 make_draft+build_draft，含发光/关键帧注入）
│   └── jy_draftc.py             # 剪映草稿解密/加密（本地 videoeditor.dll）
├── scripts/
│   └── run_pipeline.py          # ⭐ 唯一 CLI 入口（子命令：probe/transcribe/merge/edl/draft/all）
├── config.json                  # 模型/样式/剪法/镜头参数（见 §6）
├── docs/
│   ├── models-catalog.md        # 百炼模型实测清单
│   └── ARCHITECTURE.md          # 本文档
└── outputs/                     # 中间产物 + 草稿
    ├── work/                    # audio.wav / silences.json / words.json / phrases.json / source.srt / edl.json / mapped.srt
    └── drafts/                  # 生成的剪映草稿（或直写剪映草稿目录）
```

**重构要点**：
- 删除旧 `liveclip/make_draft.py` 的重复逻辑，草稿生成统一进 `draft_builder.py`
- 删除 `scripts/build_edl.py` / `scripts/build_draft.py` 的 CLI 形态，逻辑下沉为库函数，`run_pipeline.py` 统一调度
- 样式/发光常量从 build_draft.py 迁到 `styles.py`（配置化）

---

## 4. EDL 数据模型（扩展：段标签体系）

### 4.1 现状（M1 已实现）

```json
{
  "source_path": "...", "title": "live-clip",
  "width": 1920, "height": 1080, "source_duration_s": 249.68, "fps": 60,
  "segments": [
    {"kind": "keep", "source_start": 0.0, "source_end": 5.2, "reason": "...", "is_golden": false, "text": "..."},
    {"kind": "cut",  "source_start": 5.2, "source_end": 7.0, "reason": "气口"}
  ]
}
```

### 4.2 扩展（M2/M3 段标签）

```json
{
  "segments": [
    {
      "kind": "keep",
      "source_start": 12.3, "source_end": 20.5,
      "reason": "连麦搞笑段",
      "mode": "dialogue",                          // normal | dialogue | narrative | emphasis
      "effects": ["dialogue_zoom"],                 // 效果标签，可组合
      "speakers": [                                 // 说话人镜头切换点（判断层产出，可为空）
        {"t": 0.0, "who": "A", "scale": 1.8, "cx": 0.25, "cy": 0.5},
        {"t": 3.1, "who": "B", "scale": 1.8, "cx": 0.75, "cy": 0.5},
        {"t": 6.0, "who": "A", "scale": 1.8, "cx": 0.25, "cy": 0.5}
      ],
      "speed": 1.0,                                 // 变速（>1 快进）
      "is_golden": true,                            // 金句（未来花字/强调）
      "text": "..."
    },
    {
      "kind": "keep",
      "source_start": 45.0, "source_end": 65.0,
      "mode": "narrative",                          // 叙述段：无特效，只剪气口+字幕
      "effects": []
    }
  ]
}
```

**语义约定**：
- `mode`：段的叙事类型（决定默认处理方式）
- `effects`：显式效果标签，渲染器按标签逐个渲染，可组合（`["dialogue_zoom","subtitle_emphasis"]`）
- `speakers.t` 是**段内相对时间**（秒），渲染器换算到目标时间轴
- `cx/cy` 是画面归一化坐标（0~1），A 在左 (0.25)、B 在右 (0.75)
- 判断层只负责打标签；渲染层只负责按标签渲染 —— 新增复杂操作不改对方

---

## 5. 判断层设计

### 5.1 两级判断

| 级别 | 执行者 | 输入 → 输出 | 特点 |
|---|---|---|---|
| 规则级 | `rules/*.py`（确定性） | 短语/静音/时长 → 基础留剪 | 快、可复现、免费 |
| LLM 级 | 豆包（`judge.py` 适配器） | 中间产物 + 规则候选 → 标签/金句/说话人 | 语义理解、覆盖规则盲区 |

流程：**规则先粗剪（minimal 保证不丢内容）→ 豆包在候选上做精修（打标签/判金句/给说话人）**。豆包只在小范围内做判断，成本可控。

### 5.2 规则接口（rules/base.py）

```python
@dataclass
class RuleContext:
    phrases: List[Phrase]          # 词级合并后的短语（源时间轴）
    silences: List[Silence]        # 静音段
    words: List[Word]              # 词级时间戳
    probe: ProbeInfo               # 视频信息（宽高/时长/方向）
    config: Dict                   # 该规则的参数

class ClipRule(Protocol):
    name: str
    def decide(self, ctx: RuleContext) -> EDL: ...
```

### 5.3 豆包判断接口（judge.py）

```python
def refine_with_llm(edl: EDL, ctx: RuleContext, prompts: Dict) -> EDL:
    """豆包对规则产出的 EDL 做精修：
    1. 候选段筛选：只把需要判断的段（规则标记 judge_needed）发给豆包
    2. 豆包返回：mode / effects / speakers / is_golden / reason
    3. 回填 EDL，sanitize 时间戳（clamp 到段内），失败则保留规则结果（baseline）
    """
```

**豆包能力边界**（待验证项）：
- 说话人切换点：豆包听音频判断（需实测音频理解能力）
- A/B 画面位置：预设配置（连麦布局固定时一次配好），或豆包抽帧看图判断

---

## 6. 配置体系（config.json 扩展）

```json
{
  "asr": { /* 现状，不动 */ },

  "ffmpeg_path": "",
  "draft_folder": "",

  "silence_noise_db": -30.0,
  "silence_min_duration": 0.8,

  "styles": {
    "font": "后现代体",
    "landscape_size": 5.0,
    "portrait_size": 15.0,
    "color": "#ffffff",
    "border": {"color": "#ff0023", "width": 33.3, "alpha": 1.0},
    "shadow": {"color": "#fd0000", "alpha": 0.398, "diffuse": 0, "distance": 0, "angle": 0},
    "glow": {"effect_id": "9762325", "color": "#d70000", "strength": 0.55, "range": 0.66}
  },

  "rules": {
    "minimal": { "pause_keep_s": 0.8, "lead_s": 0.15 },
    "live_clip": {
      "pause_keep_s": 0.8,
      "min_keep_s": 1.0,
      "max_keep_s": 90.0,
      "dialogue_zoom": {
        "enabled": true,
        "scale": 1.8,
        "switch_mode": "smooth",        // hard | smooth
        "smooth_s": 0.2,
        "min_switch_gap_s": 0.5,        // 说话间隔太短不切换
        "speaker_positions": {          // 连麦布局预设（A 左 B 右）
          "A": {"cx": 0.25, "cy": 0.5},
          "B": {"cx": 0.75, "cy": 0.5}
        }
      },
      "golden": { "enabled": false }     // 金句花字（M3 开启）
    }
  },

  "judge": {
    "mode": "doubao",                    // rule | doubao | manual
    "max_judge_segments": 20             // 单次最多让豆包判断的候选段
  }
}
```

---

## 7. 渲染器设计（draft_builder.py）

### 7.1 核心函数

```python
def make_draft(edl: EDL, srt_path: str, video_path: str,
               styles: Styles, cfg: Dict, draft_folder: str,
               name: str) -> DraftResult:
    # 1. 基础：视频段截取（source_timerange）+ 字幕导入（import_srt + style_reference）
    # 2. 按段标签渲染：
    #    - dialogue_zoom  → 注入缩放/位置关键帧（speakers 切换点）
    #    - speed != 1.0   → VideoSegment(speed=...)
    #    - 转场/淡入淡出   → add_fade / transition
    # 3. 后处理注入（pyJianYingDraft 不原生支持的能力）：
    #    - 发光（inject_text_glow）
    #    - 复杂关键帧（缩放镜头切换的精细控制）
    # 4. save() + 校验（回读 JSON 验证段数/样式/关键帧）
```

### 7.2 关键帧注入（dialogue_zoom 渲染）

```python
# 对带 dialogue_zoom 标签的段：
#   段内每个 speakers 切换点 → 写 uniform_scale + position_x/y 关键帧
#   切换点之间：平滑过渡（smooth_s=0.2s）或硬切
#   关键帧值：scale ∈ [1.0, cfg.scale]，position 从 cx/cy 换算（像素或归一化）
# pyJianYingDraft: VideoSegment.common_keyframes 支持
#   KeyframeProperty.uniform_scale / position_x / position_y
```

---

## 8. 数据流与产物约定

| 阶段 | 输入 | 输出 | 校验点 |
|---|---|---|---|
| probe | 视频文件 | probe.json（时长/宽高/fps/方向） | 分辨率、时长>0 |
| preprocess | 视频 | audio.wav + silences.json | 音频可转写、静音列表单调 |
| transcribe | audio.wav | words.json（词级+毫秒时间戳） | 词数>0、时间戳单调 |
| merge | words.json | phrases.json + source.srt | 短语连续、无重叠 |
| edl（判断层） | phrases/silences + 规则 + 豆包 | edl.json（含标签） | validate() 通过 |
| draft（渲染层） | edl.json + mapped.srt + 视频 | 剪映草稿目录 | 回读 JSON 验证段数/样式/关键帧 |

**中间产物全部落盘 `outputs/work/`，用户可手改任何一步再续跑**（如手动标注说话人 → 重跑 draft）。

---

## 9. 演进路线

| 阶段 | 内容 | 状态 |
|---|---|---|
| **M1** | 词级转写 → 极简剪法（全保留+剪静音）→ 草稿生成 + 样式传承（字号/描边/阴影/发光） | ✅ 已闭环 |
| **M2** | 判断层接入：rules 可插拔 + 豆包精修（金句/废话/气口）→ 段标签体系 | 🔄 本文档定稿，下一步实现 |
| **M2.5** | `dialogue_zoom` 说话人镜头切换（说话人判断 + 缩放关键帧渲染）技术验证 | 待做 |
| **M3** | 复杂效果：金句花字、变速、转场、BGM 音量闪避、片头模板 | 按需 |

**skill 形态**：上述全部完成后，用户只需说"剪这个视频"，豆包读配置 → 跑 pipeline → 生成草稿 → 用户剪映精修。用户无需接触代码。
