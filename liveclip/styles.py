# -*- coding: utf-8 -*-
"""
styles.py — 剪映草稿样式确定参数（用户已逐项确认）

来源：用户精修成品草稿 final_合订.draft_content.json（荣一鸣"考拉P图大赛"）
所有取值均从解密草稿直接读出，用户在对话中逐条确认。
"""

# ============ 画布 ============
CANVAS_RATIO = "4:3"            # 用户确认：固定 4:3
CANVAS_WIDTH = 1920             # 4:3 画布宽（高 1440）
CANVAS_HEIGHT = 1440

# ============ 普通字幕（弹幕字幕）============
# 字号规则（用户确认）：按画布比例映射
#   9:16 竖屏 -> 15 | 4:3 -> 8 | 16:9 -> 5
SUBTITLE_FONT_SIZE_BY_RATIO = {
    "9:16": 15.0,
    "4:3": 8.0,
    "16:9": 5.0,
}
DEFAULT_RATIO = "4:3"   # 用户主用比例

SUBTITLE_FONT_ID = 6740435494053614093           # 后现代体
SUBTITLE_FONT_NAME = "后现代体"
SUBTITLE_COLOR = (1.0, 1.0, 1.0)                 # 白字
SUBTITLE_ALIGN = 1                               # 居中

# 描边：红 #ff0023
SUBTITLE_BORDER_COLOR = (1.0, 0.0, 0.13725490868091583)
# 剪映 GUI 归一化宽度（用户确认值，来自成品标题样式）
SUBTITLE_BORDER_WIDTH_GUI = 0.0666
# pyJianYingDraft 写入草稿使用的宽度（33.3 与 GUI 0.0666 是否 1:500 换算待实渲染验证；
# 此前用 33.3 生成的字幕剪映显示正常，暂沿用）
SUBTITLE_BORDER_WIDTH = 33.3

# 阴影：红
SUBTITLE_SHADOW_COLOR = (0.9921568627450981, 0.0, 0.0)
SUBTITLE_SHADOW_ALPHA = 0.39760762453079224
SUBTITLE_SHADOW_DIFFUSE = 0
SUBTITLE_SHADOW_DISTANCE = 0
SUBTITLE_SHADOW_ANGLE = 0

# 发光：轮廓光 bloom
SUBTITLE_GLOW = {
    "effect_id": "9762325",
    "name": "轮廓光",
    "panel_id": "text_glow",
    "resource_id": "7202575978646737469",
    "path": "C:/Users/Chinese/AppData/Local/JianyingPro/User Data/Cache/effect/9762325/5e73b9655ea23c6ea73a491b59567c8e",
    "type": "bloom",
    "bloom_params": {
        "color": "#d70000",    # 红
        "dir_x": 0.5,
        "dir_y": 0.5,
        "range": 0.66,
        "strength": 0.55,
    },
}

# ============ 标题 ============
TITLE_FONT_SIZE = 12.0                  # 用户确认：标题字号 12
TITLE_STYLE_FAMILY = "与字幕同族"        # 后现代体 + 白字 + 红描边 + 红阴影
TITLE_FULL_SPAN = True                  # 标题全程贯穿（0 - 总时长）

# ============ 花字注释 ============
# 用户确认：花字为【可选】项（看情况加，非每期必有）
FLOWER_TEXT_ENABLED = True               # 总开关（由判断层按需开启）
FLOWER_TEXT_FONT_ID = 7035911487646339598          # 喜鹊古字典体(字节试用版)
FLOWER_TEXT_FONT_NAME = "喜鹊古字典体(字节试用版)"
FLOWER_TEXT_SIZE = 8.0                  # 与字幕一致
FLOWER_TEXT_COLOR = (1.0, 1.0, 1.0)     # 白
FLOWER_TEXT_PREFIX = "("                # 括号弹幕式
FLOWER_TEXT_SUFFIX = ")"
FLOWER_TEXT_EFFECT_ID = "6896138122774531335"      # 清新粉色发光灯箱感花字
FLOWER_TEXT_EFFECT_NAME = "清新粉色发光灯箱感花字"
FLOWER_TEXT_EFFECT_PATH = "C:/Users/Chinese/AppData/Local/JianyingPro/User Data/Cache/artistEffect/6896138122774531335/6f037828d751dee02763113f2db2cec2"

# ============ 剪辑风格（用户成品印证）============
# 纯硬切：无转场 / 无变速 / 无关键帧 / 无贴纸 / 无滤镜
NO_TRANSITIONS = True
NO_SPEED_CHANGE = True
NO_KEYFRAMES = True
NO_STICKERS = True
NO_FILTERS = True

# 结构规则（来自成品反推）
OPENING_HOOK = True                     # 开头 3 秒内上最炸的梗
MUSIC_ENDING = True                     # 结尾用原片音乐段收尾
NO_LONG_PAUSE = True                    # 无语境长留白（>5s 沉默）不留
THEME_RELEVANCE_FIRST = True            # 主题相关性 > 搞笑性
TITLE_EVERY_EPISODE = True              # 每期都要有主题标题（用户确认）


def subtitle_style(font_size=None, ratio=DEFAULT_RATIO):
    """构造 pyJianYingDraft 普通字幕样式对象（供 make_draft/build_draft 使用）

    font_size 优先；未给时按 ratio 查表（9:16->15 / 4:3->8 / 16:9->5）。
    """
    from pyJianYingDraft import draft
    if font_size is None:
        font_size = SUBTITLE_FONT_SIZE_BY_RATIO.get(ratio, SUBTITLE_FONT_SIZE_BY_RATIO[DEFAULT_RATIO])
    return draft.TextStyle(
        font_size=font_size,
        align=SUBTITLE_ALIGN,
        color=SUBTITLE_COLOR,
        border=draft.TextBorder(alpha=1.0, color=SUBTITLE_BORDER_COLOR,
                                width=SUBTITLE_BORDER_WIDTH),
        shadow=draft.TextShadow(alpha=SUBTITLE_SHADOW_ALPHA,
                                color=SUBTITLE_SHADOW_COLOR,
                                diffuse=SUBTITLE_SHADOW_DIFFUSE,
                                distance=SUBTITLE_SHADOW_DISTANCE,
                                angle=SUBTITLE_SHADOW_ANGLE),
    )
