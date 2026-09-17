"""judge_clips.py — 对 P1/P2 做人工判断层标注（留白/重复/幻听过滤）。

输入: outputs/work/rongyiming/{P1,P2}_words.json
输出: outputs/work/rongyiming/judged_{P1,P2}.json
    [ {keep, source_start, source_end, text, reason, words:[{start,end,word}]} ]

判断依据（2026-09-13，用户确认过素材内容）：
- 留白: P2 30-43s 是"奶蛙脸P到主播脸上"冷场，公屏刷笑死我了 → 保留
- 重复: P1 62-69s 与 12-19s 内容重复 → 剪
- 幻听: P2 64.88-183s 的"歌词"是低音量BGM上ASR幻听（能量-34~-39dB）→ 剪
- 碎片: 单个语气碎片词（"我。"、"我操多。"）→ 剪
- 保留: 有效台词 + 搞笑留白静默 + 情绪余韵
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "outputs" / "work" / "rongyiming"

# ---------- 判断区间表 ----------
# (start, end, keep, reason, silent?)  秒；silent=True 表示保留时段但无台词（留白/氛围，不出字幕）
P1_JUDGE = [
    (0.0, 12.33, False, "片头静默", False),
    (12.33, 14.21, True, "开场评价", False),
    (16.29, 19.29, True, "翻相册说明", False),
    (19.29, 29.51, False, "翻页操作间隙+碎片'我。'", False),
    (29.51, 32.27, True, "评价浅头发", False),
    (33.15, 34.11, True, "评价延续", False),
    (34.75, 41.15, True, "吐槽染发", False),
    (41.99, 43.31, True, "吐槽延续", False),
    (44.19, 46.07, True, "吐槽延续", False),
    (47.06, 50.38, True, "怕烫坏", False),
    (51.22, 54.10, True, "怕烫坏延续", False),
    (54.10, 62.07, False, "评价后停顿(留白待定，暂剪)", False),
    (62.07, 63.95, False, "重复:与12-14s重复", False),
    (65.99, 68.99, False, "重复:与16-19s重复", False),
    (69.0, 74.20, False, "翻页操作间隙", False),
    (74.20, 81.80, True, "看AI手艺人", False),
    (82.40, 83.56, True, "感叹", False),
    (84.46, 89.63, False, "感叹后短停顿(留1s待定)", False),
    (89.63, 92.19, True, "好没毛病", False),
    (93.0, 97.11, False, "看图反应停顿(留2s待定)", False),
    (97.11, 98.91, True, "漂亮感叹", False),
    (100.73, 102.93, True, "太舒服了", False),
    (103.0, 112.36, False, "大停顿(留3s待定)", False),
    (112.36, 118.00, True, "化妆发型", False),
    (119.20, 120.16, False, "碎片'我操多。'", False),
    (122.56, 125.08, True, "实话实说", False),
    (127.26, 127.98, True, "舒服", False),
    (129.06, 135.66, True, "姐姐嘉年华梗", False),
    (136.0, 142.02, False, "梗后余韵(留2-3s待定)", False),
    (142.02, 146.22, True, "结尾评价", False),
    (146.94, 149.70, True, "结尾延续", False),
]

P2_JUDGE = [
    (0.0, 7.78, False, "片头静默", False),
    (7.78, 9.10, True, "调音乐", False),
    (10.81, 11.81, True, "选歌", False),
    (12.0, 14.57, False, "选歌间隙(可留0.5s)", False),
    (14.57, 16.61, True, "铺垫", False),
    (17.45, 19.45, True, "铺垫延续", False),
    (19.45, 22.51, False, "语气停顿(可留1s)", False),
    (22.51, 23.31, True, "大差不差", False),
    (24.27, 28.07, True, "还原大差不差", False),
    (28.63, 29.91, True, "和我本人一样", False),
    (29.91, 43.26, True, "⭐留白13s:奶蛙脸P图冷场(公屏笑死我了)", True),
    (43.26, 52.90, False, "观众互动跑题:内定冠军话题与照片评价无关", False),
    (53.0, 54.74, False, "节奏停顿", False),
    (54.74, 60.46, True, "看小宝", False),
    (60.46, 69.68, True, "⭐留白9s:冷场延续", True),
    (69.68, 73.32, True, "有没有精品(真话,前面'等结局'是幻听已剪)", False),
    (73.32, 74.16, False, "短停顿", False),
    (74.16, 82.04, True, "这张挺漂亮", False),
    (83.12, 86.12, True, "头型漂亮", False),
    (88.73, 92.89, True, "完全本人", False),
    (93.0, 95.97, False, "低语间隙", False),
    (95.97, 97.61, True, "不是这张完全本人", False),
    (98.0, 105.05, False, "低语间隙", False),
    (105.05, 107.29, True, "主要眼睛", False),
    (107.29, 117.06, False, "静默3s+低语(留2s待定)", False),
    (117.06, 120.46, True, "大差不差没毛病", False),
    (122.26, 124.46, True, "这是啥呀", False),
    (125.38, 127.22, True, "咋这么奇怪", False),
    (127.22, 131.23, False, "低语间隙", False),
    (131.23, 135.51, True, "比例贼怪", False),
    (136.31, 140.27, True, "下巴收敛耳机线", False),
    (140.27, 186.63, False, "⭐幻听歌词区(故事远去...举杯饮醉意)非真实人声", False),
    (186.63, 187.15, False, "观众互动:谢谢与照片评价无关", False),
    (187.15, 191.47, False, "结尾停顿", False),
    (191.47, 193.99, False, "观众互动:评论失败吐槽与照片评价无关", False),
]


def build(words, judge):
    """按判断表给词打标，合并成台词段。silent=True 的保留段不含字幕词。"""
    segs = []
    for start, end, keep, reason, silent in judge:
        if silent:
            segs.append({
                "keep": True, "silent": True,
                "source_start": round(start, 2), "source_end": round(end, 2),
                "text": "", "reason": reason, "words": [],
            })
            continue
        seg_words = [w for w in words if w["start"] >= start - 1e-6 and w["end"] <= end + 1e-6]
        text = "".join(w["word"] for w in seg_words)
        segs.append({
            "keep": keep, "silent": False,
            "source_start": round(start, 2),
            "source_end": round(end, 2),
            "text": text,
            "reason": reason,
            "words": [{"start": round(w["start"], 2), "end": round(w["end"], 2), "word": w["word"]} for w in seg_words],
        })
    return segs


def main():
    for name in ("P1", "P2"):
        data = json.loads((OUT / f"{name}_words.json").read_text(encoding="utf-8"))
        judge = P1_JUDGE if name == "P1" else P2_JUDGE
        segs = build(data["words"], judge)
        keep_txt = sum(len(s["text"]) for s in segs if s["keep"])
        keep_s = sum(s["source_end"] - s["source_start"] for s in segs if s["keep"])
        out = OUT / f"judged_{name}.json"
        out.write_text(json.dumps(segs, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"===== {name}: 保留台词{keep_txt}字 / 保留时长{keep_s:.1f}s =====")
        for s in segs:
            tag = "KEEP" if s["keep"] else "CUT "
            print(f"  {tag} [{s['source_start']:6.2f}-{s['source_end']:6.2f}] {s['text'][:40]!r}  ({s['reason']})")


if __name__ == "__main__":
    main()
