"""两段事件粗剪 EDL（保留气口版）。

- 块内不剪气口（保留自然停顿）
- 只剪：纯冷场长静音 + 黑名单（找人口令/榴莲维密/拉票）
- 字幕按块映射到新时间轴
"""
import sys, io, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from scripts.build_edl import write_srt

phrases = json.load(open('outputs/work/phrases_full.json', encoding='utf-8'))

# 保留大块（源时间轴）：(source_start, source_end, label)
BLOCKS = [
    (7.0,   13.45,  "1a 开场"),
    (24.4,  43.3,   "1b 白门老铁"),
    (62.7,  167.0,  "1c 北冥有鱼→被提醒"),
    (173.0, 185.9,  "1d 两个都输/号严"),
    (329.6, 511.2,  "2a 金句密集区"),
    (512.0, 517.9,  "2b 你们废了"),
    (523.8, 530.5,  "2c 本命年收尾"),
]

# 构建 keep 段
keep = []
cursor = 0.0
for s, e, label in BLOCKS:
    dur = e - s
    keep.append({
        "source_start": round(s, 3),
        "source_end": round(e, 3),
        "target_start": round(cursor, 3),
        "duration": round(dur, 3),
        "_label": label,
    })
    cursor += dur

target_duration = round(cursor, 3)
print(f"keep blocks: {len(keep)}  target_duration: {target_duration:.1f}s ({target_duration/60:.2f} min)")
for k in keep:
    print(f"  {k['source_start']:7.2f}->{k['source_end']:7.2f}  dur={k['duration']:6.2f}  -> t={k['target_start']:7.2f}  {k['_label']}")

# 字幕映射：短语中心落在哪个块
def block_of(p):
    c = (p['start'] + p['end']) / 2
    for k in keep:
        if k['source_start'] - 0.5 <= c <= k['source_end'] + 0.5:
            return k
    return None

entries = []
for p in phrases:
    k = block_of(p)
    if k is None:
        continue
    offset = k['source_start'] - k['target_start']
    entries.append({
        "index": len(entries) + 1,
        "start": round(p['start'] - offset, 3),
        "end": round(p['end'] - offset, 3),
        "text": p['text'].strip(),
    })

# 清理内部字段
for k in keep:
    k.pop('_label', None)

edl = {"keep": keep, "target_duration": target_duration}
Path('outputs/work/edl_v2.json').write_text(json.dumps(edl, ensure_ascii=False, indent=2), encoding='utf-8')
write_srt(entries, Path('outputs/work/mapped_v2.srt'))
print(f"\nEDL -> outputs/work/edl_v2.json")
print(f"SRT -> outputs/work/mapped_v2.srt  ({len(entries)} 条)")
