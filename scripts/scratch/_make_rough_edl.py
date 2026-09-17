"""粗剪 EDL 生成：自动剪停顿 + 手动排除无意义大段。

判断层（豆包）粗剪规则：
- 用 build_edl.build() 自动合并短语窗口、剪 >0.8s 停顿
- 手动排除 43.2~62.8：找在线人/喊"玻璃樽空半子"的操作口令，无内容
- 其余保留主体对话，由 build_draft 落草稿；精剪留给用户
"""
import sys, io, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from scripts.build_edl import build, map_srt, write_srt

phrases = json.load(open('outputs/work/phrases_full.json', encoding='utf-8'))
DURATION = 595.709

# 排除区间（粗剪判断）：(start, end, reason)
BLACKOUT = [
    (43.2, 62.8, "找在线人/喊人操作口令，无内容"),
]

def in_blackout(s, e):
    for bs, be, _ in BLACKOUT:
        # 短语中心落在黑名单内则剔除
        c = (s + e) / 2
        if bs <= c <= be:
            return True
    return False

kept_phrases = [p for p in phrases if not in_blackout(p['start'], p['end'])]
print(f'phrases: total={len(phrases)} kept={len(kept_phrases)} removed={len(phrases)-len(kept_phrases)}')

edl = build(kept_phrases, DURATION, pause_keep=0.8, lead=0.15)
entries = map_srt(kept_phrases, edl['keep'])

# 打印 keep 段概览
print(f'\nkeep segments: {len(edl["keep"])}  target_duration: {edl["target_duration"]:.1f}s ({edl["target_duration"]/60:.2f} min)')
print('被剪掉的大段:')
for bs, be, reason in BLACKOUT:
    print(f'  {bs:6.1f} -> {be:6.1f}  ({be-bs:.1f}s) {reason}')

out_edl = Path('outputs/work/edl_cut.json')
out_srt = Path('outputs/work/mapped_cut.srt')
out_edl.write_text(json.dumps(edl, ensure_ascii=False, indent=2), encoding='utf-8')
write_srt(entries, out_srt)
print(f'\nEDL  -> {out_edl}')
print(f'SRT  -> {out_srt}  ({len(entries)} 条)')
