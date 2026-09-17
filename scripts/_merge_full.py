"""用全量 words 重新合并短语并打印时间轴。"""
import sys, io, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from liveclip.merge_words import merge_to_phrases
from liveclip.transcribe import Word
from liveclip.config import load_config  # noqa

d = json.load(open('outputs/work/words_full.json', encoding='utf-8'))
words = [Word(word=w['word'], start=w['start'], end=w['end'], probability=w.get('probability',1.0)) for w in d['words']]
words.sort(key=lambda w: w.start)
phr = merge_to_phrases(words, max_gap=0.3, max_chars=24)
print('phrase count:', len(phr))
for p in phr:
    print(f'{p.start:7.2f} -> {p.end:7.2f}  {p.text.strip()}')
# save
out = [p.to_dict() for p in phr]
json.dump(out, open('outputs/work/phrases_full.json','w',encoding='utf-8'), ensure_ascii=False, indent=2)
