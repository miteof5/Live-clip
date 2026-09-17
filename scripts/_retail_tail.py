"""补转 265s 之后的音频：抽片段 + 直接调 ASR。"""
import sys, io, json, subprocess, os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from liveclip.config import load_config, find_ffmpeg
from liveclip.transcribe import transcribe_audio

cfg = load_config()
ff = find_ffmpeg()
work = Path('outputs/work')
src = work / 'audio.wav'

# 抽 265s 到结尾
clip = work / 'tail.wav'
subprocess.run([ff, '-y', '-hide_banner', '-loglevel', 'error',
                '-ss', '265.0', '-i', str(src),
                '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', str(clip)],
               check=True)
print('tail clip size MB:', round(clip.stat().st_size/1e6, 2))

# 取第一个可用后端（bailian-fun-asr-flash）
backends = cfg['asr']['backends']
active = cfg['asr']['active']
be = next(b for b in backends if b['name'] == active[0])
print('using backend:', be['name'], be['model'])

tr = transcribe_audio(
    str(clip),
    api_key=be.get('api_key') or os.environ.get(be.get('api_key_env',''),''),
    model=be['model'], base_url=be['base_url'],
    language=be.get('language','zh'),
    chunk_max_seconds=float(be.get('max_seconds',300)),
    chunk_max_mb=float(be.get('max_mb',10)),
    work_dir=work/'chunks_tail',
    api_mode=be.get('api_mode','openai'),
)
used = be['name']
print('backend_used:', used)
print('tail words:', len(tr.words))
for w in tr.words:
    print(f'  {w.start+265.0:7.2f} -> {w.end+265.0:7.2f}  {w.word}')

# 保存合并后的完整 words
orig = json.load(open(work/'words.json', encoding='utf-8'))
# tail 词已 +265 偏移
tail_words = [{'word': w.word, 'start': round(w.start+265.0,3), 'end': round(w.end+265.0,3), 'probability': w.probability} for w in tr.words]
merged = orig['words'] + tail_words
merged.sort(key=lambda x: x['start'])
# 去重（若原 words 已含 265 附近）
merged_text = orig['text'] + tr.text
json.dump({'text': merged_text, 'language': 'zh', 'words': merged},
          open(work/'words_full.json','w',encoding='utf-8'), ensure_ascii=False, indent=2)
print('saved words_full.json total:', len(merged))
