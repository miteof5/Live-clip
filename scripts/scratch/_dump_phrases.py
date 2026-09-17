import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
d = json.load(open('outputs/work/phrases.json', encoding='utf-8'))
ph = d['phrases'] if isinstance(d, dict) and 'phrases' in d else d
print('count', len(ph))
print('keys sample:', list(ph[0].keys()) if ph else 'none')
for p in ph:
    s = p.get('start', p.get('source_start', 0))
    e = p.get('end', p.get('source_end', 0))
    t = (p.get('text') or '').strip()
    print(f'{s:7.2f} -> {e:7.2f}  {t}')
