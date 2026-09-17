import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
d = json.load(open('outputs/work/silences.json', encoding='utf-8'))
print('type:', type(d).__name__)
if isinstance(d, dict):
    print('keys:', list(d.keys()))
    sil = d.get('silences', d)
else:
    sil = d
print('count:', len(sil))
for s in sil:
    if isinstance(s, dict):
        print(f"  {s.get('start',0):7.2f} -> {s.get('end',0):7.2f}  dur={s.get('end',0)-s.get('start',0):.2f}")
    else:
        print(' ', s)
