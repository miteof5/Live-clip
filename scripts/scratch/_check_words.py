import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
d = json.load(open('outputs/work/words.json', encoding='utf-8'))
print('top-level type:', type(d).__name__)
if isinstance(d, dict):
    print('keys:', list(d.keys()))
    # try to find words list
    for k,v in d.items():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            print(f'list field "{k}": {len(v)} items')
            words = v
            break
    else:
        words = []
else:
    words = d
if words:
    starts = [w.get('start', w.get('st', 0)) for w in words]
    ends = [w.get('end', w.get('et', 0)) for w in words]
    print('word count:', len(words))
    print('first word:', words[0])
    print('last word:', words[-1])
    print('start range:', min(starts), '->', max(ends))
    # gaps > 5s between consecutive words
    big = []
    for i in range(1, len(words)):
        gap = starts[i] - ends[i-1]
        if gap > 5:
            big.append((ends[i-1], starts[i], gap))
    print('gaps > 5s:', len(big))
    for g in big[:20]:
        print(f'  gap {g[0]:.2f} -> {g[1]:.2f}  ({g[2]:.2f}s)')
