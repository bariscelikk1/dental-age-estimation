import sys, re, hashlib, collections
from pathlib import Path
from PIL import Image
import numpy as np
root = Path(sys.argv[1])
rows = []
for p in sorted(root.glob('*.jpg')):
    m = re.match(r'^(\d+)', p.name)
    im = Image.open(p)
    rows.append(dict(name=p.name, age=int(m.group(1)) if m else None, size=im.size, mode=im.mode, kb=p.stat().st_size//1024,
                     md5=hashlib.md5(p.read_bytes()).hexdigest(),
                     phash=hashlib.md5(np.asarray(im.convert('L').resize((32,32))).__rshift__(4).tobytes()).hexdigest()))
print('n', len(rows))
print('ages', sorted(collections.Counter(r['age'] for r in rows).items()))
print('sizes', collections.Counter(r['size'] for r in rows).most_common(12))
print('modes', collections.Counter(r['mode'] for r in rows))
print('kb max', sorted((r['kb'], r['name']) for r in rows)[-8:])
for key in ('md5','phash'):
    g = collections.defaultdict(list)
    for r in rows: g[r[key]].append(r['name'])
    print(key, 'dup groups', [v for v in g.values() if len(v)>1])
print('odd names', [r['name'] for r in rows if not re.match(r'^\d+( +\d+)?\.jpg$', r['name'])])
