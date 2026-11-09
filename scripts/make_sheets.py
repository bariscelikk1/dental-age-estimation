"""Kör evreleme için sayfalar: her görüntünün diş bölgesi (yaşı gizli, rastgele kimlikle), sayfa başına 3 görüntü.
Sıra rastgele kimliğe göredir. İkinci geçiş için farklı sırayla ayrı sayfalar üretilir."""
import sys, numpy as np, pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
ROOT = Path(__file__).resolve().parents[1]
key = pd.read_csv(ROOT / 'annot/key.csv', dtype={'id': str})
src = ROOT / 'data/zenodo_16745408/extracted/dataset ++'
font = ImageFont.load_default(size=26)
for pas, seed in [(1, None), (2, 7)]:
    order = sorted(key.id) if seed is None else list(np.random.default_rng(seed).permutation(sorted(key.id)))
    out = ROOT / f'annot/pass{pas}'; out.mkdir(parents=True, exist_ok=True)
    name = dict(zip(key.id, key.name))
    for k in range(0, len(order), 3):
        crops = []
        for i in order[k:k + 3]:
            im = Image.open(src / name[i]).convert('L'); W, H = im.size
            im = im.crop((int(.04 * W), int(.22 * H), int(.96 * W), int(.92 * H)))
            crops.append((i, im.resize((800, round(800 * im.height / im.width)), Image.LANCZOS)))
        h = sum(c.height + 8 for _, c in crops)
        sheet = Image.new('L', (870, h), 255); d = ImageDraw.Draw(sheet); y = 0
        for i, c in crops:
            sheet.paste(c, (70, y)); d.text((6, y + c.height // 2 - 13), i, fill=0, font=font); y += c.height + 8
        sheet.save(out / f'{k // 3:03d}.png')
print('tamam')
