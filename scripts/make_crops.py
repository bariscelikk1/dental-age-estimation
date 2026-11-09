"""Yirmilik diş bölgesinin kör (yaşı gizlenmiş) kesitlerini hazırlar.
Her görüntüye rastgele bir kimlik verilir; kimlik-dosya eşlemesi annot/key.csv dosyasındadır
ve işaretleme bitene kadar açılmaz. Her kesit: sol yarı hastanın sağ alt (48), sağ yarı hastanın sol alt (38) bölgesi."""
import sys, numpy as np, pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp
OUT = exp.ROOT / 'annot'; (OUT / 'crops').mkdir(parents=True, exist_ok=True)
df = exp.load_df()
rng = np.random.default_rng(2209)
ids = rng.permutation(len(df))
key = pd.DataFrame(dict(id=[f'{i:03d}' for i in ids], name=df.name))
key.to_csv(OUT / 'key.csv', index=False)
for (_, r), i in zip(df.iterrows(), key.id):
    im = Image.open(r.path).convert('L'); W, H = im.size
    boxes = [(.08, .30, .40, .85), (.60, .30, .92, .85)]
    parts = [im.crop((int(a * W), int(b * H), int(c * W), int(d * H))) for a, b, c, d in boxes]
    parts = [p.resize((p.width * 300 // p.height, 300), Image.LANCZOS) for p in parts]
    sheet = Image.new('L', (parts[0].width + parts[1].width + 10, 300), 255)
    sheet.paste(parts[0], (0, 0)); sheet.paste(parts[1], (parts[0].width + 10, 0))
    sheet.save(OUT / 'crops' / f'{i}.png')
# 4 kesitlik sayfalar (okuma sayısını azaltmak için); sıra kimliğe göre, yaşla ilgisi yok
from PIL import ImageFont
order = sorted(key.id)
(OUT / 'sheets').mkdir(exist_ok=True)
for k in range(0, len(order), 4):
    ims = [Image.open(OUT / 'crops' / f'{i}.png') for i in order[k:k + 4]]
    w = max(im.width for im in ims)
    sheet = Image.new('L', (w + 70, 310 * len(ims)), 255); d = ImageDraw.Draw(sheet)
    for j, (i, im) in enumerate(zip(order[k:k + 4], ims)):
        sheet.paste(im, (70, j * 310)); d.text((8, j * 310 + 140), i, fill=0, font=ImageFont.load_default(size=28))
    sheet.save(OUT / 'sheets' / f'{k // 4:02d}.png')
print(len(key), 'kesit')
