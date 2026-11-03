import sys, re
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np, pandas as pd
root = Path(sys.argv[1]); out = sys.argv[2]
ps = sorted(root.glob('*.jpg'))
X = np.stack([np.asarray(Image.open(p).convert('L').resize((96,48), Image.BICUBIC), float).ravel() for p in ps])
Z = (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)
C = Z @ Z.T / Z.shape[1]
iu = np.triu_indices(len(ps), 1)
v = C[iu]
print('pair corr pctiles', np.percentile(v, [50, 90, 99, 99.9]).round(3), 'max', v.max().round(4))
o = np.argsort(-v)[:25]
for k in o:
    i, j = iu[0][k], iu[1][k]
    print(round(v[k], 4), ps[i].name, '|', ps[j].name)
df = pd.DataFrame([dict(name=p.name, age=int(re.match(r'\d+', p.name).group()), w=Image.open(p).size[0], mode=Image.open(p).mode,
    rgb_really_gray=(lambda a: None if a.ndim==2 else bool(np.abs(a[...,0].astype(int)-a[...,1]).max()<12))(np.asarray(Image.open(p)))) for p in ps])
df['hires'] = df.w > 500
print(df.groupby('mode').age.describe().round(2)); print(df.groupby('hires').age.describe().round(2)); print(df.rgb_really_gray.value_counts(dropna=False))
print(pd.crosstab(df.age, df['mode'])); print(pd.crosstab(df.age, df.hires))
# contact sheet: 4 per age band + all pairs top-4
W,H=300,150; sel=[]
for a in [9,11,13,15,17,19]:
    sel += list(df[df.age==a].sample(4, random_state=0).index)
for k in o[:4]:
    sel += [iu[0][k], iu[1][k]]
cols=4; rows=int(np.ceil(len(sel)/cols)); sheet=Image.new('RGB',(cols*W,rows*(H+14)),'white'); d=ImageDraw.Draw(sheet)
for n,i in enumerate(sel):
    x,y=(n%cols)*W,(n//cols)*(H+14); sheet.paste(Image.open(ps[i]).convert('RGB').resize((W,H)),(x,y+14)); d.text((x+2,y+1),ps[i].name+' '+str(Image.open(ps[i]).size),fill='black')
sheet.save(out)
