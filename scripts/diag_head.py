import os, re, copy
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='1'
from pathlib import Path
import numpy as np, torch, timm, torch.nn as nn
from PIL import Image
ps=sorted(Path('data/zenodo_16745408/extracted/dataset ++').glob('*.jpg'))
y=np.array([int(re.match(r'\d+',p.name).group()) for p in ps],float)
X=torch.stack([torch.from_numpy(np.asarray(Image.open(p).convert('L').resize((224,224),Image.BICUBIC),dtype=np.uint8).copy()).unsqueeze(0).repeat(3,1,1) for p in ps])
idx=np.random.default_rng(0).permutation(len(ps)); te,tr=idx[:60],idx[60:]
mu,sd=y[tr].mean(),y[tr].std()
names=['convnext_tiny.fb_in1k','legacy_xception.tf_in1k','densenet121.tv_in1k','efficientnet_b0.ra_in1k','resnet50.tv_in1k']
torch.manual_seed(0)
for n in names:
    m=timm.create_model(n,pretrained=True,num_classes=1); h=m.get_classifier()
    c=m.pretrained_cfg; mean=torch.tensor(c['mean']).view(3,1,1); std=torch.tensor(c['std']).view(3,1,1)
    xb=(X[tr[:32]].float()/255-mean)/std; m.eval()
    with torch.no_grad(): o=m(xb)
    print(f'{n:28s} head weight std {h.weight.std().item():.4f} max {h.weight.abs().max().item():.3f} | start output std {o.std().item():.3f}')
# fix on EfficientNet only
dev='mps'; rng=np.random.default_rng(1); torch.manual_seed(0)
m=timm.create_model('efficientnet_b0.ra_in1k',pretrained=True,num_classes=1)
h=m.get_classifier(); nn.init.trunc_normal_(h.weight,std=.01); nn.init.zeros_(h.bias); m=m.to(dev)
c=m.pretrained_cfg; mean=torch.tensor(c['mean']).view(3,1,1); std=torch.tensor(c['std']).view(3,1,1)
prep=lambda i:((X[i].float()/255-mean)/std)
opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=1e-2)
for e in range(12):
    m.train(); L=[]; p=rng.permutation(tr)
    for k in range(0,len(p)-15,16):
        q=p[k:k+16]; l=nn.functional.l1_loss(m(prep(q).to(dev)).squeeze(1),torch.tensor((y[q]-mu)/sd,dtype=torch.float32,device=dev))
        opt.zero_grad();l.backward();opt.step();L.append(l.item())
    m.eval()
    with torch.no_grad(): pe=torch.cat([m(prep(te[k:k+32]).to(dev)).squeeze(1).cpu() for k in range(0,60,32)]).numpy()*sd+mu
    print('fixed ep',e+1,'train',round(np.mean(L)*sd,2),'test',round(np.abs(pe-y[te]).mean(),2),flush=True)
