import os, re, copy
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='1'
from pathlib import Path
import numpy as np, torch, timm, torch.nn as nn
from PIL import Image
ps=sorted(Path('data/zenodo_16745408/extracted/dataset ++').glob('*.jpg'))
y=np.array([int(re.match(r'\d+',p.name).group()) for p in ps],float)
X=torch.stack([torch.from_numpy(np.asarray(Image.open(p).convert('L').resize((224,224),Image.BICUBIC),dtype=np.uint8).copy()).unsqueeze(0).repeat(3,1,1) for p in ps])
rng0=np.random.default_rng(0); idx=rng0.permutation(len(ps)); te,tr=idx[:60],idx[60:]
mu,sd=y[tr].mean(),y[tr].std()
name='efficientnet_b0.ra_in1k'
base=timm.create_model(name,pretrained=True,num_classes=1)
c=base.pretrained_cfg; mean=torch.tensor(c['mean']).view(3,1,1); std=torch.tensor(c['std']).view(3,1,1)
prep=lambda i:((X[i].float()/255-mean)/std)
b=tr[:16]; xb=prep(b); yb=torch.tensor((y[b]-mu)/sd,dtype=torch.float32)
# 1) one forward/backward: CPU vs MPS
res={}
for dev in ['cpu','mps']:
    m=copy.deepcopy(base).to(dev); m.train()
    out=m(xb.to(dev)).squeeze(1); l=nn.functional.l1_loss(out,yb.to(dev)); l.backward()
    g=torch.cat([p.grad.flatten().cpu() for p in m.parameters() if p.grad is not None])
    res[dev]=(out.detach().cpu(),g)
    print(dev,'out std',round(out.std().item(),4),'loss',round(l.item(),4),'grad norm',round(g.norm().item(),4))
print('out diff',(res['cpu'][0]-res['mps'][0]).abs().max().item(),'grad cos',torch.nn.functional.cosine_similarity(res['cpu'][1],res['mps'][1],dim=0).item())
# 2) train short on CPU vs MPS, same settings, from same start
def run(dev,model=None,ep=4):
    torch.manual_seed(0); rng=np.random.default_rng(1)
    m=copy.deepcopy(base if model is None else model).to(dev)
    opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=1e-2)
    for e in range(ep):
        m.train(); L=[]; p=rng.permutation(tr)
        for k in range(0,len(p)-15,16):
            q=p[k:k+16]; l=nn.functional.l1_loss(m(prep(q).to(dev)).squeeze(1),torch.tensor((y[q]-mu)/sd,dtype=torch.float32,device=dev))
            opt.zero_grad();l.backward();opt.step();L.append(l.item())
        m.eval()
        with torch.no_grad(): pe=torch.cat([m(prep(te[k:k+32]).to(dev)).squeeze(1).cpu() for k in range(0,60,32)]).numpy()*sd+mu
        print(dev,'ep',e+1,'train MAE',round(np.mean(L)*sd,2),'test MAE',round(np.abs(pe-y[te]).mean(),2),flush=True)
run('cpu')
