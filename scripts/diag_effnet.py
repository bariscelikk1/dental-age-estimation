import os, json, re, sys
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='1'
from pathlib import Path
import numpy as np, torch, timm, torch.nn as nn
from PIL import Image
h=json.load(open('results/efficientnet_b0_ra_in1k_fold0.json'))['hist']
print('saved hist train/val', [(r['epoch'],round(r['train_mae'],2),round(r['val_mae'],2)) for r in h[:6]])
ps=sorted(Path('data/zenodo_16745408/extracted/dataset ++').glob('*.jpg'))
y=np.array([int(re.match(r'\d+',p.name).group()) for p in ps],float)
X=torch.stack([torch.from_numpy(np.asarray(Image.open(p).convert('L').resize((224,224),Image.BICUBIC),dtype=np.uint8).copy()).unsqueeze(0).repeat(3,1,1) for p in ps])
rng=np.random.default_rng(0); idx=rng.permutation(len(ps)); te,tr=idx[:60],idx[60:]
mu,sd=y[tr].mean(),y[tr].std()
def go(dev,epochs=6):
    torch.manual_seed(0)
    m=timm.create_model('efficientnet_b0.ra_in1k',pretrained=True,num_classes=1).to(dev)
    c=m.pretrained_cfg; mean=torch.tensor(c['mean']).view(3,1,1); std=torch.tensor(c['std']).view(3,1,1)
    prep=lambda i:((X[i].float()/255-mean)/std)
    opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=1e-2)
    for ep in range(epochs):
        m.train(); p=rng.permutation(tr); L=[]
        for k in range(0,len(p)-15,16):
            b=p[k:k+16]; l=nn.functional.l1_loss(m(prep(b).to(dev)).squeeze(1),torch.tensor((y[b]-mu)/sd,dtype=torch.float32,device=dev))
            opt.zero_grad();l.backward();opt.step();L.append(l.item())
        m.eval()
        with torch.no_grad():
            pe=torch.cat([m(prep(te[k:k+32]).to(dev)).squeeze(1).cpu() for k in range(0,60,32)]).numpy()*sd+mu
            m_cpu=None
        print(dev,'ep',ep+1,'train',round(np.mean(L)*sd,2),'test eval',round(np.abs(pe-y[te]).mean(),2),'pred std',round(pe.std(),2),flush=True)
    return m,prep
m,prep=go('mps')
# same trained weights: eval on MPS vs CPU, and BN buffer sanity
with torch.no_grad():
    a=torch.cat([m(prep(te[k:k+32]).to('mps')).squeeze(1).cpu() for k in range(0,60,32)]).numpy()*sd+mu
    m2=m.to('cpu'); b=torch.cat([m2(prep(te[k:k+32])).squeeze(1) for k in range(0,60,32)]).numpy()*sd+mu
print('same weights, MPS eval MAE',round(np.abs(a-y[te]).mean(),2),'CPU eval MAE',round(np.abs(b-y[te]).mean(),2),'mean abs diff',round(np.abs(a-b).mean(),3))
bn=[(n,x) for n,x in m2.named_buffers() if 'running_var' in n]
print('BN running_var range',min(x.min().item() for _,x in bn),max(x.max().item() for _,x in bn))
print('drop_path', m2.drop_path_rate if hasattr(m2,'drop_path_rate') else None, 'cfg', m2.pretrained_cfg.get('input_size'), m2.pretrained_cfg.get('mean'))
