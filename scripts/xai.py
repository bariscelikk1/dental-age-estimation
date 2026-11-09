"""Belirsizlik ve Grad-CAM hesapları (ConvNeXt-Tiny, kare girdi).

Her görüntü, onu eğitimde görmemiş 3 modelle (3 tekrarın ilgili katı) tahmin edilir.
Her model için 8 test zamanı çoğaltması (TTA) uygulanır.
Grad-CAM, ConvNeXt'in 3. aşamasının (14x14) çıktısı üzerinde hesaplanır ve işaretli
(ReLU uygulanmamış) harita olarak saklanır; belirsizlik haritası analizde bundan türetilir.

Kullanım:  .venv/bin/python scripts/xai.py [deney adı]
Çıktı: results_v2/xai/<deney>.npz
"""
import os, sys, json
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
from pathlib import Path
import numpy as np, torch, timm
import torchvision.transforms.v2.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp

TTA = [dict(), dict(translate=[7, 0]), dict(translate=[-7, 0]), dict(translate=[0, 5]), dict(translate=[0, -5]),
       dict(scale=.97), dict(scale=1.03), dict(bright=.9)]


def tta_batch(x_uint8, t):
    x = x_uint8
    if 'translate' in t or 'scale' in t:
        x = F.affine(x, angle=0., translate=t.get('translate', [0, 0]), scale=t.get('scale', 1.), shear=[0., 0.],
                     interpolation=F.InterpolationMode.BILINEAR)
    x = F.to_dtype(x, torch.float32, scale=True)
    if 'bright' in t:
        x = (x * t['bright']).clamp(0, 1)
    return x


def main(name='convnext_square'):
    cfg = exp.CONFIGS[name]
    df = exp.load_df()
    images = torch.stack([exp.load_image(p, cfg['input']) for p in df.path])
    N = len(df)
    preds = np.full((N, exp.N_REPS, len(TTA)), np.nan)
    maps = np.full((N, exp.N_REPS, 14, 14), np.nan, dtype=np.float32)
    for rep in range(exp.N_REPS):
        for fold in range(exp.N_FOLDS):
            r = json.loads((exp.OUT / name / f'rep{rep}_fold{fold}.json').read_text())
            ck = torch.load(exp.WEIGHTS / name / f'rep{rep}_fold{fold}.pt', map_location='cpu', weights_only=False)
            model = timm.create_model(cfg['model'], pretrained=False, num_classes=1)
            model.load_state_dict({k: v.float() for k, v in ck['state'].items()})
            model = model.to(exp.DEVICE).eval()
            mean = torch.tensor(model.pretrained_cfg['mean']).view(1, 3, 1, 1).to(exp.DEVICE)
            std = torch.tensor(model.pretrained_cfg['std']).view(1, 3, 1, 1).to(exp.DEVICE)
            te = np.array(r['te']); xb = images[te]
            with torch.no_grad():
                for k, t in enumerate(TTA):
                    x = (tta_batch(xb, t).to(exp.DEVICE) - mean) / std
                    preds[te, rep, k] = model(x).squeeze(1).cpu().numpy() * ck['sd'] + ck['mu']
            store = {}
            h = model.stages[2].register_forward_hook(lambda m, i, o: store.__setitem__('A', o))
            x = (tta_batch(xb, {}).to(exp.DEVICE) - mean) / std
            out = model(x).squeeze(1)
            A = store['A']; h.remove()
            grad, = torch.autograd.grad(out.sum(), A)
            w = grad.mean(dim=(2, 3), keepdim=True)
            maps[te, rep] = (w * A).sum(1).detach().cpu().numpy() * ck['sd']   # yıl ölçeğinde işaretli harita
            print(f'rep{rep} fold{fold} tamam', flush=True)
            del model; torch.mps.empty_cache() if exp.DEVICE == 'mps' else None
    d = exp.OUT / 'xai'; d.mkdir(exist_ok=True)
    np.savez_compressed(d / f'{name}.npz', preds=preds, maps=maps, age=df.age.values, names=df.name.values)
    print('kaydedildi:', d / f'{name}.npz')


if __name__ == '__main__':
    main(*sys.argv[1:])
