"""Grad-CAM haritalarını CPU'da yeniden hesaplar ve xai/<deney>.npz içindeki 'maps' alanını günceller.
Neden: Apple MPS üzerinde bu modelin gradyanları bozuk çıkıyor (haritaların %96'sı yanlış işaretli/sıfıra yakın);
CPU'da aynı model ve görüntülerle haritalar tutarlı. Tahminler (preds) GPU'da doğru olduğundan korunur."""
import sys, json
from pathlib import Path
import numpy as np, torch, timm
sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp
name = sys.argv[1] if len(sys.argv) > 1 else 'convnext_square'
torch.set_num_threads(4)
cfg = exp.CONFIGS[name]; df = exp.load_df()
images = torch.stack([exp.load_image(p, cfg['input']) for p in df.path])
f = exp.OUT / 'xai' / f'{name}.npz'
z = dict(np.load(f, allow_pickle=True))
maps = np.full_like(z['maps'], np.nan)
for rep in range(exp.N_REPS):
    for fold in range(exp.N_FOLDS):
        r = json.loads((exp.OUT / name / f'rep{rep}_fold{fold}.json').read_text())
        ck = torch.load(exp.WEIGHTS / name / f'rep{rep}_fold{fold}.pt', map_location='cpu', weights_only=False)
        model = timm.create_model(cfg['model'], pretrained=False, num_classes=1)
        model.load_state_dict({k: v.float() for k, v in ck['state'].items()}); model.eval()
        mean = torch.tensor(model.pretrained_cfg['mean']).view(1, 3, 1, 1); std = torch.tensor(model.pretrained_cfg['std']).view(1, 3, 1, 1)
        te = np.array(r['te'])
        for s in range(0, len(te), 20):
            idx = te[s:s + 20]
            x = ((images[idx].float() / 255) - mean) / std
            store = {}; h = model.stages[2].register_forward_hook(lambda m, i, o: store.__setitem__('A', o))
            out = model(x).squeeze(1); A = store['A']; h.remove()
            g, = torch.autograd.grad(out.sum(), A)
            M = (g.mean(dim=(2, 3), keepdim=True) * A).sum(1)
            maps[idx, rep] = M.detach().numpy() * ck['sd']
        print(f'rep{rep} fold{fold} tamam', flush=True)
z['maps'] = maps
np.savez_compressed(f, **z)
print('kaydedildi', f)
