"""Ön çalışma v2 deneyleri: tekrarlı 5 katlı çapraz doğrulama.

Kullanım:  .venv/bin/python scripts/exp.py convnext_square convnext_pad ...
Her (tekrar, kat) sonucu results_v2/<deney>/rep<r>_fold<f>.json dosyasına yazılır;
var olan sonuçlar atlanır, böylece çalışma kesilirse kaldığı yerden devam eder.
Her turda doğrulama ve test tahminleri kaydedilir. Tur seçimi yalnızca doğrulama
hatasıyla yapılır; test tahminleri seçim kurallarını sonradan karşılaştırmak içindir.
"""
import os, re, sys, json, time, hashlib, gc
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
import torch, torch.nn as nn, timm
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get('DENTAL_DATA', ROOT / 'data/zenodo_16745408/extracted/dataset ++'))
OUT = ROOT / 'results_v2'
WEIGHTS = ROOT / 'models'
DEVICE = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')

SEED = 42
N_REPS = 3
N_FOLDS = 5
BATCH = 16
WORKERS = 2 if DEVICE == 'cuda' else 0
LR = 1e-4
WEIGHT_DECAY = 1e-2
MAX_EPOCHS = 80
PATIENCE = 12      # yumuşatılmış doğrulama hatası bu kadar tur iyileşmezse dur
SMOOTH = 5         # son 5 turun ortalaması
NEARDUP = 0.90
SIZES = {'square': (224, 224), 'pad': (160, 320)}   # (yükseklik, genişlik); ikisi de ~50 bin piksel

CONFIGS = {
    'convnext_square': dict(model='convnext_tiny.fb_in1k', input='square', save=True),
    'convnext_pad':    dict(model='convnext_tiny.fb_in1k', input='pad', save=True),
    'convnext_rank':   dict(model='convnext_tiny.fb_in1k', input='square', rank=0.5),
    'convnext_sym':    dict(model='convnext_tiny.fb_in1k', input='square', sym=1.0),
    'convnext_feat':   dict(model='convnext_tiny.fb_in1k', input='square', feat=True),
    'resnet50_square':     dict(model='resnet50.tv_in1k', input='square'),
    'efficientnet_square': dict(model='efficientnet_b0.ra_in1k', input='square'),
    'densenet_square':     dict(model='densenet121.tv_in1k', input='square'),
    'xception_square':     dict(model='legacy_xception.tf_in1k', input='square'),
}

BAND_EDGES = [8, 11, 13, 15, 16, 17, 19]
BAND_LABELS = ['9-11', '12-13', '14-15', '16', '17', '18-19']


def load_df():
    rows = []
    for p in sorted(DATA_DIR.glob('*.jpg')):
        rows.append(dict(path=str(p), name=p.name, age=int(re.match(r'\d+', p.name).group()),
                         md5=hashlib.md5(p.read_bytes()).hexdigest()))
    df = pd.DataFrame(rows).drop_duplicates('md5').reset_index(drop=True)
    X = np.stack([np.asarray(Image.open(p).convert('L').resize((96, 48), Image.BICUBIC), float).ravel() for p in df.path])
    Z = (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)
    C = Z @ Z.T / Z.shape[1]
    group = np.arange(len(df))
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            if C[i, j] > NEARDUP:
                group[group == group[j]] = group[i]
    df['group'] = group
    df['band'] = pd.cut(df.age, BAND_EDGES, labels=BAND_LABELS)
    return df


def load_image(path, mode):
    im = Image.open(path).convert('L')
    H, W = SIZES[mode]
    if mode == 'square':
        im = im.resize((W, H), Image.BICUBIC)
    else:   # oranı koru, kalan kısmı siyahla doldur
        s = min(W / im.width, H / im.height)
        nw, nh = round(im.width * s), round(im.height * s)
        canvas = Image.new('L', (W, H), 0)
        canvas.paste(im.resize((nw, nh), Image.BICUBIC), ((W - nw) // 2, (H - nh) // 2))
        im = canvas
    return torch.from_numpy(np.asarray(im, dtype=np.uint8).copy()).unsqueeze(0).repeat(3, 1, 1)


def make_splits(df, rep):
    seed = SEED + rep
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    splits = []
    for fold, (trval, te) in enumerate(sgkf.split(df, df.band, df.group)):
        inner = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed + fold)
        tr_i, va_i = next(inner.split(df.iloc[trval], df.band.iloc[trval], df.group.iloc[trval]))
        tr, va = trval[tr_i], trval[va_i]
        assert not (set(df.group[tr]) & set(df.group[te])) and not (set(df.group[tr]) & set(df.group[va]))
        splits.append((tr, va, te))
    return splits


def load_features(df):
    """Kör evrelenmiş yirmilik diş evreleri (0-4), 0-1 aralığına ölçeklenmiş: [görüntü, 2]."""
    f = pd.read_csv(ROOT / 'annot/third_molar_stages.csv').set_index('name').loc[df.name]
    return torch.tensor(f[['s48', 's38']].values / 4, dtype=torch.float32)


class FeatNet(nn.Module):
    """Derin öznitelik vektörü (havuzlanmış) + el yapımı öznitelikler -> tek doğrusal katman."""
    def __init__(self, timm_name, n_feat):
        super().__init__()
        self.backbone = timm.create_model(timm_name, pretrained=True, num_classes=0)
        self.head = nn.Linear(self.backbone.num_features + n_feat, 1)
        nn.init.trunc_normal_(self.head.weight, std=.01); nn.init.zeros_(self.head.bias)
        self.pretrained_cfg = self.backbone.pretrained_cfg
    def forward(self, x, f):
        return self.head(torch.cat([self.backbone(x), f], 1))


class XrayDS(Dataset):
    def __init__(self, images, ages, idx, mean, std, train):
        self.images, self.ages, self.idx = images, ages, idx
        aug = [v2.RandomAffine(degrees=5, translate=(.03, .03), scale=(.95, 1.05), interpolation=v2.InterpolationMode.BILINEAR),
               v2.ColorJitter(brightness=.15, contrast=.15)] if train else []
        self.tf = v2.Compose(aug + [v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean, std)])
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        j = self.idx[i]
        return self.tf(self.images[j]), float(self.ages[j]), j


def build_model(timm_name):
    model = timm.create_model(timm_name, pretrained=True, num_classes=1)
    if 'efficientnet' in timm_name:   # timm tek çıktılı son katmanı çok büyük başlatıyor
        head = model.get_classifier(); nn.init.trunc_normal_(head.weight, std=.01); nn.init.zeros_(head.bias)
    return model


FEATS = None   # özniteliklerle eğitilen deneylerde [görüntü, öznitelik] tensörü


def forward(model, x, j):
    return model(x, FEATS[j].to(DEVICE)) if isinstance(model, FeatNet) else model(x)


@torch.no_grad()
def predict(model, loader):
    model.eval(); out = []
    for x, _, j in loader:
        out.append(forward(model, x.to(DEVICE), j).squeeze(1).float().cpu())
    return torch.cat(out).numpy()


def rank_loss(p, t):
    """İkili sıralama kaybı (RankNet): yaşı büyük olanın tahmini de büyük olmalı."""
    dy = t[:, None] - t[None, :]
    dp = p[:, None] - p[None, :]
    mask = dy > 0
    if not mask.any():
        return p.sum() * 0
    return nn.functional.softplus(-5 * dp[mask]).mean()


def train_fold(name, cfg, df, images, splits, rep, fold):
    seed = SEED + rep * 100 + fold
    torch.manual_seed(seed); np.random.seed(seed)
    tr, va, te = splits[fold]
    model = (FeatNet(cfg['model'], FEATS.shape[1]) if cfg.get('feat') else build_model(cfg['model'])).to(DEVICE)
    mean, std = model.pretrained_cfg['mean'], model.pretrained_cfg['std']
    y = df.age.values.astype(float); mu, sd = y[tr].mean(), y[tr].std()
    g = torch.Generator().manual_seed(seed)
    tr_dl = DataLoader(XrayDS(images, y, tr, mean, std, True), BATCH, shuffle=True, drop_last=True, generator=g, num_workers=WORKERS)
    va_dl = DataLoader(XrayDS(images, y, va, mean, std, False), 64)
    te_dl = DataLoader(XrayDS(images, y, te, mean, std, False), 64)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, MAX_EPOCHS)
    w_rank, w_sym = cfg.get('rank', 0), cfg.get('sym', 0)
    hist, val_preds, test_preds = [], [], []
    best_s, best_ep, state = 1e9, None, None
    t0 = time.time()
    for ep in range(MAX_EPOCHS):
        model.train(); losses = []
        for xb, yb, jb in tr_dl:
            xb = xb.to(DEVICE); t = ((yb - mu) / sd).float().to(DEVICE)
            if w_sym:
                out = model(torch.cat([xb, torch.flip(xb, dims=[3])])).squeeze(1)
                p, pf = out[:len(xb)], out[len(xb):]
                loss = nn.functional.l1_loss(p, t) + w_sym * (p - pf).abs().mean()
            else:
                p = forward(model, xb, jb).squeeze(1)
                loss = nn.functional.l1_loss(p, t)
            if w_rank:
                loss = loss + w_rank * rank_loss(p, t)
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
        sch.step()
        pv = predict(model, va_dl) * sd + mu; pt = predict(model, te_dl) * sd + mu
        val_preds.append(pv.tolist()); test_preds.append(pt.tolist())
        vmae = float(np.abs(pv - y[va]).mean())
        hist.append(dict(epoch=ep + 1, train_loss=float(np.mean(losses)), val_mae=vmae))
        if ep + 1 >= SMOOTH:
            s = float(np.mean([h['val_mae'] for h in hist[-SMOOTH:]]))
            if s < best_s:
                best_s, best_ep = s, ep
                if cfg.get('save'):
                    state = {k: v.detach().to('cpu', torch.float16).clone() for k, v in model.state_dict().items()}
            elif ep - best_ep >= PATIENCE:
                break
    secs = time.time() - t0
    if state is not None:
        d = WEIGHTS / name; d.mkdir(parents=True, exist_ok=True)
        torch.save(dict(state=state, mu=mu, sd=sd, model=cfg['model'], input=cfg['input']), d / f'rep{rep}_fold{fold}.pt')
    best_single = int(np.argmin([h['val_mae'] for h in hist]))
    res = dict(tr=list(map(int, tr)), va=list(map(int, va)), te=list(map(int, te)), mu=mu, sd=sd,
               hist=hist, val_preds=val_preds, test_preds=test_preds,
               epoch_smooth=best_ep, epoch_best=best_single, seconds=secs, epochs=len(hist))
    del model, opt, state; gc.collect()
    if DEVICE == 'mps': torch.mps.empty_cache()
    return res


def run(name):
    global FEATS
    cfg = CONFIGS[name]
    df = load_df()
    FEATS = load_features(df) if cfg.get('feat') else None
    images = torch.stack([load_image(p, cfg['input']) for p in df.path])
    d = OUT / name; d.mkdir(parents=True, exist_ok=True)
    for rep in range(N_REPS):
        splits = make_splits(df, rep)
        for fold in range(N_FOLDS):
            f = d / f'rep{rep}_fold{fold}.json'
            if f.exists() or (SMOKE and fold > 0):
                continue
            r = train_fold(name, cfg, df, images, splits, rep, fold)
            f.write_text(json.dumps(r))
            y = df.age.values
            pt = np.array(r['test_preds'][r['epoch_smooth']])
            print(f'{name} rep{rep} fold{fold}: test MAE {np.abs(pt - y[r["te"]]).mean():.3f} | '
                  f'tur {r["epochs"]} (seçilen {r["epoch_smooth"] + 1}) | {r["seconds"] / 60:.1f} dk', flush=True)


SMOKE = os.environ.get('DENTAL_SMOKE')   # hızlı deneme klasörü: 2 tur, tek kat

if __name__ == '__main__':
    if SMOKE:
        MAX_EPOCHS, SMOOTH, N_REPS = 2, 1, 1
        OUT = Path(SMOKE); WEIGHTS = OUT / 'models'
    for name in sys.argv[1:]:
        run(name)
