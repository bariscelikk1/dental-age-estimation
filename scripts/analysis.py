"""v2 sonuçlarının analizinde kullanılan yardımcı fonksiyonlar (notebook bunları çağırır)."""
import json
from pathlib import Path
import numpy as np, pandas as pd
import torch
from scipy import stats
import exp

BANDS = exp.BAND_LABELS


def load_runs(name):
    """{tekrar: [kat sonuçları]}; eksik katlar atlanır."""
    out = {}
    for rep in range(exp.N_REPS):
        fs = [exp.OUT / name / f'rep{rep}_fold{f}.json' for f in range(exp.N_FOLDS)]
        if all(f.exists() for f in fs):
            out[rep] = [json.loads(f.read_text()) for f in fs]
    return out


def oof(runs, n, rule='smooth', calibrate=False):
    """Her tekrar için kat dışı tahminler: dizi [tekrar, görüntü].
    rule: 'smooth' (son 5 turun ortalaması), 'best' (tek en iyi tur), 'last' (son tur), ya da tamsayı (sabit tur).
    calibrate: her katta doğrulama kümesine doğru uydurup test tahminlerine tersini uygular."""
    P = np.full((len(runs), n), np.nan)
    for i, (rep, folds) in enumerate(sorted(runs.items())):
        for r in folds:
            if rule == 'smooth': e = r['epoch_smooth']
            elif rule == 'best': e = r['epoch_best']
            elif rule == 'last': e = len(r['test_preds']) - 1
            else: e = min(rule, len(r['test_preds'])) - 1
            pt = np.array(r['test_preds'][e])
            if calibrate:
                yv = np.array(r['va_age']) if 'va_age' in r else None
                pv = np.array(r['val_preds'][e])
                a, b = np.polyfit(yv, pv, 1)
                pt = (pt - b) / a
            P[i, r['te']] = pt
    return P


def add_val_ages(runs, ages):
    for folds in runs.values():
        for r in folds:
            r['va_age'] = ages[r['va']].tolist()
    return runs


def metrics(y, p):
    e = p - y; a = np.abs(e)
    return dict(MAE=a.mean(), RMSE=np.sqrt((e ** 2).mean()), R2=1 - (e ** 2).mean() / y.var(),
                within1=(a <= 1).mean() * 100, within2=(a <= 2).mean() * 100, bias=e.mean(),
                slope=np.polyfit(y, p, 1)[0])


def summarize(P, y, runs):
    """Tekrar başına MAE, ortalama ± std, kat std'si ve 3 tekrarın ortalamasının (topluluk) MAE'si."""
    rep_mae = [np.abs(P[i] - y).mean() for i in range(len(P))]
    fold_sd = [np.std([np.abs(P[i][r['te']] - y[r['te']]).mean() for r in folds])
               for i, (_, folds) in enumerate(sorted(runs.items()))]
    ens = P.mean(0)
    return dict(reps=len(P), MAE_mean=np.mean(rep_mae), MAE_sd=np.std(rep_mae, ddof=1) if len(P) > 1 else np.nan,
                MAE_reps=rep_mae, fold_sd=np.mean(fold_sd), MAE_ensemble=np.abs(ens - y).mean(),
                epochs=np.mean([r['epochs'] for f in runs.values() for r in f]),
                epoch_selected=np.mean([r['epoch_smooth'] + 1 for f in runs.values() for r in f]),
                hit_cap=sum(r['epochs'] == exp.MAX_EPOCHS for f in runs.values() for r in f))


def band_mae(P, y, bands):
    a = np.abs(P - y).mean(0) if P.ndim == 2 else np.abs(P - y)
    return pd.Series(a).groupby(np.asarray(bands), observed=True).mean().reindex(BANDS)


# --- Grad-CAM bölgeleri: panoramik görüntünün kaba, sabit bir bölümlemesi (segmentasyon değildir) ---
REGIONS = {
    'Yirmilik bölgesi':        [((.20, .30), (.25, .85)), ((.70, .80), (.25, .85))],
    'Büyük azılar':            [((.30, .40), (.25, .85)), ((.60, .70), (.25, .85))],
    'Ön dişler, küçük azılar': [((.40, .60), (.25, .85))],
}
S = 224


def region_masks():
    xs = (np.arange(S) + .5) / S
    X, Y = np.meshgrid(xs, xs)
    masks = {}
    for k, boxes in REGIONS.items():
        m = np.zeros((S, S), bool)
        for (x0, x1), (y0, y1) in boxes:
            m |= (X >= x0) & (X < x1) & (Y >= y0) & (Y < y1)
        masks[k] = m
    masks['Diş dışı (kemik, kafatası)'] = ~np.any(list(masks.values()), axis=0)
    return masks


def upsample(maps):
    t = torch.from_numpy(np.asarray(maps, np.float32)).reshape(-1, 1, *maps.shape[-2:])
    return torch.nn.functional.interpolate(t, size=(S, S), mode='bilinear', align_corners=False).numpy().reshape(*maps.shape[:-2], S, S)


def xai_maps(npz):
    """Atıf haritası: üyelerin pozitif Grad-CAM ortalaması.
    Belirsizlik haritası: topluluk varyansının her üyenin öznitelik haritasına göre Grad-CAM'i,
    yani relu(d_k * M_k) ortalaması; d_k üyenin tahmininin topluluk ortalamasından sapması."""
    p0 = npz['preds'][:, :, 0]                      # TTA'sız tahminler [N, tekrar]
    d = p0 - p0.mean(1, keepdims=True)
    M = npz['maps']                                  # [N, tekrar, 14, 14], CPU'da hesaplanmış işaretli harita
    attr = np.maximum(M, 0).mean(1)
    unc = np.maximum(d[:, :, None, None] * M, 0).mean(1)
    return upsample(attr), upsample(unc)


def region_shares(maps, masks):
    tot = maps.sum((1, 2)) + 1e-12
    return pd.DataFrame({k: (maps * m).sum((1, 2)) / tot for k, m in masks.items()})


def uncertainty_table(npz, bands):
    P = npz['preds']; y = npz['age']
    mean = P.mean((1, 2))
    epi = P.mean(2).std(1)                 # tekrarlar arası (model) belirsizlik
    tta = P.std(2).mean(1)                 # çoğaltmalar arası duyarlılık
    tot = P.reshape(len(P), -1).std(1)
    err = np.abs(mean - y)
    return pd.DataFrame(dict(name=npz['names'], age=y, band=np.asarray(bands), pred=mean, err=err,
                             sd_model=epi, sd_tta=tta, sd_total=tot))


def stage_baseline(df, runs):
    """Yalnızca yirmilik diş evresiyle (iki taraf) doğrusal regresyon; her katta eğitim+doğrulama ile uydurulur."""
    f = pd.read_csv(exp.ROOT / 'annot/third_molar_stages.csv').set_index('name').loc[df.name][['s48', 's38']].values
    X = np.c_[np.ones(len(f)), f]; y = df.age.values.astype(float)
    P = np.full((len(runs), len(df)), np.nan)
    for i, (_, folds) in enumerate(sorted(runs.items())):
        for r in folds:
            tr = np.r_[r['tr'], r['va']]
            w = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]
            P[i, r['te']] = X[r['te']] @ w
    return P


def median_baseline(df, runs):
    y = df.age.values.astype(float)
    P = np.full((len(runs), len(df)), np.nan)
    for i, (_, folds) in enumerate(sorted(runs.items())):
        for r in folds:
            P[i, r['te']] = np.median(y[np.r_[r['tr'], r['va']]])
    return P
