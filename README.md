# Panoramik Röntgenden Dental Yaş Tahmini

Panoramik diş röntgeninden kişinin yaşını tahmin etmek için beş CNN modelinin aynı koşullarda karşılaştırılması.

**TÜBİTAK 2209-A ön çalışması, 1. kısım** · Hazırlayan: Nurullah Barış Çelik · Danışman: Prof. Dr. Selami Beyhan

Sonuçlar ve açıklamalar için [not defterini](dental_age_comparison.ipynb) açın. Kurulumsuz görmek için `dental_age_comparison.html` dosyasını indirip tarayıcıda açabilirsiniz.

## Sonuçlar

Temizlik sonrası 294 görüntü, 9-19 yaş, 5 katlı çapraz doğrulama.

| Sıra | Model | MAE (yıl) | %95 güven aralığı | R² | ±2 yıl içinde | Parametre |
|---|---|---|---|---|---|---|
| 1 | **ConvNeXt-Tiny** | **1,06** | 0,96–1,16 | 0,68 | %85,0 | 27,8 M |
| 2 | ResNet-50 | 1,11 | 1,01–1,21 | 0,67 | %85,4 | 23,5 M |
| 3 | EfficientNet-B0 | 1,14 | 1,04–1,24 | 0,65 | %84,7 | 4,0 M |
| 4 | Xception | 1,20 | 1,10–1,32 | 0,61 | %79,9 | 20,8 M |
| 5 | DenseNet-121 | 1,20 | 1,11–1,30 | 0,63 | %83,7 | 7,0 M |
| – | *Taban çizgisi (hep ortanca yaşı söylemek)* | *2,04* | | *−0,15* | *%69,7* | |

- Beş model de taban çizgisinin hatasını yaklaşık yarıya indiriyor.
- Net bir kazanan yok: Friedman testi anlamlı değil (p = 0,084).
- Bunlar ön sonuçlardır: veri az, yaş aralığı dar, dış doğrulama yok. Adli kullanım için yeterli değildir.

## Klasörler

| Yol | İçerik |
|---|---|
| `dental_age_comparison.ipynb` | Açıklamalı ve çalıştırılmış not defteri |
| `dental_age_comparison.html` | Not defterinin kurulum gerektirmeyen kopyası |
| `results/` | Özet tablo ve her modelin her katındaki sonuçlar |
| `results_effnet_bug/` | EfficientNet-B0'ın düzeltme öncesi hatalı sonuçları (not defteri, Bölüm 11) |
| `scripts/` | Not defterini üreten kod, veri denetimi ve hata teşhis betikleri |

## Yeniden çalıştırma

Python 3.12 ile:

```bash
pip install -r requirements.txt
```

1. Görüntüleri [Zenodo](https://doi.org/10.5281/zenodo.16745408) üzerinden indirin.
2. `data/zenodo_16745408/extracted/dataset ++/` klasörüne çıkarın.
3. Not defterini çalıştırın. `results/` içindeki hazır sonuçlar kullanıldığı için eğitim tekrarlanmaz. Eğitimi baştan yapmak için oradaki `.json` dosyalarını silin.

## Veri

Görüntüler bu repoda yoktur. Not defterindeki örnek röntgenler Zenodo 16745408 setinden alınmıştır (CC BY 4.0).
