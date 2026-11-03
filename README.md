# Panoramik Röntgenden Dental Yaş Tahmini

Bu çalışmada, panoramik diş röntgeninden yaş tahmini için beş CNN modeli aynı koşullarda karşılaştırılmıştır.

**TÜBİTAK 2209-A ön çalışması, 1. kısım** · Hazırlayan: Nurullah Barış Çelik · Danışman: Prof. Dr. Selami Beyhan

Yöntem ve sonuçların ayrıntıları [not defterinde](dental_age_comparison.ipynb) yer almaktadır. `dental_age_comparison.html` dosyası kurulum gerektirmeden tarayıcıda görüntülenebilir.

## Sonuçlar

Temizlik sonrası 294 görüntü, 9-19 yaş, 5 katlı çapraz doğrulama.

| Sıra | Model | MAE (yıl) | %95 güven aralığı | R² | ±2 yıl içinde | Parametre |
|---|---|---|---|---|---|---|
| 1 | **ConvNeXt-Tiny** | **1,06** | 0,96–1,16 | 0,68 | %85,0 | 27,8 M |
| 2 | ResNet-50 | 1,11 | 1,01–1,21 | 0,67 | %85,4 | 23,5 M |
| 3 | EfficientNet-B0 | 1,14 | 1,04–1,24 | 0,65 | %84,7 | 4,0 M |
| 4 | Xception | 1,20 | 1,10–1,32 | 0,61 | %79,9 | 20,8 M |
| 5 | DenseNet-121 | 1,20 | 1,11–1,30 | 0,63 | %83,7 | 7,0 M |
| – | *Taban çizgisi (tüm görüntüler için ortanca yaş)* | *2,04* | | *−0,15* | *%69,7* | |

- Beş modelin tamamı, taban çizgisinin hatasını yaklaşık yarıya indirmektedir.
- En düşük hata ConvNeXt-Tiny ile elde edilmiştir; ancak Friedman testi anlamlı bulunmamıştır (p = 0,084).
- Sonuçlar, şu an erişilebilen Zenodo veri setinde elde edilmiştir ve ön sonuç niteliğindedir. Örneklem küçük, yaş aralığı dardır; dış doğrulama yapılmamıştır. Sonuçlar adli kullanım için yeterli değildir.

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

1. Görüntüler [Zenodo](https://doi.org/10.5281/zenodo.16745408) üzerinden indirilir.
2. Dosyalar `data/zenodo_16745408/extracted/dataset ++/` klasörüne çıkarılır.
3. Not defteri çalıştırılır. `results/` içindeki kayıtlı sonuçlar kullanıldığından eğitim tekrarlanmaz. Eğitimi yeniden yapmak için bu klasördeki `.json` dosyaları silinmelidir.

## Veri

Görüntüler bu depoda bulunmamaktadır. Not defterindeki örnek röntgenler Zenodo 16745408 setinden alınmıştır (CC BY 4.0).
