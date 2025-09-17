# Stok Tahmin ve Veri Analizi Projesi

Bu proje, bir satış veri seti üzerinden **veri analizi (EDA)** ve **stok tahmin modelleri** oluşturmayı amaçlamaktadır. Proje, staj kapsamında yapılan adım adım çalışmaların GitHub üzerinde paylaşımı için düzenlenmiştir.

---

## Proje Dosya Yapısı 
```
PredictionCase/
│
├── data.csv # Satış Veri Seti
├── Control.py # Ürün İsimlendirme ve Kodlama Tutarsızlıklarının Analizi
├── dataAnalysis.py # Veri Analizi (EDA) ve Görselleştirmeler
├── dateTimeControl.py # Tarih Sütununun Veri Tipi Kontrolü
├── stokTahmin-1.py # Geçmiş Satış Verileri İle Gelecek Ay Satış Tahmini
├── stokTahmin-2.py # Haftalık Satış Verileri İle Gelecek Ay Satış Tahmini
├── stokTahmin-3.py # Aylık Satış Verileri İle Gelecek Ay Satış Tahmini
├── stokTahmin-3_GradientBoostingModel.py # Aylık Satış Verileri İle Gelecek Ay Satış Tahmini
└── README.md # Proje Açıklamaları
```

---

## Proje Amaçları

1. **Veri Analizi (EDA)**  
   - Eksik verileri tespit etmek ve görselleştirmek  
   - Zaman serisi özelliklerini (Yıl, Ay, Gün, Saat, Mevsim) eklemek  
   - Ürün satışlarının zamana, mevsime ve fiyat değişimine göre dağılımını incelemek

2. **Veri Kalitesi Kontrolü**  
   - Aynı ürün adı ile birden fazla kod olup olmadığını incelemek  
   - Aynı ürün koduna karşılık birden fazla ürün adı olup olmadığını kontrol etmek  
   - Tutarsızlık veya eksik veri varsa raporlamak

3. **Stok Tahmin Modelleri**  
   - Seçilen ürün için haftalık ve aylık satış tahminleri yapmak  
   - Walk-Forward Validation ve TimeSeriesSplit ile model performansını değerlendirmek  
   - Modeller:
     - **Random Forest Regressor** (Nonlineer İlişkileri Yakalamak)
     - **Gradient Boosting Regressor** (Daha Güçlü Tahminler)

---

## Kullanılan Teknolojiler ve Kütüphaneler

- Python 3.x  
- Pandas, NumPy  
- Matplotlib, Seaborn, Plotly  
- Scikit-learn (RandomForestRegressor, GradientBoostingRegressor)  
- missingno (Eksik Veri Görselleştirmesi)  

---

## Dosya Açıklamaları

### 1. Control.py
- Veri setindeki **ürün isimleri ve kodlarındaki tutarsızlıkları** tespit eder.  
- Aynı ürün adına karşılık birden fazla kod veya aynı kodda birden fazla ürün olup olmadığını kontrol eder.  

### 2. dataAnalysis.py
- Seçilen ürünün satış ve fiyat verilerini **detaylı analiz ve görselleştirme** ile inceler.  
- Saatlik, günlük, aylık ve mevsimsel satış dağılımlarını gösterir.  
- Fonksiyonlara ayrılmış, parametreli ve yeniden kullanılabilir şekilde tasarlanmıştır.  

### 3. dateTimeControl.py
- Veri setindeki tarih sütunlarının doğru veri tipinde olup olmadığını kontrol eder.

### 4. stokTahmin-1.py
- Seçilen bir ürünün geçmiş satış verilerinden gelecek ay satış miktarını tahmin eder.

### 5. stokTahmin-2.py
- Seçilen bir ürünün haftalık satış verilerini kullanarak gelecek ay stok tahminini yapar.
- Modelin performansını değerlendirir.

### 6. stokTahmin-3.py
- Seçilen bir ürünün aylık satış verilerini kullanarak gelecek ay tahminlerini yapar.
- Modelin performansını değerlendirir.

---

## Görselleştirmeler

- Zaman İçindeki Satış ve Fiyat Trendleri (Line Plot)
- Mevsim ve Ay Bazlı Satış Dağılımları (Bar Plot)  
- Fiyat ve Satış Miktarı İlişkisi (Scatter Plot)  
- Günlük ve Saatlik Satış Dağılımları (Line/Bar Plot)  

