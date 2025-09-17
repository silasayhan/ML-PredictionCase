"""
stokTahmin-2.py

Seçilen bir ürünün haftalık satış verilerini kullanarak:
1. Walk-Forward validasyon ile model performansını ölçer.
2. Gelecek 1 ay için toplam tahmini satış miktarını hesaplar.
3. Gerçek satışlar, test tahminleri ve gelecek tahminleri grafikte gösterir.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def hazirla_veri(dosya, urun_adi):
    df = pd.read_csv(dosya, sep = ';')
    df['Tarih'] = pd.to_datetime(df['Tarih'])

    df_urun = df[df['Ürün'] == urun_adi].copy()
    df_urun['Yıl'] = df_urun['Tarih'].dt.year
    df_urun['Hafta'] = df_urun['Tarih'].dt.isocalendar().week

    haftalik_satis = (
        df_urun.groupby(['Yıl', 'Hafta'])['Miktar']
        .sum()
        .reset_index()
        .sort_values(['Yıl', 'Hafta'])
        .reset_index(drop=True)
    )

    return haftalik_satis

def walk_forward_rmse(satislar, min_train_size = 8):
    X = np.arange(1, len(satislar) + 1).reshape(-1,1)
    y = satislar

    y_true = []
    y_pred = []

    for i in range(min_train_size, len(X)):
        X_train, y_train = X[:i], y[:i]
        X_test, y_test = X[i].reshape(1,-1), y[i]

        model = RandomForestRegressor(random_state = 42, n_estimators = 100)
        model.fit(X_train, y_train)

        pred = model.predict(X_test)[0]
        y_true.append(y_test)
        y_pred.append(pred)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return rmse, X, y, y_pred

"""Son 12 haftalık veriye göre gelecek 4 haftalık satış tahmini yapar."""
def gelecek_hafta_tahmin(satislar, hafta_sayisi = 4):
   
    X = np.arange(1, len(satislar) + 1).reshape(-1,1)
    y = satislar

    X_son12 = X[-12:]
    y_son12 = y[-12:]
    ortalama_son12 = np.mean(y_son12)

    model = RandomForestRegressor(random_state = 42, n_estimators = 100)
    model.fit(X_son12, y_son12)

    gelecek_indexler = np.arange(len(satislar) + 1, len(satislar) + hafta_sayisi+1).reshape(-1,1)
    gelecek_tahminler = model.predict(gelecek_indexler)

    toplam_bir_ay = np.sum(gelecek_tahminler)
    return ortalama_son12, gelecek_indexler, gelecek_tahminler, toplam_bir_ay


def grafik_ciz(X, y, y_pred_wf, gelecek_indexler, gelecek_tahminler, urun_adi):
    plt.figure(figsize = (12, 6))
    plt.plot(X, y, marker = 'o', label = 'Gerçek Satışlar')
    plt.plot(X[len(X)-len(y_pred_wf):], y_pred_wf, 'rx--', label = 'Walk-Forward Tahminleri')
    plt.plot(gelecek_indexler, gelecek_tahminler, 'gs--', label = 'Gelecek 4 Hafta Tahmin')
    plt.title(f"{urun_adi} - Walk-Forward Tahmin & Gelecek 1 Ay")
    plt.xlabel("Hafta")
    plt.ylabel("Satış Miktarı")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    secili_urun = "FRUTİBON YABAN MERSİNİ  50GR * 12ADT"
    haftalik_satis = hazirla_veri("data.csv", secili_urun)
    satislar = haftalik_satis['Miktar'].values

    print(f"Kaç Haftalık Veri Var?: {len(satislar)}")

    # Walk-Forward RMSE
    rmse, X, y, y_pred_wf = walk_forward_rmse(satislar)
    print(f"\nWalk-Forward Test RMSE: {rmse:.2f}")

    # Son 3 Ay Ortalama & Gelecek Ay Tahmini
    ortalama_son12, gelecek_indexler, gelecek_tahminler, toplam_bir_ay = gelecek_hafta_tahmin(satislar)
    print(f"\nSon 3 Ayda Haftalık Ortalama Satış: {round(ortalama_son12)} Adet")
    print(f"\nGelecek Ay Toplam Tahmini Satış: {round(toplam_bir_ay)} Adet")

    # Grafik
    grafik_ciz(X, y, y_pred_wf, gelecek_indexler, gelecek_tahminler, secili_urun)
