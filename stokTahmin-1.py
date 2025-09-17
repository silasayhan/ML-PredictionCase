"""
stokTahmin-1.py
--------------------
Seçilen bir ürünün geçmiş satış verilerinden yola çıkarak gelecek ay satış miktarını tahmin eder.

Adımlar:
1. Veriyi okur ve tarih sütununu işler.
2. Seçilen ürün için aylık satış toplamlarını hesaplar.
3. Satış trendini görselleştirir.
4. RandomForestRegressor modeli ile tahmin yapar.
5. Modelin hatasını (RMSE) hesaplar ve gelecek ay için tahmin verir.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def hazirla_veri(dosya, urun_adi):
    df = pd.read_csv(dosya, sep = ";")
    df["Tarih"] = pd.to_datetime(df["Tarih"])

    # Filtreleme
    df_urun = df[df["Ürün"] == urun_adi].copy()

    df_urun["Yıl"] = df_urun["Tarih"].dt.year
    df_urun["Ay"] = df_urun["Tarih"].dt.month

    # Aylık Satış Toplamı
    aylik_satis = (
        df_urun.groupby(["Yıl", "Ay"])["Miktar"].sum().reset_index()
    )

    aylik_satis["Tarih"] = pd.to_datetime(
        aylik_satis["Yıl"].astype(str) + "-" + aylik_satis["Ay"].astype(str) + "-01"
    )

    aylik_satis = aylik_satis.sort_values(["Yıl", "Ay"]).reset_index(drop = True)

    return aylik_satis

# Aylık Satış Grafiği
def gorsellestir_satislar(aylik_satis, urun_adi):
    plt.figure(figsize = (12, 6))
    plt.plot(aylik_satis["Tarih"], aylik_satis["Miktar"], marker = "o")
    plt.title(f"{urun_adi} İçin Aylık Satışlar")
    plt.xlabel("Tarih")
    plt.ylabel("Satış Miktarı")
    plt.grid(True)
    plt.show()

def egit_model(aylik_satis):
    satislar = aylik_satis["Miktar"].values
    X = np.arange(1, len(satislar) + 1).reshape(-1, 1)
    y = satislar

    train_size = int(len(X) * 0.8)
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]

    # Model
    model = RandomForestRegressor(random_state = 42, n_estimators = 100)
    model.fit(X_train, y_train)

    # Tahminler ve Hata
    y_pred_test = model.predict(X_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    # Gelecek Ay Tahmini
    gelecek_ay_index = np.array([[len(satislar) + 1]])
    gelecek_ay_tahmin = model.predict(gelecek_ay_index)[0]

    # Grafik
    plt.figure(figsize =(6, 3))

    plt.plot(
        range(len(y_test)), y_test, marker = "o", label = "Gerçek Değerler", color = "blue"
    )

    plt.plot(
        range(len(y_test)), y_pred_test, marker = "x", label = "Tahminler", color = "red"
    )

    plt.title("Test Verisi: Gerçek vs Tahmin")
    plt.xlabel("Test Veri İndeksi")
    plt.ylabel("Satış Miktarı")
    plt.legend()
    plt.grid(True)
    plt.show()

    return rmse_test, round(gelecek_ay_tahmin)

if __name__ == "__main__":
    secili_urun = "FRUTİBON YABAN MERSİNİ  50GR * 12ADT"
    aylik_satis = hazirla_veri("data.csv", secili_urun)

    print(f"Kaç Aylık Veri Var?: {len(aylik_satis)}")
    print(aylik_satis)
    print(aylik_satis.groupby("Yıl").size())

    gorsellestir_satislar(aylik_satis, secili_urun)

    rmse, tahmin = egit_model(aylik_satis)
    print(f"Model RMSE: {rmse:.2f}")
    print(f"Gelecek Ay Tahmini Satış Miktarı: {tahmin} Adet")
