"""
stokTahmini-3_GradientBoostingModel.py

Seçilen bir ürünün aylık satış verilerini kullanarak:
1. Yıllık ve aylık satış görselleştirmeleri yapar.
2. Gradient Boosting Regressor ile model kurar.
3. Test verisi üzerinde model performansını (MSE) değerlendirir.
4. Gelecek 3 ay için tahminler üretir.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error


def hazirla_veri(dosya, urun_adi):
    df = pd.read_csv(dosya, sep = ';')
    df['Tarih'] = pd.to_datetime(df['Tarih'])

    df_urun = df[df['Ürün'] == urun_adi].copy()
    df_urun['Yıl'] = df_urun['Tarih'].dt.year
    df_urun['Ay'] = df_urun['Tarih'].dt.month

    aylik_satis = df_urun.groupby(['Yıl', 'Ay'])['Miktar'].sum().reset_index()
    return aylik_satis

# Yıllık toplam ve aylık satışları görselleştirir.
def gorsellestir_satislar(aylik_satis):
    
    # Yıllık Toplam Satışlar
    plt.figure(figsize = (8,5))
    yillik_toplam = aylik_satis.groupby('Yıl')['Miktar'].sum()
    sns.barplot(x = yillik_toplam.index, y = yillik_toplam.values)
    plt.title('Yıllık Toplam Satışlar')
    plt.xlabel('Yıl')
    plt.ylabel('Satış Miktarı')
    plt.show()

    # Aylık Satışların Yıllara Göre KArşılaştırılması
    pivot = aylik_satis.pivot(index = 'Ay', columns = 'Yıl', values = 'Miktar')
    plt.figure(figsize = (10,6))
    pivot.plot(marker = 'o')
    plt.title('Aylık Satışlar Yıllara Göre')
    plt.xlabel('Ay')
    plt.ylabel('Satış Miktarı')
    plt.xticks(range(1,13))
    plt.legend(title = 'Yıl')
    plt.show()

def train_gb_model(X, y, n_splits = 5):
    tscv = TimeSeriesSplit(n_splits = n_splits)
    mse_list = []
    model = GradientBoostingRegressor(n_estimators = 100, learning_rate = 0.1, random_state = 42)

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mse_list.append(mse)
        print(f"Fold {fold + 1}: MSE = {mse:.2f}")

    print(f"\nWalk Forward Validation Ortalama MSE: {np.mean(mse_list):.2f}")
    # Son fold modeli geri döndürülüyor; istenirse tüm veri ile yeniden fit edilebilir.
    model.fit(X, y)
    return model


def gelecek_ay_tahmin(model, gelecek_df):
    tahminler = model.predict(gelecek_df)
    gelecek_df['TahminiSatis'] = tahminler.astype(int)
    return gelecek_df

def grafik_tahmin(y_test, y_pred):
    plt.figure(figsize = (5, 3))
    plt.plot(y_test.values, label = 'Gerçek Satış', marker = 'o')
    plt.plot(y_pred, label = 'Tahmin Edilen Satış', marker = 'x')
    plt.title('Gerçek ve Tahmin Edilen Satış Karşılaştırması (Gradient Boosting)')
    plt.xlabel('Test Örnekleri')
    plt.ylabel('Satış Miktarı')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    secili_urun = "FRUTİBON YABAN MERSİNİ  50GR * 12ADT"
    aylik_satis = hazirla_veri("data.csv", secili_urun)
    gorsellestir_satislar(aylik_satis)

    X = aylik_satis[['Yıl', 'Ay']].values
    y = aylik_satis['Miktar'].values

    gb_model = train_gb_model(X, y, n_splits = 5)

    # Test tahmini için tüm veri kullanıldığı için görselleştirme burada basit olur.
    y_pred_full = gb_model.predict(X)
    grafik_tahmin(pd.Series(y), y_pred_full)

    # Gelecek 3 Ay Tahmini
    gelecek_aylar = pd.DataFrame({
        'Yıl': [2025, 2025, 2025],
        'Ay': [6, 7, 8]
    })
    
    gelecek_tahmin_df = gelecek_ay_tahmin(gb_model, gelecek_aylar)

    print("\n2025 Yaz Ayları Tahmini Satış Miktarları (Gradient Boosting):")
    for idx, row in gelecek_tahmin_df.iterrows():
        print(f"{row['Yıl']} Yılı {row['Ay']}. Ayında Tahmini Satış: {row['TahminiSatis']}.")
