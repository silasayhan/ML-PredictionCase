"""
stokTahmin-3.py

Seçilen bir ürünün aylık satış verilerini kullanarak:
1. Walk-Forward (TimeSeriesSplit) validasyon ile model performansını ölçer.
2. Gelecek 3 ay için satış tahminleri yapar.
3. Fold bazlı ve ortalama MSE çıktısı verir.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
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

def walk_forward_validation(X, y, n_splits = 5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    mse_list = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = RandomForestRegressor(n_estimators = 100, random_state = 42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mse_list.append(mse)

        print(f"Fold {fold + 1}: MSE = {mse:.2f}")

    ortalama_mse = np.mean(mse_list)
    print(f"\nWalk Forward Validation Ortalama MSE: {ortalama_mse:.2f}")
    return model  # Son fold modeli yerine tüm veri ile tekrar eğitmek için geri dönebiliriz.


def gelecek_ay_tahmin(model, gelecek_df):
    tahminler = model.predict(gelecek_df)
    gelecek_df['TahminiSatis'] = tahminler.astype(int)
    return gelecek_df

if __name__ == "__main__":
    secili_urun = "FRUTİBON YABAN MERSİNİ  50GR * 12ADT"
    aylik_satis = hazirla_veri("data.csv", secili_urun)

    X = aylik_satis[['Yıl', 'Ay']].values
    y = aylik_satis['Miktar'].values

    print(f"{secili_urun} İçin Toplam {len(y)}Aylık Veri Vardır.\n")

    # Walk-Forward Validation
    rf_model = walk_forward_validation(X, y, n_splits = 5)

    # Gelecek 3 Ay İçin Tahmin
    gelecek_aylar = pd.DataFrame({
        'Yıl': [2025, 2025, 2025],
        'Ay': [6, 7, 8]
    })
    
    gelecek_tahmin_df = gelecek_ay_tahmin(rf_model, gelecek_aylar)

    print("\n2025 Yaz Ayları Tahmini Satış Miktarları (Random Forest):")
    for idx, row in gelecek_tahmin_df.iterrows():
        print(f"{row['Yıl']} Yılı {row['Ay']}. Ayında Tahmini Satış: {row['TahminiSatis']} Adet")
