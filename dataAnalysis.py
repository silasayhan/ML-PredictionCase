"""
dataAnalysis.py

Bu dosya, ürün satış veri seti için kapsamlı veri analizi (EDA) ve görselleştirme yapar.
- Eksik veri kontrolü sağlar.
- Zaman serisi özellikleri eklenir.
- Ürün bazlı satış ve fiyat analizleri yapılır.
- Mevsim, saat, ay bazlı görselleştirmeler yapılır.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio
import missingno as msno

pio.renderers.default = "browser"
plt.rcParams["figure.figsize"] = (10,6)
sns.set_style("whitegrid")
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_columns', None)


def veri_oku(dosya):
    df = pd.read_csv(dosya, sep = ';', parse_dates = ['Tarih'], index_col = 'Tarih')
    df['Fiyat'] = df['Fiyat'].astype(str).str.replace(",", ".").astype(float)
    return df

def eksik_veri_ve_sutun_bilgisi(df):
    print("İlk 5 Satır:\n", df.head())
    print("\nDataFrame Bilgisi:\n")
    print(df.info())
    print("\nSayısal Özet:\n", df.describe())
    print("\nKategorik Özet:\n", df.describe(include = 'object'))
    print("\nEksik Veri Sayısı:\n", df.isnull().sum())
    msno.bar(df)
    plt.show()

def zaman_serisi_ozellik_ekle(df):
    df['Yıl'] = df.index.year
    df['Ay'] = df.index.month
    df['Gün'] = df.index.day
    df['Saat'] = df.index.hour

    def mevsim_bul(ay):
        if ay in [12, 1, 2]:
            return 'Kış'
        
        elif ay in [3, 4, 5]:
            return 'İlkbahar'
        
        elif ay in [6, 7, 8]:
            return 'Yaz'
        
        else:
            return 'Sonbahar'

    df['Mevsim'] = df['Ay'].apply(mevsim_bul)
    return df

def urun_sec(df, urun_adi):
    df_urun = df[df['Ürün'] == urun_adi].copy()
    df_urun.drop(columns = ['Ürün', 'Kod', 'Yıl'], inplace = True)
    df_urun['Tarih'] = df_urun.index
    return df_urun

def satis_ve_fiyat_gorsel(df_urun):
    # Plotly Çizimi
    px.line(df_urun, x = "Tarih", y = "Miktar", title = 'Zaman İçinde Satış Miktarı').show()
    px.line(df_urun, x = "Tarih", y = "Fiyat", title = 'Zaman İçinde Fiyat').show()
    px.bar(df_urun, x = "Mevsim", y = "Miktar", title = 'Mevsime Göre Satış Miktarı').show()

    # Matplotlib & Seaborn
    plt.figure(figsize = (10,5))
    sns.barplot(x = df_urun['Saat'], y = df_urun['Miktar'], estimator = sum, ci = None, palette = 'viridis')
    plt.xlabel('Saat'); plt.ylabel('Toplam Satış Miktarı'); plt.title('Saatlik Satış Dağılımı'); plt.show()

    plt.figure(figsize = (10,5))
    sns.barplot(x = df_urun['Ay'], y = df_urun['Miktar'], estimator = sum, ci = None, palette = 'coolwarm')
    plt.xlabel('Ay'); plt.ylabel('Toplam Satış Miktarı'); plt.title('Aylık Satış Dağılımı'); plt.show()

    plt.figure(figsize = (10,5))
    sns.scatterplot(x = df_urun['Fiyat'], y = df_urun['Miktar'], alpha = 0.7)
    plt.xlabel('Fiyat'); plt.ylabel('Satış Miktarı'); plt.title('Fiyat ve Satış Miktarı İlişkisi'); plt.show()

    plt.figure(figsize = (12,5))
    sns.lineplot(x = df_urun['Gün'], y = df_urun['Miktar'], estimator = sum, marker = 'o', ci = None)
    plt.xlabel('Gün'); plt.ylabel('Toplam Satış Miktarı'); plt.title('Günlük Satış Dağılımı'); plt.xticks(range(1, 32)); plt.grid(True); plt.show()

if __name__ == "__main__":
    dosya = "data.csv"
    secili_urun = "FRUTİBON YABAN MERSİNİ  50GR * 12ADT"

    df = veri_oku(dosya)
    df = zaman_serisi_ozellik_ekle(df)
    eksik_veri_ve_sutun_bilgisi(df)

    df_urun = urun_sec(df, secili_urun)
    satis_ve_fiyat_gorsel(df_urun)
