import pandas as pd

df = pd.read_csv("data.csv", delimiter=';')

""" Her urun adi icin kac farkli urun kodu vardir?
    Ornegin:
    "TÜRK KAHVESİ ORTA KAVRULMUŞ 100G" 2 farkli kodla listelenmisse, bu sayi 2 olur. """

urun_adina_gore_farkli_kod_sayisi = df.groupby("Ürün")["Kod"].nunique()

# Her urun icin kac farkli urun kodu vardir?
kod_gore_farkli_urun_sayisi = df.groupby("Kod")["Ürün"].nunique()

# Kac tane urun adi birden fazla koda sahiptir?
print("Birden Fazla Koda Sahip Urun Adi Sayisi:", (urun_adina_gore_farkli_kod_sayisi > 1).sum()) # .sum() True degerlerini sayiyor.

# Kac tane kod birden fazla urun adina sahiptir?
print("Birden Fazla Urun Adina Sahip Urun Kodu Sayisi:", (kod_gore_farkli_urun_sayisi > 1).sum())

print("\nAyni Urun Adina Karsilik Gelen Birden Fazla Kod Ornekleri:")
print(urun_adina_gore_farkli_kod_sayisi[urun_adina_gore_farkli_kod_sayisi > 1].head())

print("\nAyni Koda Karsilik Gelen Birden Fazla Urun Adi Ornekleri:")
print(kod_gore_farkli_urun_sayisi[kod_gore_farkli_urun_sayisi > 1].head())
