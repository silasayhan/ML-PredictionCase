import pandas as pd

df = pd.read_csv("data.csv", sep=';')

print(df['Tarih'].dtype)

print(df.isnull().sum())

