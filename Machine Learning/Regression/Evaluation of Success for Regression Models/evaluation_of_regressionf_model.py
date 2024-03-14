import numpy as np
import pandas as pd

pd.set_option('display.width', 1881)
pd.set_option('display.max_columns', 1881)

df = pd.DataFrame(
    {
        'Deneyim_Yılı(x)': [5, 7, 3, 3, 2, 7, 3, 10, 6, 4, 8, 1, 1, 9, 1],
        'Maaş(y)':         [600, 900, 550, 500, 400, 950, 540, 1200, 900, 550, 1100, 460, 400, 1000, 380]
        }
    )

# Doğrusal regresyon model denklemi (bias = 275, weight = 90): y = 275 + (90 * x)
bias = 275
weight = 90
# Denklemden maaş(y) tahmini ve df'e eklenmesi:
df['Maaş(ý)_Tahmin'] = bias + (weight * df['Deneyim_Yılı(x)'])

# Hata(y-ý):
df['Hata(y-ý)'] = df['Maaş(y)'] - df['Maaş(ý)_Tahmin']

# Hata Kareleri:
df['Hata Kareleri'] = (df['Hata(y-ý)']) ** 2

# Mutlak Hata (|y-ý|):
df['Mutlak Hata'] = [x * (-1) if x < 0 else x for x in df['Hata(y-ý)']]

# MSE, RMSE ve MAE:
MSE = df['Hata Kareleri'].sum() / len(df)

RMSE = MSE ** (1/2)

MAE = df['Mutlak Hata'].mean()