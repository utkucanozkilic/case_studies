import pandas as pd

pd.set_option('display.max_columns', 1881)
pd.set_option('display.width', 1881)

df = pd.DataFrame(
    {
        'Gerçek Değer':                                           [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        'Model Olasılık Tahmini (1 sınıfına ait olma olasılığı)': [0.7, 0.8, 0.65, 0.9, 0.45, 0.5, 0.55, 0.35, 0.4,
                                                                   0.25]
        }
    )

threshold = 0.5

# Tahminler:
df['Tahmin'] = [1 if x >= 0.5 else 0 for x in df['Model Olasılık Tahmini (1 sınıfına ait olma olasılığı)']]

df['true_positive'] = [1 if gercek == tahmin == 1 else 0 for gercek, tahmin in zip(df['Gerçek Değer'], df['Tahmin'])]
df['true_negative'] = [1 if gercek == tahmin == 0 else 0 for gercek, tahmin in zip(df['Gerçek Değer'], df['Tahmin'])]
df['false_positive'] = [1 if (gercek == 0 and tahmin == 1) else 0 for gercek, tahmin in
                        zip(df['Gerçek Değer'], df['Tahmin'])]
df['false_negative'] = [1 if (gercek == 1 and tahmin == 0) else 0 for gercek, tahmin in
                        zip(df['Gerçek Değer'], df['Tahmin'])]
df['accuracy'] = [1 if (gercek == tahmin == 1) or (gercek == tahmin == 0) else 0 for gercek, tahmin in
                  zip(df['Gerçek Değer'], df['Tahmin'])]

true_positive, true_negative, false_positive, false_negative = (df['true_positive'].sum(), df['true_negative'].sum(),
                                                                df['false_positive'].sum(), df['false_negative'].sum())

accuracy = df['accuracy'].sum() / len(df)

precision = true_positive / (true_positive + false_positive)

recall = true_positive / (true_positive + false_negative)

f1_score = 2 * (precision * recall) / (precision + recall)