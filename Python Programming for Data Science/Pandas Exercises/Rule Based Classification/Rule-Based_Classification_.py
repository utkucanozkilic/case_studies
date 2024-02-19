import numpy as np
import pandas as pd

pd.set_option('display.max_columns', 1881)
pd.set_option('display.width', 1881)

df_copy = pd.read_csv(
    "../../../../Exercises_Data_Analaysis/Ruled-Based_Classification/persona.csv"
    )

df = df_copy.copy()


def bracket(i):
    print("-------------------------------{}---------------------------------" .format(i))


def get_number_of_unique(dataframe, column_name):
    return "Number of uniques for " + column_name + " is " + str(dataframe[column_name].nunique())


def get_value_counts(dataframe, column_name, head_tail = 10):
    return dataframe[column_name].value_counts().head(head_tail)


def get_agg_multi_cols(dataframe, target_column, agg_, *cols):
    return dataframe.groupby(list(cols))[target_column].agg([agg_])


# 1) persona.csv dosyasını okutunuz ve veri seti ile ilgili genel bilgileri gösteriniz.
bracket(1)
print(df.info())
bracket(1)
print(df.describe().T)
bracket(2)

# 2) Kaç unique SOURCE vardır? Frekansları nedir?
print(get_number_of_unique(df, "SOURCE"), "\n")
print("Value counts for", get_value_counts(df, "SOURCE"))
bracket(3)

# 3) Kaç unique PRICE vardır?
print(get_number_of_unique(df, "PRICE"))
bracket(4)

# 4) Hangi PRICE'dan kaçar tane satış gerçekleşmiş?
print("Value counts for", get_value_counts(df, "PRICE"))
bracket(5)

# 5) Hangi ülkeden kaçar tane satış olmuş?
print("Value counts for", get_value_counts(df, "COUNTRY"))
bracket(6)

# 6) Ülkelere göre satışlardan toplam ne kadar kazanılmış?
print(get_agg_multi_cols(df, "PRICE", "sum", "COUNTRY"))
bracket(7)

# 7) SOURCE türlerine göre satış sayıları nedir?
print(get_value_counts(df, "SOURCE"))
bracket(8)

# 8) Ülkelere göre PRICE ortalamaları nedir?
print(get_agg_multi_cols(df, "PRICE", "mean", "COUNTRY"))
bracket(9)

# 9) SOURCE'lara göre PRICE ortalamaları nedir?
print(get_agg_multi_cols(df, "PRICE", "mean", "SOURCE"))
bracket(10)

# 10) COUNTRY-SOURCE kırılımında PRICE ortalamaları nedir?
print(get_agg_multi_cols(df, "PRICE", "mean", "COUNTRY", "SOURCE"))
bracket(11)


# 11) COUNTRY, SOURCE, SEX, AGE kırılımında ortalama kazançlar nedir?
print(get_agg_multi_cols(df, "PRICE", "mean", "COUNTRY", "SOURCE", "SEX", "AGE"))
bracket(12)

# 12) Önceki sorudaki çıktıyı daha iyi görebilmek için
# sort_values metodunu azalan olacak şekilde PRICE’a göre uygulayınız
agg_df = get_agg_multi_cols(df, "PRICE", "mean", "COUNTRY", "SOURCE", "SEX", "AGE").sort_values(
    ascending = False, by = "mean")
bracket(13)

# 13)  Üçüncü sorunun çıktısında yer alan PRICE dışındaki
# tüm değişkenler index isimleridir. Bu isimleri değişken isimlerine çeviriniz.
agg_df.reset_index(inplace = True)
bracket(14)

# 14) Age sayısal değişkenini kategorik değişkene çeviriniz. (‘0_18', ‘19_23', '24_30', '31_40', '41_70')
agg_df["AGE_CAT"] = pd.cut(
    agg_df["AGE"], bins = [0, 18, 23, 30, 40, np.inf],
    labels = ["0_18", "19_23", "24_30", "31_40", "41_70"])
bracket(15)

# 15) Yeni seviye tabanlı müşterileri tanımlayınız ve veri setine değişken olarak ekleyiniz.
# Yeni eklenecek değişkenin adı: customers_level_based
agg_df["customers_level_based"] = [agg_df.iloc[index, 0].upper() + "_" + agg_df.iloc[index, 1].upper() + "_" +
                                   agg_df.iloc[index, 2].upper() + "_" + agg_df.iloc[index, 5].upper()
                                   for index in agg_df.index]

agg_df = agg_df.groupby("customers_level_based")["mean"].agg(["mean"])
bracket(16)

# 16) Yeni müşterileri (personaları) segmentlere ayırınız.
# Segmentleri SEGMENT isimlendirmesi ile değişken olarak agg_df’e ekleyiniz.
agg_df["SEGMENT"] = pd.qcut(x = agg_df["mean"], q = 4, labels = ["D", "C", "B", "A"])

agg_list = ["min", "max", "sum"]
agg_df.reset_index(inplace = True)

# Segmentleri betimleyiniz (Segmentlere göre group by yapıp price mean, max, sum’larını alınız)
for i in agg_list:
    print(get_agg_multi_cols(agg_df, "mean", i, "SEGMENT"))

bracket(17)

# 17)33 yaşında ANDROID kullanan bir Türk kadını hangi segmente aittir ve ortalama ne kadar gelir kazandırması beklenir?
# 35 yaşında IOS kullanan bir Fransız kadını hangi segmente aittir ve ortalama ne kadar gelir kazandırması beklenir?
new_user = "TUR_ANDROID_FEMALE_31_40"
new_user_2 = "FRA_IOS_FEMALE_31_40"
print(agg_df[agg_df["customers_level_based"] == new_user])
bracket(17)
print(agg_df[agg_df["customers_level_based"] == new_user_2])