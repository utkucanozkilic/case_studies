import seaborn as sns
import pandas as pd


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1881)

# 1
df = sns.load_dataset("titanic")
print("\n************************************(2)***************************************\n")

# 2
print(df["sex"].value_counts(), "\n************************************(3)***************************************\n")

# 3 nunique bul
for col in df.columns:
    print("Number of unique values in", col, ":", len(df[col].unique()), "(with NaN values)", "|",
          df[col].nunique(), "(without NaN values)")
print("\n************************************(4)***************************************\n")

# 4 'pclass' için unique değerlerin sayısı
print("'number of unique values of pclass'", df["pclass"].nunique())
print("\n************************************(5)***************************************\n")

# 5 'pclass' ve 'parch' için
print("'number of unique values of pclass'", df["pclass"].nunique())
print("'number of unique values of parch'", df["parch"].nunique())
print("\n************************************(6)***************************************\n")

# 6 embarked tip kontrol et ve 'category' yap
print("Before changing type of 'embarked':", df['embarked'].dtype)
df['embarked'] = df['embarked'].astype('category')
print("After changing type of 'embarked'", df['embarked'].dtype)
print("\n************************************(7)***************************************\n")

# 7 embarked == C olanları listele
embarked_c = df[df['embarked'] == 'C']
print(embarked_c.head())
print("\n************************************(8)***************************************\n")

# 8
embarked_non_c = df[df['embarked'] != 'C']
print(embarked_non_c.head())
print("\n************************************(9)***************************************\n")

# 9 Yaş < 30 and female
lower_30_female = df[(df['sex'] == 'female') & (df['age'] < 30)]
print(lower_30_female.head())
print("\n************************************(10)***************************************\n")

# 10) fare > 500 or age > 70
lower_500_fare_upper_age_70 = df[(df['fare'] > 500) | (df['age'] > 70)]
print(lower_500_fare_upper_age_70.head())
print("\n************************************(11)***************************************\n")

# 11) Herbir değişkendeki boş değerlerin toplamı
for col in df.columns:
    if df[col].isnull().sum():
        print("The '{}' has {} NaN values." .format(col, df[col].isnull().sum()))
print("\n************************************(12)***************************************\n")

# 12) drop who
df.drop("who", axis=1, inplace=True)
print("\n************************************(13)***************************************\n")

# 13) Deck's NaN = mode(deck)
print("Before 'fillna':", df["deck"].isnull().sum())
mode_deck = df["deck"].mode()  # En çok tekrar eden değeri pandas serisi olarak döndürür.
df["deck"].fillna(mode_deck[0], inplace=True)
print("After 'fillna':", df["deck"].isnull().sum())
print("\n************************************(14)***************************************\n")

# 14) age's NaN = median(age)
print("Before 'fillna':", df["age"].isnull().sum())
median_age = df["age"].median()
df["age"].fillna(median_age, inplace=True)
print("After 'fillna':", df["age"].isnull().sum())
print("\n************************************(15)***************************************\n")

# 15) survived değişkeninin pclass ve cinsiyet değişkenleri kırılımınında sum, count, mean değerlerini bulunuz.
columns = ["pclass", "sex"]
agg_strings = ["sum", "count", "mean"]
concat_list = []

for col in columns:
    for agg_ in agg_strings:

        concat_list.append(df.groupby(col)["survived"].agg([agg_]))

    print(pd.concat(concat_list, axis = 1), "\n")
    concat_list.clear()
print("\n************************************(16)***************************************\n")
# print(df.groupby("pclass")["survived"].agg(["sum", "count", "mean"]))
# print(df.groupby("sex")["survived"].agg(["sum", "count", "mean"]))

# 16) 30 yaşın altında olanlar 1, 30'a eşit ve üstünde olanlara 0 vericek bir fonksiyon yazın. Yazdığınız
# fonksiyonu kullanarak titanik veri setinde age_flag adında bir değişken oluşturunuz oluşturunuz.
# (apply ve lambda yapılarını kullanınız)

df["age_flag"] = df["age"].apply(func = lambda x: 1 if x < 30 else 0)
print(pd.concat([df["age_flag"], df["age"]], axis = 1).head(10))
print(df.columns)
print("\n************************************(17)***************************************\n")

# 17) Seaborn kütüphanesi içerisinden Tips veri setini tanımlayınız.
df = sns.load_dataset("tips")
print("\n************************************(18)***************************************\n")

# 18) Time değişkeninin kategorilerine (Dinner, Lunch) göre total_bill değerlerinin
# toplamını, min, max ve ortalamasını bulunuz.
print(df.groupby("time")["total_bill"].agg(["sum", "min", "max", "mean"]))
print("\n************************************(19)***************************************\n")

# 19) Günlere ve time göre total_bill değerlerinin toplamını, min, max ve ortalamasını bulunuz.
print(df.groupby(["day", "time"])["total_bill"].agg(["sum", "min", "max", "mean"]))
print(df.groupby(["day", "time"])["total_bill"].agg(["sum", "min", "max", "mean"]).unstack())
print("\n************************************(20)***************************************\n")

# 20) Lunch zamanına ve kadın müşterilere ait total_bill ve tip değerlerinin day'e göre
# toplamını, min, max ve ortalamasını bulunuz
print((df[(df["time"] == "Lunch") & (df["sex"] == "Female")].groupby(["day"])[["total_bill", "tip"]].agg(
    ["sum", "min", "max", "mean"])))
print("\n************************************(21)***************************************\n")

# 21) size'i 3'ten küçük, total_bill'i 10'dan büyük olan siparişlerin ortalaması nedir? (loc kullanınız)

print("size'i 3'ten küçük, total_bill'i 10'dan büyük olan siparişlerin ortalaması:",
      df.loc[(df["size"] < 3) & (df["total_bill"] > 10)]["total_bill"].agg("mean"))
print("\n************************************(22)***************************************\n")

# 22) total_bill_tip_sum adında yeni bir değişken oluşturunuz.
# Her bir müşterinin ödediği totalbill ve tip in toplamını versin.
df["total_bill_tip_sum"] = df["total_bill"] + df["tip"]
print(df["total_bill_tip_sum"].head())
print("\n************************************(23)***************************************\n")

# 23) total_bill_tip_sum değişkenine göre büyükten küçüğe sıralayınız ve ilk 30 kişiyi yeni bir dataframe'e atayınız

first_30_people = []

for i in range(30):
    first_30_people.append(df["total_bill_tip_sum"].sort_values(ascending = True).index.tolist()[i])

new_df = pd.DataFrame(data = df.iloc[first_30_people])

print(new_df.head())
