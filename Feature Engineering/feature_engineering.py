import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler


###### Attributes: ######

# Pregnancies Hamilelik sayısı
# Glucose Oral glikoz tolerans testinde 2 saatlik plazma glikoz konsantrasyonu (<140mg normal, >200 diabet)
# Blood Pressure Kan Basıncı (Küçük tansiyon) (mm Hg)
# SkinThickness Cilt Kalınlığı
# Insulin 2 saatlik serum insülini (mu U/ml)
# DiabetesPedigreeFunction Fonksiyon (Oral glikoz tolerans testinde 2 saatlik plazma glikoz konsantrasyonu)
# BMI Vücut kitle endeksi
# Age Yaş (yıl)
# Outcome Hastalığa sahip (1) ya da değil (0)


def grab_col_names(dataframe, cat_th = 10, car_th = 20, info = False):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtype in ['object', 'category', 'bool']]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].dtype in ['int64', 'float64']
                   and dataframe[col].nunique() < cat_th]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th
                   and dataframe[col].dtype in ['object', 'category']]

    cat_cols = cat_cols + num_but_cat

    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtype in ['float64', 'int64']]

    num_cols = [col for col in num_cols if col not in num_but_cat]

    if not info:
        return cat_cols, num_cols, cat_but_car
    else:
        print("Observations: {}, Variables: {}".format(dataframe.shape[0], dataframe.shape[1]))
        print("Caterogical columns:", len(cat_cols))
        print("Numerical columns:", len(num_cols))
        print('Caterogical but cardinal columns:', len(cat_but_car))
        print('Numerical but caterogical columns:', len(num_but_cat))

        return cat_cols, num_cols, cat_but_car


def outlier_threshold(dataframe, column, first_percent = 0.25, third_percent = 0.75):
    q1 = dataframe[column].quantile(first_percent)
    q3 = dataframe[column].quantile(third_percent)
    iqr = q3 - q1
    up_limit = q3 + 1.5 * iqr
    low_limit = q1 - 1.5 * iqr

    return low_limit, up_limit


def check_outlier(dataframe, column, first_percent = 0.25, third_percent = 0.75):
    q1 = dataframe[column].quantile(first_percent)
    q3 = dataframe[column].quantile(third_percent)
    iqr = q3 - q1
    up_limit = q3 + 1.5 * iqr
    low_limit = q1 - 1.5 * iqr

    return dataframe[(dataframe[column] < low_limit) | (dataframe[column] > up_limit)].any(axis = None)


def replacement_with_thresholds(dataframe, column):
    low_limit, up_limit = outlier_threshold(dataframe, column)
    dataframe.loc[(dataframe[column] < low_limit), column] = low_limit.astype(dataframe[column].dtype)
    dataframe.loc[(dataframe[column] > up_limit), column] = up_limit.astype(dataframe[column].dtype)


def grab_outliers(dataframe, column, index = False):
    """
    Veri setinin istenen sütununda, aykırı değer olan gözlem noktalarını bastırır. İstenirse, bu gözlem noktalarının
    indeks bilgilerini döndürür.

    Parameters
    ----------
    dataframe:
        Üzerinde işlem yapılacak veri seti.
    column:
        Veri setinin üzerinde işlem yapılacak sütunu.
    index: bool
        indeks bilgisinin istenmesinin kontrolü.

    Returns
    -------
        Aykırı değere sahip gözlem noktalarının indeks bilgisi.
    """
    low, up = outlier_threshold(dataframe, column)

    if len(dataframe[(dataframe[column] < low) | (dataframe[column] > up)]) > 10:
        print(dataframe[(dataframe[column] < low) | (dataframe[column] > up)].head())
    else:
        print(dataframe[(dataframe[column] < low) | (dataframe[column] > up)])

    if index:
        return dataframe[(dataframe[column] < low) | (dataframe[column] > up)].index


def missing_values_table(dataframe, na_name = False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending = False)
    ratio = (dataframe[na_columns].isnull().sum() / len(dataframe) * 100).sort_values(ascending = False)
    missin_df = pd.concat(objs = [n_miss, np.round(ratio, 2)], axis = 1, keys = ['n_miss', 'ratio'])

    print(missin_df)

    if na_name:
        return na_columns


def rare_analyser(dataframe, target, categorical_col):
    for col in categorical_col:
        print(col, ":", dataframe[col].nunique(), sep = '')
        print(
            pd.DataFrame(
                {
                    'COUNT': dataframe[col].value_counts(),
                    'RATIO(%)': dataframe[col].value_counts() / len(dataframe) * 100,
                    'TARGET_MEAN': dataframe.groupby(col)[target].mean()
                    }
                )
            )
        print("\n")


pd.set_option('display.max_columns', 1881)
pd.set_option('display.width', 1881)

df = pd.read_csv(r"C:\Users\Souljah_Pc\PycharmProjects\case_studies\Feature Engineering\diabets.csv")

df.info()
df.describe().T

caterogical_columns, numerical_columns, cat_but_cardinal_columns = grab_col_names(df, info = True)

# Aykırı değer kontrolü:
for col in numerical_columns:
    print(check_outlier(df, col))

for col in numerical_columns:
    sns.boxplot(x = df[col])
    plt.show()

for col in numerical_columns:
    print(col)
    grab_outliers(df, col)
    print("\n\n")

# Hedef değişkene göre sayısal değişkenlerin ortalaması:
df.groupby('Outcome').agg(
    {
        'Pregnancies': 'mean',
        'Glucose': 'mean',
        'BloodPressure': 'mean',
        'SkinThickness': 'mean',
        'Insulin': 'mean',
        'BMI': 'mean',
        'DiabetesPedigreeFunction': 'mean',
        'Age': 'mean'
        }
    )

# Eksik gözlem kontrolü:
missing_values_table(df, True)

# Korelasyon analizi:
df.corr()['Outcome'].sort_values(ascending = False)
# Sütunlarda örneklerin eksik değerlere göre frekansı:
msno.matrix(df, figsize = (10, 10))
plt.show()
# Eksik değerlere göre ısı haritası (eksik değer olmadığı için korelasyon grafiği boş):
msno.heatmap(df.isnull(), figsize = (10, 10))  # Çalıştıramıyorum!!!
plt.show()
# Frekans grafiği
msno.bar(df, figsize = (10, 10))
plt.show()

sns.heatmap(df.corr(), annot = True, cmap = 'viridis', fmt = '.2f')
plt.show()

# Değeri '0' olan verilere eksik değer gibi davran. '0' ları 'NaN' ile değiştir:
replace_na_cols = numerical_columns.copy()
replace_na_cols.remove('Pregnancies')

df[replace_na_cols] = df[replace_na_cols].map(lambda x: np.nan if x == 0 else x)
missing_values_table(df, True)

# Aykırı değer işlemleri:
df['Pregnancies'], df['Age'] = df['Pregnancies'].astype('int64'), df['Age'].astype('int64')
for col in numerical_columns:
    replacement_with_thresholds(df, col)

# Eksik değer işlemleri:
na_columns = [col for col in df.columns if df[col].isnull().any() == True]
for col in na_columns:
    df[col].fillna(df[col].mean(), inplace = True)

# Yeni özellikler türetme
# Yaş için:
df.loc[(df['Age'] <= 35), 'New_Age_Cat'] = 'young_adult'
df.loc[(df['Age'] > 35) & (df['Age'] <= 55), 'New_Age_Cat'] = 'middle_age_adult'
df.loc[df['Age'] > 55, 'New_Age_Cat'] = 'old_adult'

df.groupby('Pregnancies').agg({'Outcome': 'mean'}).sort_values(by = 'Pregnancies', ascending = False)
rare_analyser(df, 'Outcome', ['Pregnancies'])

# df.groupby(['Pregnancies', 'Age']).agg({'Outcome': 'mean'})
# Çocuk sahini olma/yaş kırılımında diabet olma oranları:
with pd.option_context('display.max.rows', None):
    print(df.groupby(['Pregnancies', 'Age']).agg({'Outcome': 'mean'}))

# Vücut kitle indeksi/kan basıncı oranı (BMI / BloodPressure):
df['New_BMI/BloodPressure'] = df['BMI'] / df['BloodPressure']

# Şeker ve Hamilelik Sayısının Çarpımı:
df['New_GlucoseXPregnancies'] = df['Glucose'] * df['Pregnancies']

# İnsülin ve glukoz oranı (Vücuttaki insülin duyarlılığı):
df['New_Insulin/Glucose'] = df['Insulin'] / df['Glucose']

# Glukoz ve diabette genetik yatkınlık çarpımı:
df['New_GlucoseXDiabetesPedigreeFunction'] = df['Glucose'] * df['DiabetesPedigreeFunction']

# Yaş ve diabette genetik yatkınlık çarpımı:
df['New_AgeXDiabetesPedigreeFunction'] = df['Glucose'] * df['DiabetesPedigreeFunction']

# Yeni eklenen özellikler sonrası sütun isimlerini çekme:
caterogical_columns, numerical_columns, categoric_but_cardinals_columns = grab_col_names(df)

df = pd.concat(objs = (df, pd.get_dummies(df['New_Age_Cat'], drop_first = True, dtype = 'int64')), axis = 1)

# Standartlaştırma:
caterogical_columns, numerical_columns, cat_but_cardinal_columns = grab_col_names(df, info = True)

scaler = MinMaxScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Modelleme:
X = df.drop(['Outcome', 'New_Age_Cat'], axis = 1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

rf = RandomForestClassifier(random_state = 42).fit(X_train, y_train)

y_pred = rf.predict(X_test)

accuracy_score(y_pred, y_test)


# Değişkenlerin model başarısındaki etkisi:
def plot_importance(model, features, num = len(X), save = False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize = (20, 5))
    sns.set(font_scale = 1)
    sns.barplot(
        x = "Value", y = "Feature", data = feature_imp.sort_values(
            by = "Value",
            ascending = False
            )[0: num]
        )
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(rf, X_train)
