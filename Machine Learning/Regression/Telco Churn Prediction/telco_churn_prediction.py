# İş tanımı: Şirketi terk edecek müşterileri tahmin edebilecek bir makine öğrenmesi modeli geliştirilmesi beklenmektedir

# Attributes:
# CustomerId Müşteri İd’si
# Gender Cinsiyet
# SeniorCitizen Müşterinin yaşlı olup olmadığı (1, 0)
# Partner Müşterinin bir ortağı olup olmadığı (Evet, Hayır)
# Dependents Müşterinin bakmakla yükümlü olduğu kişiler olup olmadığı (Evet, Hayır)
# tenure Müşterinin şirkette kaldığı ay sayısı
# PhoneService Müşterinin telefon hizmeti olup olmadığı (Evet, Hayır)
# MultipleLines Müşterinin birden fazla hattı olup olmadığı (Evet, Hayır, Telefon hizmeti yok)
# InternetService Müşterinin internet servis sağlayıcısı (DSL, Fiber optik, Hayır)
# OnlineSecurity Müşterinin çevrimiçi güvenliğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# OnlineBackup Müşterinin online yedeğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# DeviceProtection Müşterinin cihaz korumasına sahip olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# TechSupport Müşterinin teknik destek alıp almadığı (Evet, Hayır, İnternet hizmeti yok)
# StreamingTV Müşterinin TV yayını olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# StreamingMovies Müşterinin film akışı olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# Contract Müşterinin sözleşme süresi (Aydan aya, Bir yıl, İki yıl)
# PaperlessBilling Müşterinin kağıtsız faturası olup olmadığı (Evet, Hayır)
# PaymentMethod Müşterinin ödeme yöntemi (Elektronik çek, Posta çeki, Banka havalesi (otomatik), Kredi kartı (otomatik))
# MonthlyCharges Müşteriden aylık olarak tahsil edilen tutar
# TotalCharges Müşteriden tahsil edilen toplam tutar
# Churn Müşterinin kullanıp kullanmadığı (Evet veya Hayır)

import pandas as pd
import numpy as np


def grab_col_names(dataframe, cat_th = 10, car_th = 20, info = False):
    """

    Parameters
    ----------
    dataframe
    cat_th
    car_th
    info

    Returns
    -------
    cat_cols, num_cols, cat_but_car
    """
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

df = pd.read_csv(r"C:\Users\Souljah_Pc\PycharmProjects\case_studies\Machine Learning\Regression"
                 r"\Telco Churn Prediction\Telco-Customer-Churn.csv")

grab_col_names(df, info = True)
gr
rare_analyser(df, target = 'Churn', )