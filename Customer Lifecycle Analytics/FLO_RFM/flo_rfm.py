import datetime as dt
import re

import pandas as pd

pd.set_option('display.max_columns', 1881)
pd.set_option('display.width', 1881)

# Attributes
# master_id Eşsiz müşteri numarası
# order_channel Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile)
# last_order_channel En son alışverişin yapıldığı kanal
# first_order_date Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online Müşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline Müşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12 Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi

df = pd.read_csv("/Customer Lifecycle Analytics/FLO_RFM/flo_data_20k.csv")

df_copy = df.copy()


def bracket():
    print("-----------------------------------------------------------------------------------------------------------")


def get_info(dataframe):

    print(dataframe.head(10))
    bracket()

    print(dataframe.columns)
    bracket()

    print(dataframe.info())
    bracket()

    print(dataframe.describe())
    bracket()

    print(dataframe.isnull().any())
    bracket()


def change_dtype(dataframe, column, new_type):

    dataframe[column] = dataframe[column].astype(new_type)


def prepare_data(dataframe):

    df["total_orders"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
    df["total_values"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

    date_columns = ["first_order_date", "last_order_date", "last_order_date_online", "last_order_date_offline"]
    for i in date_columns:
        change_dtype(df, i, "datetime64[ns]")


df["order_channel"].describe().T
df["order_channel"].value_counts()
df["total_orders"].describe().T

df["total_values"].sort_values(ascending = True).head(10)
df["total_orders"].sort_values(ascending = False).head(10)


def get_rfm_metrics(dataframe):

    today_date = dt.datetime(2021, 12, 30)
    rfm_df = dataframe.groupby("master_id").agg(
        {
            "last_order_date": lambda x: (today_date - x.max()).days,
            "total_orders": lambda x: x,
            "total_values": lambda x: x
            }
        )

    rfm_df.columns = ["recency", "frequency", "monetary"]

    return rfm_df


rfm = get_rfm_metrics(df)


def get_rf_score(dataframe):

    dataframe["recency_score"] = pd.qcut(dataframe["recency"], q = 5, labels = [5, 4, 3, 2, 1])
    dataframe["frequency_score"] = pd.qcut(
        dataframe["frequency"].rank(method = "first"), q = 5, labels = [1, 2, 3, 4, 5]
        )
    dataframe["monetary_score"] = pd.qcut(dataframe["monetary"], q = 5, labels = [1, 2, 3, 4, 5])

    dataframe["RF_SCORE"] = dataframe["recency_score"].astype(str) + dataframe["frequency_score"].astype(str)

    return dataframe


get_rf_score(rfm)


def get_rfm_by_segments(dataframe):

    rf_values = ["hibernating", "at risk", "can't loose them", "about to sleep", "need attention",
                 "loyal customers", "promising", "potential loyallists", "new customers", "champions"]
    rf_keys = [r"[12][12]", r"[12][34]", r"[12][5]", r"[3][12]", r"[3][3]",
               r"[34][45]", r"[4][1]", r"[45][23]", r"[5][1]", r"[5][45]"]
    rf_labels = dict(zip(rf_keys, rf_values))

    dataframe["segments"] = dataframe["RF_SCORE"].replace(rf_labels, regex = True)


get_rfm_by_segments(rfm)

# Segmentlerin recency, frequnecy ve monetary ortalamalarını inceleyiniz.
rfm.groupby("segments").agg(
    {
        "recency": "mean",
        "frequency": "mean",
        "monetary": "mean"
        }
    )

# FLO bünyesine yeni bir kadın ayakkabı markası dahil ediyor. Dahil ettiği markanın ürün fiyatları genel müşteri
# tercihlerinin üstünde.
# Bu nedenle markanın tanıtımı ve ürün satışları için ilgilenecek profildeki müşterilerle özel olarak
# iletişime geçmek isteniliyor.
# Sadık müşterilerinden(champions, loyal_customers) ve kadın kategorisinden alışveriş
# yapan kişiler özel olarak iletişim kurulacak müşteriler.
# Bu müşterilerin id numaralarını csv dosyasına kaydediniz.

loyal_cust_ids = rfm[(rfm["segments"] == "champions") | (rfm["segments"] == "loyal_customers")]

loyal_cust_ids.reset_index(inplace = True)
loyal_cust_ids_list = list(loyal_cust_ids["master_id"])

(df[df["master_id"].isin(loyal_cust_ids_list) & df["interested_in_categories_12"].str.contains('KADIN')]
["master_id"]).to_csv("FLO_RFM/loyal_customers_ids.csv")

# Erkek ve Çocuk ürünlerinde %40'a yakın indirim planlanmaktadır. Bu indirimle ilgili kategorilerle ilgilenen geçmişte
# iyi müşteri olan ama uzun süredir alışveriş yapmayan kaybedilmemesi gereken müşteriler, uykuda olanlar ve yeni
# gelen müşteriler özel olarak hedef alınmak isteniyor. Uygun profildeki müşterilerin id'lerini csv dosyasına kaydediniz

pd.DataFrame(
    {
        "Customer Id": (rfm[(rfm["segments"] == "hibernating") |
                            (rfm["segments"] == "about to sleep") |
                            (rfm["segments"] == "new customers")].index)
        }
    ).to_csv("FLO_RFM/customer_ids.csv")
