import pandas as pd
import datetime as dt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter

pd.set_option('display.max_columns', 1881)
pd.set_option('display.width', 1881)

df_copy = pd.read_csv("/Customer Lifecycle Analytics/FLO_RFM/flo_data_20k.csv")

df = df_copy.copy()


def outlier_thresholds(dataframe, variable):

    quantile_1 = dataframe[variable].quantile(0.01)
    quantile_3 = dataframe[variable].quantile(0.75)

    interquantile_range = quantile_3 - quantile_1

    up_limit = (quantile_3 + 1.5 * interquantile_range).round()
    low_limit = (quantile_1 - 1.5 * interquantile_range).round()

    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):

    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


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

    date_columns = ["first_order_date", "last_order_date", "last_order_date_online", "last_order_date_offline"]
    for i in date_columns:
        change_dtype(df, i, "datetime64[ns]")

    order_value_columns = ["order_num_total_ever_online", "order_num_total_ever_offline",
                           "customer_value_total_ever_offline", "customer_value_total_ever_online"]
    for i in order_value_columns:
        replace_with_thresholds(dataframe, i)

    df["total_orders"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
    df["total_values"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]


prepare_data(df)
df.describe()
df.info()


deneme = (df["last_order_date"] - df["first_order_date"]).dt.total_seconds() / (24 * 60 * 60) / 7


def get_cltv(dataframe):

    today_date = dt.datetime(2021, 6, 1)

    cltv = pd.DataFrame({
        "customer_id": df["master_id"],
        "recency_cltv_weekly": (df["last_order_date"] - df["first_order_date"]).dt.total_seconds() / (24 * 60 * 60) / 7,
        "T_weekly": (today_date - df["first_order_date"]).dt.total_seconds() / (24 * 60 * 60) / 7,
        "frequency": df["total_orders"],
        "monetary_cltv_avg": df["total_values"] / df["total_orders"]
        })

    bgf = BetaGeoFitter(penalizer_coef = 0.001)
    bgf.fit(cltv['frequency'], cltv['recency_cltv_weekly'], cltv['T_weekly'])

    cltv["exp_sales_3_month"] = bgf.predict(
        12,
        cltv['frequency'],
        cltv['recency_cltv_weekly'],
        cltv['T_weekly']
        )

    cltv["exp_sales_6_month"] = bgf.predict(
        18,
        cltv['frequency'],
        cltv['recency_cltv_weekly'],
        cltv['T_weekly']
        )

    ggf = GammaGammaFitter(penalizer_coef = 0.01)
    ggf.fit(cltv['frequency'], cltv["monetary_cltv_avg"])
    cltv["expected_avarage_profit"] = ggf.conditional_expected_average_profit(cltv["frequency"],
                                                                              cltv["monetary_cltv_avg"])

    cltv["cltv"] = ggf.customer_lifetime_value(
        bgf,
        cltv["frequency"],
        cltv["recency_cltv_weekly"],
        cltv["T_weekly"],
        cltv["monetary_cltv_avg"],
        time = 6,
        freq = "W",  # T'nin frekans bilgisi
        discount_rate = 0.01
        )

    return cltv


cltv = get_cltv(df)

cltv["cltv_segment"] = pd.qcut(cltv["cltv"], 4, ["D", "C", "B", "A"])
