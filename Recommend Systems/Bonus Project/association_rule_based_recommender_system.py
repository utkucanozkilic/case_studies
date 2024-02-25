import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
import seaborn as sns

pd.set_option("display.max_columns", 1881)
pd.set_option("display.width", 1881)


df_ = pd.read_excel(r"C:\Users\Souljah_Pc\PycharmProjects\case_studies\Recommend Systems"
                    r"\Bonus Project\online_retail_II.xlsx", sheet_name = 'Year 2010-2011')

df = df_.copy()

df.head()


def iqr(dataframe, column, q1_ratio = 0.25, q3_ratio = 0.75, k = 1):
    q1 = dataframe[column].quantile(q1_ratio)
    q3 = dataframe[column].quantile(q3_ratio)
    iqr_range = q3 - q1
    lower_bound = q1 - k * iqr_range
    upper_bound = q3 + k * iqr_range
    return lower_bound, upper_bound


def iqr_replace(dataframe, column, q1_ratio = 0.25, q3_ratio = 0.75, k = 1):
    lower_bound, upper_bound = iqr(dataframe, column, q1_ratio, q3_ratio, k)
    dataframe.loc[(dataframe[column] < lower_bound), column] = lower_bound
    dataframe.loc[(dataframe[column] > upper_bound), column] = upper_bound


def prepare_data(dataframe, q1_ratio = 0.25, q3_ratio = 0.75, k = 1):
    dataframe.dropna(inplace = True)
    dataframe = dataframe[(dataframe['StockCode'] != 'POST') & (~dataframe['Invoice'].str.contains('C', na = False))]
    dataframe = dataframe[(dataframe['Quantity'] > 0) & (dataframe['Price'] > 0)]
    iqr_replace(dataframe, 'Quantity', q1_ratio, q3_ratio, k)
    iqr_replace(dataframe, 'Price', q1_ratio, q3_ratio, k)
    return dataframe


df = prepare_data(df)


def create_invoice_product_df(dataframe, id = True):
    if id:
        return pd.pivot_table(
            dataframe[['Invoice', 'StockCode']].astype(str),
            index = 'Invoice',
            columns = 'StockCode',
            aggfunc = lambda x: True,
            fill_value = False
            )
    else:
        return pd.pivot_table(
            dataframe[['Invoice', 'Description']].astype(str),
            index = 'Invoice',
            columns = 'Description',
            aggfunc = lambda x: True,
            fill_value = False
            )

df_germany = df[df['Country'] == 'Germany']
df_germany_pivot = create_invoice_product_df(df_germany, True)


def create_rules(dataframe):
    freq_items = apriori(dataframe, min_support = 0.01, use_colnames = True, verbose = True)
    rules = association_rules(freq_items, metric = 'support', min_threshold = 0.01)
    return rules


df_germany_rules = create_rules(df_germany_pivot)


def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe['StockCode'] == stock_code]['Description'].values[0]
    return product_name


def arl_recommender(rules, product, recommend_count = 3):
    recommended_products = []
    rules = rules.sort_values(by = 'support', ascending = False)

    for index, product_ in enumerate(rules['antecedents']):
        if product in list(product_):
            recommended_products.append(list(rules.iloc[index]['consequents'])[0])

    return set(recommended_products)


arl_recommender(df_germany_rules, '15056BL')

