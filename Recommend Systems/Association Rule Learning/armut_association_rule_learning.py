import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


df = pd.read_csv(r"C:\Users\Souljah_Pc\PycharmProjects\case_studies\Recommend Systems\Association Rule "
                 r"Learning\armut_data.csv")

pd.set_option('display.width', 1881)
pd.set_option('display.max_columns', 1881)


# def get_arl_matrix(dataframe):
#     dataframe_pivot_table = pd.pivot_table(
#         data = dataframe[['SepetId', 'Hizmet']],
#         index = ['SepetId'],
#         columns = ['Hizmet'],
#         aggfunc = lambda x: 1,
#         fill_value = 0
#         )
#     return dataframe_pivot_table



# veri gözlemleri:
df.shape
df.info()
df.columns
df['CreateDate'].min()
df['CreateDate'].max()
df.index

# ServisID her bir CategoryID özelinde farklı bir hizmeti temsil etmektedir. ServiceID ve CategoryID’yi "_" ile
# birleştirerek bu hizmetleri temsil edecek yeni bir değişken oluşturunuz.
df['Hizmet'] = df['ServiceId'].astype(str) + '_' + df['CategoryId'].astype(str)

# Alıştırma için list comprehension ile:
# df['Hizmet'] = [str(serviceid) + '_' + str(categoryid) for serviceid, categoryid
# in zip(df['ServiceId'], df['CategoryId'])]

df[(df['UserId'] == 7256) & (df['Hizmet'] == '46_4')]  # Mükerrer mi?

# Veri seti hizmetlerin alındığı tarih ve saatten oluşmaktadır, herhangi bir sepet tanımı (fatura vb. ) bulunmamaktadır.
# Association Rule Learning uygulayabilmek için bir sepet (fatura vb.) tanımı oluşturulması gerekmektedir.
# Burada sepet tanımı her bir müşterinin aylık aldığı hizmetlerdir.
# Örneğin, 7256 id'li müşteri 2017'in 8.ayında aldığı 9_4, 46_4 hizmetleri bir sepeti;
# 2017’in 10.ayında aldığı 9_4, 38_4 hizmetleri başka bir sepeti ifade etmektedir.
# Sepetleri unique bir ID ile tanımlanması gerekmektedir. Bunun için öncelikle sadece yıl ve ay içeren yeni bir
# date değişkeni oluşturunuz.
# UserID ve yeni oluşturduğunuz date değişkenini "_" ile birleştirirek ID adında yeni bir değişkene atayınız.
df.info()
df['CreateDate'] = pd.to_datetime(df['CreateDate'])
df['New_Date'] = [str(year) + '_' + str(month) for year, month in
                  zip(df['CreateDate'].dt.year, df['CreateDate'].dt.month)]

df['SepetId'] = df['UserId'].astype(str) + '_' + df['New_Date']

# Pivot_Table oluşturunuz
# Önemli, MultiIndex'li sütunlar olmaması için sadece ilgili sütunları data'ya gönder
df_pivot_table = pd.pivot_table(data = df[['SepetId', 'Hizmet']], index = 'SepetId', columns = 'Hizmet',
                                aggfunc = lambda x: 1, fill_value = 0)


def create_rules(arl_matrix, min_sp = 0.001, measurement = 'lift', min_th = 1):
    frequent_itemsets = apriori(arl_matrix, min_support = min_sp, use_colnames = True)
    rules = association_rules(frequent_itemsets, metric = measurement, min_threshold = min_th)
    return rules


def arl_recommender(rules, item, recomment_count = 2):
    sorted_rules = rules.sort_values(by = 'lift', ascending = False)
    recommended_items = []
    for index, service in enumerate(sorted_rules['antecedents']):
        for service_j in list(service):
            if item == service_j:
                # recommended_items.append(consequents for consequents in list(sorted_rules['consequents']))
                recommended_items.append(list(sorted_rules.iloc[index]['consequents'])[0])
                continue
    return set(recommended_items[0: recomment_count])


ass_rules = create_rules(df_pivot_table)
arl_recommender(ass_rules, '2_0', recomment_count = 10)

# arl_recommender fonksiyonunu kullanarak en son 2_0 hizmetini alan bir kullanıcıya hizmet önerisinde bulununuz:

user_id = df[(df['Hizmet'] == '2_0') &
             (df['CreateDate'] == (df[df['Hizmet'] == '2_0']['CreateDate'].max()))]['UserId'].iloc[0]

"Sevgili {}, {} hizmetini alan {} hizmetlerini de aldı." .format(
    user_id, '2_0', arl_recommender(ass_rules, '2_0'))