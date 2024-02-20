import pandas as pd

pd.set_option('display.max_columns', 1881)
pd.set_option('display.width', 1881)

movie = pd.read_csv(
    r"C:\Users\Souljah_Pc\PycharmProjects\case_studies\Recommend Systems"
    r"\Hybrid Recommender System\datasets\movie.csv"
    )
rating = pd.read_csv(
    r"C:\Users\Souljah_Pc\PycharmProjects\case_studies\Recommend Systems"
    r"\Hybrid Recommender System\datasets\rating.csv"
    )
rating = rating_ = rating.copy()
# rating veri setine Id’lere ait film isimlerini ve türünü movie veri setinden ekleyiniz:
rating = rating.merge(movie, how = 'inner', on = 'movieId')

out_of_calc_films = pd.DataFrame(rating.groupby('title')['rating'].count() < 1001)
out_of_calc_films = list(out_of_calc_films[out_of_calc_films['rating'] == True].index)
rating = rating[~rating['title'].isin(out_of_calc_films)]

# Alternatif:
out_of_calc_films = pd.DataFrame(rating['title'].value_counts())
out_of_calc_films = out_of_calc_films[out_of_calc_films['count'] < 1001].index
common_films = rating[~rating['title'].isin(out_of_calc_films)]

# index'te userID'lerin sutunlarda film isimlerinin ve değer olarak ratinglerin bulunduğu
# dataframe için pivot table oluşturunuz.

user_movie_pivot = pd.pivot_table(rating, index = 'userId', columns = 'title', values = 'rating')


# Fonksiyonlaştır:
def get_user_movie_df(dataframe, count = 1000):
    # Oylanma sayısı 'count'tan düşük olan filmlerin elenmesi:
    out_of_calc_films = pd.DataFrame(dataframe.groupby('title')['rating'].count() <= count)
    out_of_calc_films = list(out_of_calc_films[out_of_calc_films['rating'] == True].index)
    common_films = dataframe[~dataframe['title'].isin(out_of_calc_films)]
    # Filtrelenmiş filmlerden oluşan df'den pivot tablosu oluşturulması:
    user_movie_df = pd.pivot_table(common_films, index = 'userId', columns = 'title', values = 'rating')
    return user_movie_df


user_movie_pivot = get_user_movie_df(rating)

# Rastgele bir kullanıcı id’si seçiniz.
random_user = user_movie_pivot.sample(1).index[0]

# Seçilen kullanıcıya ait gözlem birimlerinden oluşan random_user_df adında yeni bir dataframe oluşturunuz:
# random_user'ın satırı
random_user_one_line = user_movie_pivot[user_movie_pivot.index == random_user]
# random_user'ın izlediği filmlerin listesi:
movies_watched = [cols for cols in random_user_one_line.columns if random_user_one_line.iloc[0][cols] >= 0]
# Alternatif:
# movies_watched = random_user_one_line.columns[random_user_one_line.notna().any()].tolist()
# Random_user'ın izlediği filmler ve bu filmlere oy verenlerin df'i:
movies_watched_df = user_movie_pivot[movies_watched]

# Her bir kullancının seçili user'in izlediği filmlerin kaçını izlediğini bilgisini taşıyan user_movie_count
# adında yeni bir dataframe oluşturunuz:
user_movie_count = movies_watched_df.T.notnull().sum()
user_movie_count = user_movie_count.reset_index()
# user_movie_count.rename(columns = {0: 'movie_count'}, inplace = True)
user_movie_count.columns = ['userId', 'movie_count']
# Seçilen kullanıcının oy verdiği filmlerin yüzde 60 ve üstünü izleyenlerin kullanıcı id’lerinden
# users_same_movies adında bir liste oluşturunuz:
users_same_movies = user_movie_count[user_movie_count['movie_count'] > len(movies_watched) * 0.6]['userId']

# user_same_movies listesi içerisindeki seçili user ile benzerlik gösteren kullanıcıların id’lerinin
# bulunacağı şekilde movies_watched_df dataframe’ini filtreleyiniz:

final_df = pd.concat(
    [movies_watched_df[movies_watched_df.index.isin(users_same_movies)],
     random_user_one_line[movies_watched]]
    )
# Kullanıcıların birbirleri ile olan korelasyonlarının bulunacağı yeni bir corr_df dataframe’i oluşturunuz:

corr_df = final_df.T.corr().stack().sort_values().drop_duplicates()

corr_df = pd.DataFrame(corr_df, columns = ['corr'])

corr_df.index.names = ['user_id_1', 'user_id_2']

corr_df = corr_df.reset_index()

# Seçili kullanıcı ile yüksek korelasyona sahip (0.65’in üzerinde olan) kullanıcıları filtreleyerek
# top_users adında yeni bir dataframe oluşturunuz:
top_users = corr_df[(corr_df['user_id_1'] == random_user) & (corr_df['corr'] >= 0.65)][
    ['user_id_2', 'corr']].reset_index(drop = True)

# top_users dataframe’ine rating veri seti ile merge ediniz:
top_users.rename(columns = {"user_id_2": "userId"}, inplace = True)
top_users_ratings = top_users.merge(rating[['userId', 'movieId', 'rating']], how = 'inner')
top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user]

# Her bir kullanıcının corr ve rating değerlerinin çarpımından oluşan
# weighted_rating adında yeni bir değişken oluşturunuz:
top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']

# Film id’si ve her bir filme ait tüm kullanıcıların weighted rating’lerinin
# ortalama değerini içeren recommendation_df adında yeni bir dataframe oluşturunuz:
recommendation_df = top_users_ratings.groupby('movieId').agg({'weighted_rating': 'mean'})
recommendation_df = recommendation_df.reset_index()

#  recommendation_df içerisinde weighted rating'i 3.5'ten büyük olan filmleri
#  seçiniz ve weighted rating’e göre sıralayınız:
recommendation_df[recommendation_df["weighted_rating"] > 3.5].sort_values('weighted_rating', ascending = False)

# movie veri setinden film isimlerini getiriniz ve tavsiye edilecek ilk 5 filmi seçiniz:
movies_to_be_recommend = recommendation_df[recommendation_df['weighted_rating'] > 3.5
                                           ].sort_values('weighted_rating', ascending = False)
movies_to_be_recommend = movies_to_be_recommend['movieId']
list(movie[movie['movieId'].isin(movies_to_be_recommend)]['title'])[:5]

# movie, rating veri setlerini okutunuz: // yukarıda okutuldu

# Seçili kullanıcının 5 puan verdiği filmlerden puanı en güncel olan filmin id'sinin alınız.

film_id = rating[(rating['userId'] == random_user) & (rating['rating'] == 5)
                 ].sort_values('timestamp', ascending = False)['movieId'].iloc[0]

# Ölme eşeğim ölme yolu:
rating[
    (rating['userId'] == random_user) &
    (rating['rating'] == 5) &
    (rating['timestamp'] ==
     rating[(rating['userId'] == random_user) & (rating['rating'] == 5)]['timestamp'].max())]['movieId'].iloc[0]

# User based recommendation bölümünde oluşturulan user_movie_df dataframe’ini seçilen film id’sine göre filtreleyiniz:
movie_title = rating[rating['movieId'] == film_id]['title'].iloc[0]

movie_name = user_movie_pivot[movie_title]

# Filtrelenen dataframe’i kullanarak seçili filmle diğer filmlerin korelasyonunu bulunuz ve sıralayınız:
user_movie_pivot.corrwith(movie_name).sort_values(ascending = False)

# Seçili film’in kendisi haricinde ilk 5 film’I öneri olarak veriniz:
user_movie_pivot.corrwith(movie_name).sort_values(ascending = False)[1: 6]
