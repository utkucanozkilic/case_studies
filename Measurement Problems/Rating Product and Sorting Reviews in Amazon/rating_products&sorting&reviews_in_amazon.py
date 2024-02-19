import pandas as pd
import scipy.stats as stats
import math

pd.set_option('display.width', 1881)
pd.set_option('display.max_columns', 1881)

df_ = pd.read_csv("/Measurement Problems/Rating Product and Sorting Reviews in Amazon/amazon_review.csv")
df = df_.copy()

# Kullanılmayacak sütunlar elendi
df = df[["reviewerID", "helpful", "overall", "unixReviewTime",
         "reviewTime", "day_diff", "helpful_yes", "total_vote"]]

# helpful/total_vote uyumsuzluğu var mı?
(df[df["helpful_yes"] > df["total_vote"]]).any()

# Görev-1 (Time-Based Weighted Average)

# Ürünün ortalaması
product_mean = df["overall"].mean()

date_min, date_max = df["day_diff"].min(), df["day_diff"].max()


# Tarihe göre ağırlıklı ortalama puanı
def time_based_weighted_average(dataframe, *weight):
    if not weight:
        weight = [0.25, 0.25, 0.25, 0.25]
    return (dataframe.loc[dataframe["day_diff"] <= 250, "overall"].mean() * weight[0] +
            (dataframe.loc[(dataframe["day_diff"] > 250) &
                           (dataframe["day_diff"] <= 500), "overall"].mean() * weight[1]) +
            (dataframe.loc[(dataframe["day_diff"] > 500) &
                           (dataframe["day_diff"] <= 750), "overall"].mean() * weight[2]) +
            dataframe.loc[dataframe["day_diff"] > 750, "overall"].mean() * weight[3]
            )


time_based_weighted_average(df)

# Zaman dilimlerinin ortalamasını karşılaştırarak yorumlama (aralıkların ortalamalrı birbirine yakın)
((df.loc[df["day_diff"] <= 250, "overall"].mean() * 0.25) +
 (df.loc[(df["day_diff"] > 250) & df["day_diff"] <= 500, "overall"].mean() * 0.25) +
 (df.loc[(df["day_diff"] > 500) & df["day_diff"] <= 750, "overall"].mean() * 0.25) +
 (df.loc[df["day_diff"] > 750, "overall"].mean()) * 0.25
 )

# Görev 2  Ürün için ürün detay sayfasında görüntülenecek 20 review'i belirleyiniz.

# helpful_no değişkenini üretiniz.
df["helpful_no"] = df["total_vote"] - df["helpful_yes"]


# score_pos_neg_diff:
def score_pos_neg_diff(up, down):
    return up - down


def score_average_rating(up, down):
    if down == 0:
        return 0

    return up / down


def wilson_lower_bound(up, n, confidence = 0.95):
    if n == 0:
        return 0
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    phat = up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


# score_pos_neg_diff
df["score_pos_neg_diff"] = df.apply(lambda x: score_pos_neg_diff(x["helpful_yes"],
                                                                 x["helpful_no"]), axis=1)

# score_average_rating
df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"],
                                                                     x["helpful_no"]), axis=1)

# wilson_lower_bound
df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"],
                                                                 x["helpful_no"] + x["helpful_yes"]), axis=1)

df.sort_values(by = "wilson_lower_bound", ascending = False).head(20)