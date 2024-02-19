import pandas as pd
from scipy.stats import shapiro, levene, ttest_ind


pd.set_option('display.max_columns', 1881)
pd.set_option('display.width', 1881)


# Değişkenler #
# Impression: Reklam görüntüleme sayısı
# Click: Görüntülenen reklama tıklama sayısı
# Purchase: Tıklanan reklamlar sonrası satın alınan ürün sayısı
# Earning: Satın alınan ürünler sonrası elde edilen kazanç


def compaire_p(p, alpha = 0.05):
    if p < alpha:
        return "h0 reddedildi."
    else:
        return "h0 reddedilmedi."


# Kontrol ve test grubu okunur
cg = pd.read_excel(
    "C:/Users/Souljah_Pc/PycharmProjects/Rating Products&Soritng Reviews in Amazon/"
    "AB Test/ab_testing.xlsx", sheet_name = "Control Group"
    )
tg = pd.read_excel(
    "C:/Users/Souljah_Pc/PycharmProjects/Rating Products&Soritng Reviews in Amazon/"
    "AB Test/ab_testing.xlsx", sheet_name = "Test Group"
    )

cg.describe().T
tg.describe().T
# bidding, cg ve tg'nin birleştirilmesiyle oluşturuldu
bidding = pd.concat([cg, tg], axis = 1)

# üzerinde kolay çalışılması için sütun isimleri yeniden adlandırıldı.
bidding.columns = ["C_Impression", "C_Click", "C_Purchase", "C_Earning",
                   "T_Impression", "T_Click", "T_Purchase", "T_Earning"]

# Hipotez tanımları yapıldı
# h0: maximum bidding ve average bidding yöntemleri ile yapılan satın alımların ortalamaları arasında istatistiksel
# olarak öncemli bir fark yoktur.
# h1: ... vardır.

# İki hipotez grubunun ortalamalarının gözlenmesi:
# Out: (101711.44906769728, 120512.41175753452)
bidding.C_Impression.mean(), bidding.T_Impression.mean()

# Varsayımlar analizleri gerçekleştirildi.
# Normallik varsayımı kontrolü
c_p_value = shapiro(bidding["C_Purchase"])[1]
t_p_value = shapiro(bidding["T_Purchase"])[1]
print("Kontrol grubu p değeri = {} / {}".format(c_p_value, compaire_p(c_p_value)))
print("Test grubu grubu p değeri = {} / {}".format(t_p_value, compaire_p(t_p_value)))

# Varyansların homojenliği varsayımı kontrolü
p_value = levene(
    bidding["C_Purchase"],
    bidding["T_Purchase"]
    )[1]
print("p değeri = {} / {}".format(p_value, compaire_p(p_value)))

# Varsayımlar sağlandı. parametrik test yöntemi ile hipotez, p-value değerine göre değerlendirildi
# h0 reddedilmedi, sonucuna ulaşıldı.
# dolayısıyla maximum bidding ile average bidding yöntemleriyle yapılan satışların ortalamaları arasında
# istatistiksel olarak önemli bir fark bulunmadı.
p_value = ttest_ind(bidding["C_Purchase"],
                    bidding["T_Purchase"])[1]
print("p değeri = {} / {}".format(p_value, compaire_p(p_value)))
