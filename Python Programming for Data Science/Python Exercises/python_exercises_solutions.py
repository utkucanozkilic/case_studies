#  Görev-1 geçildi
#  Görev-2
text = "The goal is to turn data into information, and information into insight."
text = text.upper()
text = text.replace(",", "")
text = text.replace(".", "")
text = text.split(" ")
print(text)

# Görev-3
lst = ['D', 'A', 'T', 'A', 'S', 'C', 'I', 'E', 'N', 'C', 'E']

len_lst = len(lst)

i_0, i_10 = lst[0], lst[10]

lst_data = []
for i in range(4):
     lst_data.append(lst[i])
print(lst_data)

lst.pop(8)

lst.append("ZzZ")

lst.insert(8, "N")
print(lst)

# Görev 4
dict = {
    'Christian': ['America', 18],
    'Daisy': ['England', 12],
    'Antonio': ['Spain', 22],
    'Dante': ['Italy', 25]
    }

print(dict.keys())
print(dict.values())

dict.update({'Daisy': ['England', 13]})

dict.update({'Ahmet': ['Turkey', 24]})
dict.pop('Antonio')

print(dict)


# Görev 5

odd_list, even_list = [], []

def extract(liste):
    for i in liste:
        if i % 2 == 0:
            even_list.append(i)
        else:
            odd_list.append(i)

    return odd_list, even_list

# Görev 6

ogrenciler = ["Ali", "Veli", "Ayşe", "Talat", "Zeynep", "Ece"]

counter = 1
for i, element in enumerate(ogrenciler):
    if i < 3:
        print("Mühendislik fakültesi {}. öğrenci: {}" .format(i + 0, element))
    else:
        print("Tıp fakültesi {}. öğrenci: {}" .format(i-2, element))


# Görev 7

ders_kodu = ["CMP1005", "PSY1001", "HUK1005", "SEN2204"]
kredi = [3, 4, 2, 4]
kontenjan = [30, 75, 150, 25]

total_list = list(zip(ders_kodu, kredi, kontenjan))

for i in total_list:
    print("Kredisi {} olan {} kodlu dersin kontenjanı {} kişidir." .format(str(i[1]), i[0], str(i[2])))

# Görev 8

kume1 = set(["data", "python"])
kume2 = set(["data", "function", "qcut", "lambda", "python", "miuul"])

if kume1.issuperset(kume2):
    print(kume1.intersection(kume2))
else:
    print(kume2.difference(kume1))

# Görev 1

import seaborn as sns

df = sns.load_dataset("car_crashes")

colls = ["NUM_" + col.upper() if df[col].dtype in ["float"] else col.upper() for col in df.columns]
print(colls)

# Görev 2

cols = [col.upper() + "_FLAG" if "no" not in col else col.upper() for col in df.columns]

print(cols)

# Görev 3

import seaborn as sns

new_cols = [col for col in df.columns if col not in ["abbrev", "no_previous"]]

new_df = df[new_cols]

