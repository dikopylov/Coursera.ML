from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import GridSearchCV, KFold
# TF-IDF - Это показатель, который равен произведению двух чисел:
# TF (term frequency) и IDF (inverse document frequency).
# Первая равна отношению числа вхождений слова в документ к общей длине документа.
# Вторая величина зависит от того, в скольки документах выборки встречается это слово.
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# массив с текстами будет находиться в поле newsgroups.data,
# номер класса — в поле newsgroups.target.
newsgroups = datasets.fetch_20newsgroups(
                    subset='all',
                    categories=['alt.atheism', 'sci.space']
             )

tfid_newsgroups = TfidfVectorizer()
data_params = tfid_newsgroups.fit_transform(newsgroups.data)

# Чтобы понять, какому слову соответствует i-й признак,
# можно воспользоваться методом get_feature_names() у TfidfVectorizer:
feature_mapping = tfid_newsgroups.get_feature_names()


grid = {'C': np.power(10.0, np.arange(-5, 6))}
cv = KFold(n_splits=5, shuffle=True, random_state=241)
clf = svm.SVC(kernel='linear', random_state=241)
gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
gs.fit(data_params, newsgroups.target)

clf = svm.SVC(kernel='linear', C=gs.best_estimator_.C, random_state=241)
clf.fit(data_params, newsgroups.target)

#  Получение веса каждого слова и преобразование в формат, с которым можно работать
myList = []
myList = clf.coef_
a1=np.hstack(myList)
a2=a1[0].toarray()
a3=a2[0]
d=dict([(a3[0],0)])

# Создание словаря: key = вес; value = id слова
for i in range(0, len(a3)):
    d[abs(a3[i])] = i

# Сортировка словаря и получение 10 лучших весов и id слов для этих весов
keylist = d.keys()
keylist1 = sorted(keylist)
for key in keylist1:
    print("%s: %s" % (key, d[key]))

index = [22936,15606,5776,21850,23673,17802,5093,5088,12871,24019]

#********** Создание и сортировка спсика из нужных 10 слов**********
list=[]
for i in index:
    list.append(feature_mapping[i])
list.sort()
str = ""

#********** Вывод из в строку через пробел**********
for word in list:
    str+=word + " "
print(str)