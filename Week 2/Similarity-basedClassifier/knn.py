from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale
import pandas

data = pandas.read_csv('../wine.data', header=None)

data_class = data[0]
data_attribute = data.drop([0], axis=1)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

max = 0
k = 0;

file = open("gradeBeforeScale.txt", "a")

for i in range(1, 50):
    model = KNeighborsClassifier(n_neighbors=i)
    knn = model.fit(data_attribute, data_class)
    # Вычислить качество
    grade = cross_val_score(estimator=knn, X=data_attribute, y=data_class, cv=kf)

    if max < grade.mean():
        max = grade.mean()
        k = i


    file.write(str(grade) + ' ' + str(round(grade.mean(), 2)) + '\n')

file.close()

file = open("KBeforeScale.txt", "w")
file.write(str(k))
file.close()

data_attribute = scale(data_attribute)

file = open("gradeAfterScale.txt", "a")

for i in range(1, 50):
    model = KNeighborsClassifier(n_neighbors=i)
    knn = model.fit(data_attribute, data_class)
    grade = cross_val_score(estimator=knn, X=data_attribute, y=data_class, cv=kf)

    if max < grade.mean():
        max = grade.mean()
        k = i


    file.write(str(grade) + ' ' + str(round(grade.mean(), 2)) + '\n')

file.close()

file = open("KAfterScale.txt", "w")
file.write(str(k))
file.close()