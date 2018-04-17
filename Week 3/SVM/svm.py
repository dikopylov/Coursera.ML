import pandas
from sklearn import svm

# Загрузите выборку из файла svm-data.csv.
# В нем записана двумерная выборка (целевая переменная указана в первом столбце,
# признаки — во втором и третьем)
data = pandas.read_csv('svm-data.csv', header=None)

dataX = data.drop([0], axis=1)
dataY = data.drop([1, 2], axis=1)

clf = svm.SVC(C=100000, random_state=241)
clf.fit(dataX, dataY)

sv = clf.support_

file = open("support_vector.txt", "w")
file.write(str([x+1 for x in sv]))
file.close()
