import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pandas

data = pandas.read_csv('../titanic.csv', index_col='PassengerId')

data.replace(to_replace="male", value=1, inplace=True)
data.replace(to_replace="female", value=0, inplace=True)

dataX_np = np.array([data.Pclass, data.Fare, data.Age, data.Sex]).transpose()
dataY_np = np.array([data.Survived]).transpose()

res = np.where(np.isnan(dataX_np))

dataX_np = np.delete(dataX_np, res[0], axis=0)
dataY_np = np.delete(dataY_np, res[0], axis=0)

clf = DecisionTreeClassifier(random_state=241)

clf.fit(dataX_np, dataY_np)

importances = clf.feature_importances_.transpose()

print(importances)
# FARE & SEX => Эти признаки являются наиболее информативными,
# так как при крушении корабля в первую очередь спасали женщин и пассажиров первого класса,
# где ехали пассажиры с наиболее дорогими билетами


