from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import scale
from sklearn.datasets import load_boston
import numpy as np

data = load_boston()
data['data'] = scale(data['data'])

p = np.linspace(1, 10, 200)
# генератор разбиений
kf = KFold(n_splits=5, shuffle=True, random_state=42)

file = open("grade.txt", "w")

for i in p:
    model = KNeighborsRegressor(metric='minkowski', n_neighbors=5,
                                weights='distance', p=i)
    # Вычислить качество на всех разбиениях
    grade = cross_val_score(estimator=model, X=data['data'], y=data['target'],
                            cv=kf, scoring='neg_mean_squared_error')

    file.write(str(i) + ': ' + str(round(grade.mean(), 1)) + '\n')

file.close()

