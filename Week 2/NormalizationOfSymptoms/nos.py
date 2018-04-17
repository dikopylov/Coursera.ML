from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import pandas

data_train = pandas.read_csv('../perceptron-train.csv', header=None)
data_test = pandas.read_csv('../perceptron-test.csv', header=None)

dataX_train = data_train.drop([0], axis=1)
dataY_train = data_train.drop([1, 2], axis=1)

dataX_test = data_test.drop([0], axis=1)
dataY_test = data_test.drop([1, 2], axis=1)

# Обучите персептрон
clf = Perceptron(random_state=241)
clf.fit(dataX_train, dataY_train)
#  построение прогнозов
predictions_dataY_test = clf.predict(dataX_test)

# В качестве метрики качества мы будем использовать долю верных ответов (accuracy)
# первым аргументом которой является вектор правильных ответов,
# а вторым — вектор ответов алгоритма.
# Подсчитайте качество (долю правильно классифицированных объектов, accuracy)
# полученного классификатора на тестовой выборке
acc_before = accuracy_score(dataY_test, predictions_dataY_test)

# стандартизации признаков
# Нормализуйте обучающую и тестовую выборку
scaler = StandardScaler()

# находит параметры нормализации (средние и дисперсии каждого признака) по выборке,
# и сразу же делает нормализацию выборки с использованием этих параметров
dataX_train_scaled = scaler.fit_transform(dataX_train)

# делает нормализацию на основе уже найденных параметров
dataX_test_scaled = scaler.transform(dataX_test)

clf_scale = Perceptron(random_state=241)
clf_scale.fit(dataX_train_scaled, dataY_train)
#  построение прогнозов
predictions_dataY_scaled_test = clf_scale.predict(dataX_test_scaled)

acc_after = accuracy_score(dataY_test, predictions_dataY_scaled_test)

print(acc_after-acc_before)

file = open("difference_accuracy.txt", "w")
file.write(str(round(acc_after-acc_before, 3)))
file.close()