import pandas

data = pandas.read_csv('titanic.csv', index_col='PassengerId')

data_1Pclass = data[data.Pclass == 1]

data_1Pclass_percent = round((len(data_1Pclass) / len(data)) * 100, 2)

print(data_1Pclass_percent)

file = open("3.txt", "w")
file.write(str(data_1Pclass_percent))
file.close()