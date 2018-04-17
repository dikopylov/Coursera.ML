import pandas

data = pandas.read_csv('titanic.csv', index_col='PassengerId')

data_survived = data[data.Survived == 1]

data_survived_percent = round((len(data_survived) / len(data)) * 100, 2)

print(data_survived_percent)

file = open("2.txt", "w")
file.write(str(data_survived_percent))
file.close()