import pandas

data = pandas.read_csv('titanic.csv', index_col='PassengerId')

data_average = round(data.Age.mean(), 2)
data_median = round(data.Age.median())
print(data_average)
print(data_median)

file = open("4.txt", "w")
file.write(str(data_average) + " " + str(data_median))
file.close()