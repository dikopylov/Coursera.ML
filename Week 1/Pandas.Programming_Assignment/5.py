import pandas

data = pandas.read_csv('titanic.csv', index_col='PassengerId')

data_corr = round(data.corr()['SibSp']['Parch'], 2)

print(data_corr)
file = open("5.txt", "w")
file.write(str(data_corr))
file.close()