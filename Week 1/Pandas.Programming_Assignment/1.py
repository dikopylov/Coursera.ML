import pandas

data = pandas.read_csv('titanic.csv', index_col='PassengerId')

print(data['Sex'].value_counts())

file = open("1.txt", "w")
file.write(str(data['Sex'].value_counts()))
file.close()