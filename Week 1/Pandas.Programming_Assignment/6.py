import pandas
import re

data = pandas.read_csv('titanic.csv')

data_female = data[data.Sex == 'female']

data_full_name = data_female.Name.str.split('\n')

data_first_name = []

for var in data_full_name[:-1]:
    out = ''
    for ch in var[0]:
        if (ord(ch) > 64 and ord(ch) < 91) or (ord(ch) > 96 and ord(ch) < 123) or ord(ch) == 32:
            out += ch

    match = re.search(r'Mrs.', out)
    if match:
        data_first_name.append(out[match.end()-1:].split(' ')[1])
    else:
        match1 = re.search(r'Miss.', out)
        if match1:
            data_first_name.append(out[match1.end()-1:].split(' ')[1])

data_popular_name = pandas.Series(data_first_name)

print(data_popular_name.value_counts()[:1])

file = open("6.txt", "w")
file.write(str(data_popular_name.value_counts()[:1]).split('\n')[0])
file.close()
