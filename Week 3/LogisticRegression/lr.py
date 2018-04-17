from sklearn.preprocessing import scale
from sklearn.datasets import load_boston
import numpy as np
import pandas

data = pandas.read_csv('data-logistic.csv', header=None)
dataX = data.drop([0], axis=1)
dataY = data.drop([1, 2], axis=1)