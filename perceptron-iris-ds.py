import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

iris = load_iris()
iris_data = iris['data']

X = iris_data[:, (2,3)] #petal length, petal width
y = iris['target']

clf = Perceptron()
clf.fit(X, y)

y_pred = clf.predict([[2, 0.5]])
print(y_pred)