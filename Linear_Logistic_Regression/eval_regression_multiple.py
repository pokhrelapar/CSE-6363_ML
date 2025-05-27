# evaluate regression model multiple

from LinearRegression import LinearRegression

from sklearn import datasets
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt

#load iris dataset
iris_dtst = datasets.load_iris()

X = iris_dtst.data[:, :2] #  sepal length and sepal width
y = iris_dtst.data[:, 2:] # predict petal length and petal width

#split dataset
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

#init and load model
lr_model = LinearRegression()
lr_model.load("lr_model_multiple_params.npz")

#eval model
mse_error = lr_model.score(x_test,y_test)

print("MSE Error for Model Multiple: \n", mse_error)



