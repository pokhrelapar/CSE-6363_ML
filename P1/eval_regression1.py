# evaluate regression model1

from LinearRegression import LinearRegression

from sklearn import datasets
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib as plt

#load iris dataset
iris_dtst = datasets.load_iris()

#extract inputs and targets
X = iris_dtst.data[:, :2] # sepal length and sepal width
y = iris_dtst.data[:,2].reshape(-1,1)   # predict petal length

#split dataset
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

#init and load model
lr_model = LinearRegression()
lr_model.load("lr_model_1_params.npz")

#eval model
mse_error = lr_model.score(x_test,y_test)

print("MSE Error for Model 1: \n", mse_error)



