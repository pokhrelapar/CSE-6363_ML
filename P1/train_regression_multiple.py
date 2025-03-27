# sepal length and sepal width to predict petal length and petal width
# eval_regression_multiple


from sklearn import datasets
from sklearn.model_selection import train_test_split

from LinearRegression import LinearRegression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


iris_dtst = datasets.load_iris()

'''

    df = pd.DataFrame(data= np.c_[iris_dtst['data'], iris_dtst['target']],
                        columns= iris_dtst['feature_names'] + ['target'])
                        
        'data':
            sepal length:
            sepal width: 
            petal length:
            petal width


    target_names = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    df['target_names'] = df['target'].map(target_names)

'''


X = iris_dtst.data[:, :2] #  sepal length and sepal width
y = iris_dtst.data[:, 2:]   # predict petal length and petal width


# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

lr_model = LinearRegression()

lr_model.fit(x_train,y_train)

lr_model.save("lr_model_multiple_params.npz")

weights = lr_model.weights
bias = lr_model.bias

print("Weights\n", weights)
print("\n Bias: \n", bias)



plt.plot(lr_model.loss_history)

plt.xlabel("Epoch ")
plt.ylabel("MSE Loss")

plt.title("Loss Curve for Model Multiple")
plt.savefig("Train_Loss_multiple.png")
plt.show()
