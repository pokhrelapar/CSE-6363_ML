# sepal width and petal width to predict sepal length
# eval_regression4


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

X = iris_dtst.data[:, [1,3]] #  sepal width and petal width
y = iris_dtst.data[:, 0].reshape(-1,1)   # predict sepal legth

# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

lr_model = LinearRegression()

lr_model.fit(x_train,y_train)

lr_model.save("lr_model_4_params.npz")

weights = lr_model.weights
bias = lr_model.bias

print("Weights\n", weights)
print("Bias: \n", bias)



plt.plot(lr_model.loss_history)

plt.xlabel("Epoch ")
plt.ylabel("MSE Loss")

plt.title("Training Loss for Model 4")
plt.savefig("Train_Loss4.png")
plt.show()
