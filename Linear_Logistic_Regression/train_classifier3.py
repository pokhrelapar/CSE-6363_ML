# all features
# eval_classifier3


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mlxtend.plotting import plot_decision_regions


from LogisticRegression import LogisticRegression

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


X = iris_dtst.data # all features
y = iris_dtst.target

scaler = StandardScaler()

x_train = scaler.fit_transform(X)

# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42,)


lg_model = LogisticRegression()

lg_model.fit(x_train,y_train)

lg_model.save("lg_model_3_params.npz")




