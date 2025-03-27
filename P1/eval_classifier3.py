from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from LogisticRegression import LogisticRegression
import numpy as np


iris_dtst = datasets.load_iris()
X = iris_dtst.data  # All features
y = iris_dtst.target

scaler = StandardScaler()
X = scaler.fit_transform(X)


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

lg_model = LogisticRegression()
lg_model.load("lg_model_3_params.npz") 


accuracy = lg_model.score(x_test, y_test)

# Print the accuracy
print(f"Accuracy on Test : {accuracy * 100:.2f}%")
