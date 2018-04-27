import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

mnist = fetch_mldata('MNIST original')

x = mnist['data']
y = mnist['target']

x_test = x[60000:]
y_test = y[60000:]

scaler = StandardScaler()
x_test_scaled = scaler.fit_transform(x_test.astype(np.float32))

svm_clf = joblib.load('svm_mnist.pkl')

score = svm_clf.score(x_test_scaled, y_test)

print(score)