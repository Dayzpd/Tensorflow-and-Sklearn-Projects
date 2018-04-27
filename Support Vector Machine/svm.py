import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import reciprocal, uniform
from sklearn.externals import joblib

mnist = fetch_mldata('MNIST original')

x = mnist['data']
y = mnist['target']

x_train = x[:60000]
y_train = y[:60000]

x_test = x[60000:]
y_test = y[60000:]

np.random.seed(32)
permutation = np.random.permutation(60000)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train.astype(np.float32))
x_test_scaled = scaler.transform(x_test.astype(np.float32))

svm_clf = SVC(decision_function_shape="ovr")
svm_clf.fit(x_train_scaled, y_train)

param_distributions = {"gamma": reciprocal(0.001, 0.1), "C": uniform(1, 10)}
rnd_search_cv = RandomizedSearchCV(svm_clf, param_distributions, n_iter=10, verbose=2)
rnd_search_cv.fit(x_train_scaled, y_train)

y_pred = rnd_search_cv.best_estimator_.predict(x_train_scaled)
accuracy_score(y_train, y_pred)

y_pred = rnd_search_cv.best_estimator_.predict(x_test_scaled)
accuracy_score(y_test, y_pred)


joblib.dump(svm_clf, 'svm_mnist.pkl') 