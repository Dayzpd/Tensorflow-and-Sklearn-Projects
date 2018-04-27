from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

import time

def without_pca(x_train, x_test, y_train, y_test):
	rfc = RandomForestClassifier(n_jobs=-1, n_estimators=100)
	start = time.time()
	rfc.fit(x_train, y_train)
	end = time.time()
	print('Training time for Random Forest without PCA: ' + str(end - start) + ' seconds.')
	y_pred = rfc.predict(x_test)
	acc_score = accuracy_score(y_test, y_pred)
	print('Without PCA Accuracy Score: ' + str(acc_score))
	joblib.dump(rfc, 'RFC_with_PCA.pkl')
	
def with_pca(x_train, x_test, y_train, y_test):
	rfc = RandomForestClassifier(n_jobs=-1, n_estimators=100)
	pca = PCA(n_components=0.95, svd_solver='full')
	x_reduced = pca.fit_transform(x_train)
	start = time.time()
	rfc.fit(x_reduced, y_train)
	end = time.time()
	print('Training time for Random Forest with PCA: ' + str(end - start) + ' seconds.')
	x_test_reduced = pca.transform(x_test)
	y_pred = rfc.predict(x_test_reduced)
	acc_score = accuracy_score(y_test, y_pred)
	print('With PCA Accuracy Score: ' + str(acc_score))
	joblib.dump(rfc, 'RFC_without_PCA.pkl')

if __name__ == '__main__':
	mnist = fetch_mldata('MNIST original')

	x = mnist.data
	y = mnist.target

	x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85714, random_state=42)
	
	with_pca(x_train, x_test, y_train, y_test)
	
	without_pca(x_train, x_test, y_train, y_test)
	
	
	
	
	