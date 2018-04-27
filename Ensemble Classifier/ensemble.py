from sklearn.datasets import fetch_mldata
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from scipy import stats

def run():
	mnist = fetch_mldata('MNIST original')

	x = mnist.data
	y = mnist.target

	x_train, x_test_val, y_train, y_test_val = train_test_split(x, y, train_size=0.571428, random_state=42)
	x_test, x_val, y_test, y_val = train_test_split(x_test_val, y_test_val, test_size=.5, random_state=42)
	
	rfc = RandomForestClassifier(n_estimators=1000)
	etc = ExtraTreesClassifier(n_estimators=1000)
	sgd = SGDClassifier(loss='log')
	
	#sgd = joblib.load('sgd.pkl')
	pred_sgd = sgd.predict(x_test)
	score_sgd = accuracy_score(y_test, pred_sgd)
	print("Stochastic Gradient Descent: " + str(score_sgd))	
	
	#rfc = joblib.load('random_forest.pkl')
	pred_rfc = rfc.predict(x_test)
	score2 = accuracy_score(y_test, pred_rfc)
	print("Random Forest Classifier: " + str(score2))
	
	#etc = joblib.load('extra_trees.pkl')
	pred_etc = etc.predict(x_test)
	score3 = accuracy_score(y_test, pred_etc)
	print("Extra Trees: " + str(score3))
	
	print('sgd')
	val_prob_sgd = sgd.predict_proba(x_test)
	print('rfc')
	val_prob_rfc = rfc.predict_proba(x_test)
	print('etc')
	val_prob_etc = etc.predict_proba(x_test)

	length = len(x_test)
	predictions = []
	
	for x in range(0,length):
		predictions.append(stats.mode([np.argmax(val_prob_sgd[x]), np.argmax(val_prob_rfc[x]), np.argmax(val_prob_etc[x])]))
		
	print('Ensemble: ' + str(accuracy_score(y_test, predictions)))
	
if __name__ == '__main__':
	run()