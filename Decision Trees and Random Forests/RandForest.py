from sklearn import datasets
from sklearn.base import clone
from scipy.stats import mode
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
import numpy as np

def run():
	# Generate moon dataset
	x,y = datasets.make_moons(n_samples=100000, noise=0.4, random_state=42)

	# Split into training set
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=42)

	#Specifying hyperparameters
	params = {'max_leaf_nodes':list(range(2,100)), 'min_samples_split':[2,3,4]}

	dtc = DecisionTreeClassifier(random_state=42)
	clf = GridSearchCV(dtc, params, n_jobs=-1, verbose=1)
	clf.fit(x_train, y_train)

	predictions = clf.predict(x_test)
	
	score = accuracy_score(y_test, predictions)
	
	print(score)
	
	# Number of trees in Random Forest each trained with 100 randomly selected instances
	num_trees = 10000
	num_instances = 100
	
	subsets = []
	
	# ShuffleSplit is similar to a random permutation
	shuffle = ShuffleSplit(n_splits=num_trees, test_size=len(x_train)-num_instances, random_state=42)
	
	for subset_train_index, subset_test_index in shuffle.split(x_train):
		x_train_subset = x_train[subset_train_index]
		y_train_subset = y_train[subset_train_index]
		subsets.append((x_train_subset, y_train_subset))
	
	# We obtained the optimal hyperparameterswhen we trained the Decision Tree with cross verification.
	# We then clone the optimal hyperparameters for use in fit each of the trees within the forest.
	forest = [clone(clf.best_estimator_) for _ in range(num_trees)]
	
	accuracy_scores = []
	
	for tree, (x_train_subset, y_train_subset) in zip(forest, subsets):
		tree.fit(x_train_subset, y_train_subset)
		predictions = tree.predict(x_test)
		accuracy_scores.append(accuracy_score(y_test, predictions))
		
	Y_pred = np.empty([num_trees, len(x_test)], dtype=np.uint8)

	# When predicting whether an object is a moon, we'll get a prediction from each tree.
	for tree_index, tree in enumerate(forest):
		Y_pred[tree_index] = tree.predict(x_test)
	
	# After getting all the predictions from each tree, we make our final prediction based on the most common guess (i.e. take the mode).
	y_pred_majority_votes, n_votes = mode(Y_pred, axis=0)
	
	score = accuracy_score(y_test, y_pred_majority_votes.reshape([-1]))
	
	print(score)
	
if __name__ == '__main__':
	run()