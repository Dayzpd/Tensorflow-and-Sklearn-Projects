import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets
from matplotlib.colors import ListedColormap
custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])

#Targets: 
# 0 -> Setosa | Example: Performing (iris.target == 0).astype(np.int) will classify all Setosas as 1 and Versicolors and Virginicas as 0.
# 1 -> Versicolor | Example: Performing (iris.target == 0).astype(np.int) will classify all Versicolors as 1 and Setosas and Virginicas as 0. 
# 2 -> Virginica | Example: Performing (iris.target == 0).astype(np.int) will classify all Virginicas as 1 and Versicolors and Setosas as 0.

# Column feature names for iris.data
# 0 -> sepal length (cm) | Example: iris.data[:, 0:1] returns all the sepal lengths in a single column.
# 1 -> sepal width (cm) | Example: iris.data[:, 1:2] returns all the sepal widths in a single column.
# 2 -> petal length (cm) | Example: iris.data[:, 2:3] returns all the petal lengths in a single column.
# 3 -> petal width (cm) | Example: iris.data[:, 3:] returns all the petal widths in a single column.
# Furthermore, you can get multiple columns using: iris.data[:, (2,3)] which in this case, will return all the rows 
# with petal length in the first column (col=0) and petal width in the second column (col=1).

def one_hot_vector(y):
	'''
	The target data is already in the form 0,1,2 for their respective types of iris.
	This method will turn each index into a one hot vector.
	0 -> Setosa | Will convert to [1,0,0]
	1 -> Versicolor | Will convert to [0,1,0]
	2 -> Virginica | Will convert to [0,0,1]
	'''
	# the max() function will return the highest value in the array; however, we need to add 1 b/c the classes start at 0.
	num_of_classes = y.max() + 1
	num_of_features = len(y)
	
	# Preallocate an array of zeros of size [# of features, # of classes]
	one_hot = np.zeros((num_of_features, num_of_classes))
	
	# np.arange(num_of_features): This function creates an array of values from 0 to num_of features
	# Essentially, this afformentioned array is paired with the y target values.
	# What will happen is that np.arange(num_of_features) references the row and y (being 0,1, or 2)
	# references the column. For all the rows and columns referenced, the zero at a given specified
	# place will be assigned a 1. This will result in our one_hot matrix.
	one_hot[np.arange(num_of_features), y] = 1

	return one_hot
	
def softmax(softmax_score):
	'''
	The softmax sum takes in the softmax score and returns the probability.
	'''
	exps = np.exp(softmax_score)
	exp_sum = np.sum(exps, axis=1, keepdims=True)
	return exps / exp_sum

def softmax_single(softmax_score):
	'''
	The softmax sum takes in the softmax score and returns the probability.
	'''
	exps = np.exp(softmax_score)
	exp_sum = np.sum(exps)
	return exps / exp_sum
	
def run():	
	iris = datasets.load_iris()

	x = iris.data[:, (2,3)]
	y = iris.target

	# Adding a bias term
	x_bias = np.c_[np.ones([len(x), 1]), x]

	# Setting numpy's random seed
	np.random.seed(42)

	# The following code will split up our datasets into multiple categories:
	# Training Set: The data we will use to train our model
	# Validation Set: The data that we will use to evaluate the model's fit of the training set whilst tuning the weights (hyperparameters).
	# Test Set: The data we will use to evaluate the final model.
	test_ratio = .2
	validation_ratio = .2
	total_size = len(x_bias)

	test_size = int(total_size * test_ratio)
	validation_size = int(total_size * validation_ratio)
	train_size = total_size - test_size - validation_size

	# Given the total size, rand_perm will result in an array full of n members of the Naturals such that n is on the interval [1, toal_size].
	# Also notice, that since this is a random permutation, these numbers will be shuffled and (in a sense) predetermined by the random seed.
	rand_perm = np.random.permutation(total_size)

	# x_train pulls from the x_bias matrix so it will have three columns:
	# The first column (col=0) is full of 1's due to the line above "Adding a bias term"
	# The second column (col=1) represents the petal length.
	# The third column (col=2) represents the petal width.
	x_train = x_bias[rand_perm[:train_size]]

	# y_train pulls from the original y array which represents the target set. 
	# Given that numpy's random seed was set, it is in the same order as the x_train because it uses the rand_perm array to order the values.
	y_train = y[rand_perm[:train_size]]
	y_train_one_hot = one_hot_vector(y_train)

	# The code on the book's Github doesn't work in this situation (not surprising... ain't the first time...)
	# At least in this case, we already got indexes 0 to the train size, so now we want to get all the indices from the train_size
	# to the validation size.
	x_validation = x_bias[rand_perm[train_size:(train_size + validation_size)]]
	y_validation = y[rand_perm[train_size:(train_size + validation_size)]]
	y_validation_one_hot = one_hot_vector(y_validation)

	# Self-explanatory
	x_test = x_bias[rand_perm[(train_size + validation_size):]]
	y_test = y[rand_perm[(train_size + validation_size):]]
	y_test_one_hot = one_hot_vector(y_test)

	# Two features and a bias term = 3
	num_of_training_inputs = x_train.shape[1]

	# y_train is an array of many 0s, 1s, and 2s. np.unique, returns the unique values of the array ([0,1,2]). The length is 3.
	num_of_output_classes = len(np.unique(y_train))

	# Hyperparameters
	learning_rate = .01
	num_of_iterations = 5001
	epsilon = 1e-7
	alpha = .1
	best_loss = np.infty
	
	# Initial Theta prior to taking the gradient steps (theta is our weight which we are trying to optimize)
	theta = np.random.randn(num_of_training_inputs, num_of_output_classes)
	
	for i in range(num_of_iterations):
		# Get softmax score of feature x for each class
		softmax_score = x_train.dot(theta)
		
		# Get probability from softmax function
		y_proba = softmax(softmax_score)
		
		# Calculate loss using the entropy cost function
		entropy_loss = -np.mean(np.sum((y_train_one_hot) * (np.log(y_proba + epsilon)), axis=1))
		
		# l2 regularization
		l2_loss = 1/2 * np.sum(np.square(theta[1:]))
		
		loss = entropy_loss + alpha * l2_loss
		
		# Calculate error
		error = y_proba - y_train_one_hot
		
		# Calculate the gradient
		gradients = (1 / train_size) * (x_train.T).dot(error) + np.r_[np.zeros([1, num_of_output_classes]), alpha * theta[1:]]
		
		# Calculate the new theta
		theta = theta - (learning_rate * gradients)
		
		# Validate model using validation data
		softmax_score = x_validation.dot(theta)
		y_proba = softmax(softmax_score)
		# Calculate loss using the entropy cost function
		entropy_loss = -np.mean(np.sum((y_validation_one_hot) * (np.log(y_proba + epsilon)), axis=1))
		# l2 regularization
		l2_loss = 1/2 * np.sum(np.square(theta[1:]))
		loss = entropy_loss + alpha * l2_loss		
		
		# Output updates every 500 iterations
		if i % 500 == 0:
			print('Iteration Number: ' + str(i))
			print('Loss: ' + str(loss))
			
		# Early stopping
		if loss < best_loss:
			best_loss = loss
		else:
			print('Iteration Number: ' + str(iteration - 1) + ' | Best Loss: ' + str(best_loss))
			print('Iteration Number: ' + str(iteration) + ' | Loss: ' + str(loss) +  " Early stopping!")
			break
		
	print(theta)
		
	softmax_score = x_validation.dot(theta)
	y_proba = softmax(softmax_score)
	y_predict = np.argmax(y_proba, axis=1)
	
	accuracy_score = np.mean(y_predict == y_validation)
	print('Accuracy: ' + str(accuracy_score))
	
	# Contour plot
	x0, x1 = np.meshgrid(np.linspace(0, 8, 500).reshape(-1, 1), np.linspace(0, 3.5, 200).reshape(-1, 1))
	x_new = np.c_[np.ravel(x0), np.ravel(x1)]
	x_new_with_bias = np.c_[np.ones([len(x_new), 1]), x_new]

	logits = x_new_with_bias.dot(theta)
	y_proba = softmax(logits)
	y_predict = np.argmax(y_proba, axis=1)
	
	zz1 = y_proba[:, 1].reshape(x0.shape)
	zz = y_predict.reshape(x0.shape)

	plt.figure(figsize=(10, 4))
	plt.plot(x[y==2, 0], x[y==2, 1], "g^", label="Iris-Virginica")
	plt.plot(x[y==1, 0], x[y==1, 1], "bs", label="Iris-Versicolor")
	plt.plot(x[y==0, 0], x[y==0, 1], "yo", label="Iris-Setosa")

	plt.contourf(x0, x1, zz, cmap=custom_cmap, linewidth=5)
	contour = plt.contour(x0, x1, zz1, cmap=plt.cm.brg)
	plt.clabel(contour, inline=1, fontsize=12)
	plt.xlabel("Petal length", fontsize=14)
	plt.ylabel("Petal width", fontsize=14)
	plt.legend(loc="center left", fontsize=14)
	plt.axis([0, 8, 0, 3.5])

	length = float(input('Enter a petal length you would like to estimate: '))
	width = float(input('Enter a petal width you would like to estimate: '))
	soft_score = np.array([1,length,width]).dot(theta)
	prob = softmax_single(soft_score)
	prediction = (np.argmax(prob))
	if(prediction == 0):
		print('The prediction is Setosa.')
		plt.plot(length, width, "mo")
	elif(prediction == 1):
		print('The prediction is Versicolor.')
		plt.plot(length, width, "ms")
	else:
		print('The prediction is Virginica.')
		plt.plot(length, width, "m^")

	plt.show()
		
if __name__ == '__main__':
	run()
