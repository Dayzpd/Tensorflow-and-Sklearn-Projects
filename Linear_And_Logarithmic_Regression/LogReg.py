import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets

#Targets: 
# 0 -> Setosa | Example: Performing (iris.target == 0).astype(np.int) will classify all Setosas as 1 and Versicolors and Virginicas as 0.
# 1 -> Versicolor | Example: Performing (iris.target == 0).astype(np.int) will classify all Versicolors as 1 and Setosas and Virginicas as 0. 
# 2 -> Virginica | Example: Performing (iris.target == 0).astype(np.int) will classify all Virginicas as 1 and Versicolors and Setosas as 0.

# Column feature names for iris.data
# 0 -> sepal length (cm) | Example: iris.data[:, 0:1] returns all the sepal lengths in a single column.
# 1 -> sepal width (cm) | Example: iris.data[:, 1:2] returns all the sepal widths in a single column.
# 2 -> petal length (cm) | Example: iris.data[:, 2:3] returns all the petal lengths in a single column.
# 3 -> petal width (cm) | Example: iris.data[:, 3:] returns all the petal widths in a single column.

iris = datasets.load_iris()

x = iris.data[:,3:] # Get all the petal widths in a single column
y = ((iris.target == 2).astype(np.int)) # Virginicas = 1, All others = 0

log_reg = linear_model.LogisticRegression()
log_reg.fit(x,y) # Self explanatory

# As far as I can tell, what np.linspace does is creates 100 Real numbers between and including 0 and 3. 
# Furthermore, reshape(-1,1) turns this array into a single column of all 1000 values.
x_new = (np.linspace(0, 3, 1000).reshape(-1, 1))

# This will predict the estimated probabilities for flowers with petal widths from 0 to 3 cm (proba = probability).
# Furthermore, these probability values are all between 0 and 1. Remember, for a feature x, if x >= .5, then it's a Virginica.
# Else if x < .5, then it's not a Virginica. 
# Lastly, y_proba is structured as 100 rows and 2 columns. The left column (col=0) represents the probability it's not Virginica.
# And the right column (col=1) represents the probability it is a Virginica.
y_proba = log_reg.predict_proba(x_new)

plt.plot(x_new, y_proba[:,1], color="g", label="Virginica")
plt.plot(x_new, y_proba[:,0], color="b", label="Not Virginica")

predict_x = float(input('Enter a petal width you would like to estimate: '))
predict_y = log_reg.predict_proba(predict_x)

if(predict_y[0,1] >= 0.5):
	plt.scatter(predict_x, predict_y[0,1], color="r")
	print('A petal width of ' + str(predict_x) + ' has a probability of ' + str(predict_y[0,1]) + ' being a Virginica.')
	print('Thus, it is of class ' + str(log_reg.predict(predict_x)) + '.')
else:
	plt.scatter(predict_x, predict_y[0,0], color="r")
	print('A petal width of ' + str(predict_x) + ' has a probability of ' + str(predict_y[0,0]) + ' not being a Virginica.')
	print('Thus, it is of class ' + str(log_reg.predict(predict_x)) + '.')
plt.show()
