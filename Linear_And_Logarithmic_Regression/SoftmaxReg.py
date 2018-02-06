import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets
from matplotlib.colors import ListedColormap

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

custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])

iris = datasets.load_iris()

x = (iris.data[:, (2,3)])
y = iris.target

# The lbfgs solver is able to handle multinomial loss
# C=10 specifies the regularization strength (smaller values specify stronger regularization)
softmax_reg = linear_model.LogisticRegression(multi_class='multinomial', solver='lbfgs', C=10)
softmax_reg.fit(x,y)

x0, x1 = np.meshgrid(np.linspace(0, 7, 1000).reshape(-1, 1), np.linspace(0, 3.5, 1000).reshape(-1, 1))
x_new = np.c_[np.ravel(x0), np.ravel(x1)]

y_proba = softmax_reg.predict_proba(x_new)
y_predict = softmax_reg.predict(x_new)

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
plt.axis([0, 7, 0, 3.5])

length = float(input('Enter a petal length you would like to estimate: '))
width = float(input('Enter a petal width you would like to estimate: '))

prediction = softmax_reg.predict_proba([[length,width]])

if(prediction[0,0] > prediction[0,1] and prediction[0,0] > prediction[0,2]):
	print('The prediction is Setosa.')
	plt.plot(length, width, "mo")
elif(prediction[0,1] > prediction[0,0] and prediction[0,1] > prediction[0,2]):
	print('The prediction is Versicolor.')
	plt.plot(length, width, "ms")
else:
	print('The prediction is Virginica.')
	plt.plot(length, width, "m^")

plt.show()