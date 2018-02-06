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

iris = datasets.load_iris()

x = iris.data[:, (2,3)]
y = iris.target

x_bias = np.c_[np.ones([len(x), 1]), x]

print(x_bias)