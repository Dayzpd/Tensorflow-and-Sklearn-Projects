from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

xs = np.array([1,2,3,4,5,6], dtype=np.float64)
ys = np.array([5,4,6,5,6,7], dtype=np.float64)
	
def slope(xs,ys):
	m = ((mean(xs) * mean(ys)) - mean(xs*ys)) / (mean(xs)**2 - mean(xs**2))
	return m
	
def y_intercept(xs, ys, m):
	b = mean(ys) - m * mean(xs)
	return b
	
def squared_error(ys_orig, ys_line):
	return sum((ys_line - ys_orig)**2)
	
def coeff_determ(ys_orig, ys_line):
	# Essentially, you have len(y) data points, and you are creating a horizontal line for the mean os the ys
	y_mean_line = [mean(ys_orig) for y in ys_orig]

	sqrd_err_reg = squared_error(ys_orig, ys_line)
	sqrd_err_y_mean = squared_error(ys_orig, y_mean_line)
	return 1 - (sqrd_err_reg/sqrd_err_y_mean)
	
m = slope(xs,ys)

b = y_intercept(xs, ys, m)

regression_line = [ (m*x)+b for x in xs ]

predict_x = 8
predict_y = m*predict_x + b

r_squared = coeff_determ(ys, regression_line)
print(r_squared)

plt.scatter(xs,ys)
plt.scatter(predict_x, predict_y, color="g")
plt.plot(xs, regression_line)
plt.show()

