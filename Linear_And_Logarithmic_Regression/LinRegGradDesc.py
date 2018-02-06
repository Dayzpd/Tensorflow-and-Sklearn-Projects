import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

def compute_error(b, m, points):
	total_err = 0
	for i in range(0, len(points)):
		x = points[i,0]
		y = points[i,1]
		total_err += (y - (m * x + b))**2
	total_err = total_err / np.float(len(points))
	return total_err
	
def gradient_step(b_current, m_current, points, learning_rate):
	b_gradient = 0
	m_gradient = 0
	N = float(len(points))
	for i in range(0, len(points)):
		x = points[i,0]
		y = points[i,1]
		m_gradient += -x*(y-(m_current * x + b_current))
		b_gradient += -(y-(m_current * x + b_current))
	
	b_gradient = (2 * b_gradient) / N
	m_gradient = (2 * m_gradient) / N
	
	b_new = b_current - (learning_rate * b_gradient)
	m_new = m_current - (learning_rate * m_gradient)
	
	return [b_new, m_new]
		
def gradient_descent_runner(points, learning_rate, initial_b, initial_m, num_of_iterations):
	b = initial_b
	m = initial_m
	
	for x in range(num_of_iterations):
		b, m = gradient_step(b, m, np.array(points), learning_rate)
	
	return [b,m]

def run():
	points = np.genfromtxt('data.csv', delimiter=',')
	#hyperparameter
	learning_rate = .0001
	#y=mx+b
	initial_b = 0
	initial_m = 0
	num_of_iterations = 1000
	[b, m] = gradient_descent_runner(points, learning_rate, initial_b, initial_m, num_of_iterations)
	final_err = compute_error(b, m, points)
	print('y-intercept: ' + str(b))
	print('slope: ' + str(m))
	print('final error: ' + str(final_err))
	regression_line = [m*x+b for x in points[:,0]]
	xs = [x for x in points[:,0]]
	ys = [y for y in points[:,1]]
	
	plt.scatter(xs,ys, color="r")
	plt.plot(xs, regression_line)
	plt.show()
	
	
if __name__ == '__main__':
	run()