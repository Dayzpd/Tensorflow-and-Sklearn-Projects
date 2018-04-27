import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import time, math
from datetime import timedelta
import os

old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

from tensorflow.examples.tutorials.mnist import input_data

global path
path = 'G:\My Drive\School\Spring 2018\Tensorflow and Sklearn Projects\CNN\cnn_mnist\model.ckpt'

def new_weights(shape):
	return tf.Variable(tf.truncated_normal(shape, stddev=0.05))
	
def new_biases(length):
	return tf.Variable(tf.constant(0.05, shape=[length]))
	
def new_layer(prev_layer, num_channels, kernel_size, num_kernels, pooling=True):
	# Each new layer is constructed according to:
	# https://www.tensorflow.org/api_docs/python/tf/layers/conv2d
	# shape is determined by:
	# https://www.tensorflow.org/api_docs/python/tf/nn/conv2d
	shape = [kernel_size, kernel_size, num_channels, num_kernels]
	weights = new_weights(shape)
	biases = new_biases(num_kernels)
	
	layer = tf.nn.conv2d(input=prev_layer, filter=weights, strides=[1,1,1,1], padding='SAME')

	layer = layer + biases
	
	if pooling:
		layer = tf.nn.max_pool(value=layer, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
		
	layer = tf.nn.relu(layer)
	
	return layer, weights
	
def flatten_layer(layer):
	# Converts 4D tensor into 2D tensor
	
	layer_shape = layer.get_shape()
	
	num_features = np.array(layer_shape[1:4], dtype=int).prod()
	
	flat_layer = tf.reshape(layer, [-1, num_features])
	
	return flat_layer, num_features
	
def fully_connected_layer(layer, num_inputs, num_outputs, relu=True):
	weights = new_weights(shape=[num_inputs, num_outputs])
	biases = new_biases(length=num_outputs)
	
	fc_layer = tf.matmul(layer, weights) + biases

	if relu:
		fc_layer = tf.nn.relu(fc_layer)
		
	return fc_layer
	
def graph(img_size, img_size_linear, num_channels, num_classes, predict=False):

	# Conv Layer 1
	kernel_size1 = 5
	num_kernels1 = 16
	
	# Conv Layer 2
	kernel_size2 = 5
	num_kernels2 = 36
	
	#Fully-Connected Layer
	fc_size = 128
	
	tf.reset_default_graph()

	x = tf.placeholder(tf.float32, shape=[None, img_size_linear], name='x')
	x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
	
	y = tf.placeholder(tf.float32, shape=[None, 10], name='y')
	y_true_cls = tf.argmax(y, axis=1)
	
	layer1, weights1 = new_layer(x_image, num_channels, kernel_size1, num_kernels1)
	
	layer2, weights2 = new_layer(layer1, num_kernels1, kernel_size2, num_kernels2)
	
	flat_layer, num_features = flatten_layer(layer2)
	
	fc_layer1 = fully_connected_layer(flat_layer, num_features, fc_size)

	fc_layer2 = fully_connected_layer(fc_layer1, fc_size, num_classes, relu=False)
	
	y_pred = tf.nn.softmax(fc_layer2)
	
	y_pred_cls = tf.argmax(y_pred, axis=1)
	
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=fc_layer2, labels=y)
	
	cost = tf.reduce_mean(cross_entropy)
	
	optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
	
	correct_prediction = tf.equal(y_pred_cls, y_true_cls)
	
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	
	saver = tf.train.Saver()
	
	if predict:
		return x, y_pred_cls
	else:
		return x, y, optimizer, accuracy, saver
	
def train(data, predict=False):
	img_size = 28 #28 width and 28 length
	img_size_linear = img_size ** 2 #size of image when stored in a 1D array
	img_shape = (img_size, img_size) #tuple of size
	num_channels = 1 #grayscale
	num_classes = 10 #0 though 9
	
	if predict:
		x, y_pred_cls = graph(img_size, img_size_linear, num_channels, num_classes, predict=True)
		
		test_data = data.test.images
		test_labels = data.test.labels
		
		with tf.Session() as session:
			predict_x = np.asarray((test_data[100])).reshape((-1, img_size_linear))
			saver = tf.train.import_meta_graph('model.meta')
			saver.restore(session, tf.train.latest_checkpoint('./'))
			classification = session.run([y_pred_cls], {x: predict_x})
			print('Prediction: ' + str(int(classification[0])))
			print('Actual: ' + str(np.argmax(test_labels[100], axis=0)))
	else:
		x, y, optimizer, accuracy, saver = graph(img_size, img_size_linear, num_channels, num_classes)
		
		num_iterations = 5001
		batch_size = 64

		session = tf.Session()
		
		session.run(tf.global_variables_initializer())
		
		start = time.time()
		
		checkpoint_path = "./model"
		final_model_path = "./model"
		
		for z in range(0,num_iterations):
			x_batch, y_true_batch = data.train.next_batch(batch_size)
			
			session.run(optimizer, feed_dict={x:x_batch, y:y_true_batch})
			
			if z % 100 == 0:
				saver.save(session, checkpoint_path)
				acc = session.run(accuracy, feed_dict={x:x_batch, y:y_true_batch})
				print('Iteration: ' + str(z + 1) + ' Accuracy: ' + str(acc))
				
		
		end = time.time()
		
		time_elapsed = end-start
		
		print('Time Elapsed: ' + str(time_elapsed))
		
		saver.save(session, final_model_path)
	
def run():
	data = input_data.read_data_sets('data/', one_hot=True)

	tf.logging.set_verbosity(old_v)
	
	data.test.cls = np.argmax(data.test.labels, axis=1)
	
	print('Training Model...')
	train(data)
	print('Loading Model and Classify instance...')
	train(data, predict=True)
	
if __name__ == '__main__':
	run()	