from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score
from datetime import datetime
import os

def random_batch(x, y, size):
	perm = np.random.randint(0, len(x), size)
	x_batch = x[perm]
	y_batch = y[perm]
	return x_batch, y_batch
	
def log_dir(prefix=""):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs"
    if prefix:
        prefix += "-"
    name = prefix + "run-" + now
    return "{}/{}/".format(root_logdir, name)
	
def logistic_regression(x, y, num_classes = 2, epsilon = 1e-7, learning_rate = .01):
	with tf.name_scope('logistic_regression'):
		with tf.name_scope('model'):
			# theta will be 3 rows and 1 col.
			# the logits will be come from multiplying the matrices: x and theta. 
			# notice, x is of size 1x3 and theta 3x1. Thus, they can be multiplied (b/c 3=3).
			theta = tf.Variable(tf.random_uniform([num_classes + 1, 1], -1.0, 1.0, seed=42), name='theta')
			logits = tf.matmul(x, theta, name='logits')
			y_probability = 1 / (1 + tf.exp(-logits))
		with tf.name_scope('train'):
			loss = - tf.reduce_mean(y * tf.log(y_probability + epsilon) + (1 - y) * tf.log(1-y_probability + epsilon))
			# initialize optimizer to minimize the loss
			optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
			training_op = optimizer.minimize(loss)
			loss_summary = tf.summary.scalar('log_loss', loss)
		with tf.name_scope('init'):
			init = tf.global_variables_initializer()
		with tf.name_scope('save'):
			saver = tf.train.Saver()
		
	return y_probability, loss, training_op, loss_summary, init, saver

def run():
	num_moons = 1000
	num_classes = 2 # moon or not moon
	logdir = log_dir("logreg")
	
	data, target = make_moons(n_samples=num_moons, noise=.1, shuffle=True, random_state=42)

	plt.plot(data[target==1,0], data[target==1,1], 'gh', label='Moon')
	plt.plot(data[target==0,0], data[target==0,1], 'r^', label='Not Moon')
	plt.legend()
	plt.show()
	
	x_bias = np.c_[np.ones((num_moons, 1)), data]
	y_col_vector = target.reshape(-1,1)
	
	x_test = x_bias[0:num_moons - 200]
	x_train = x_bias[num_moons - 200:]
	y_test = y_col_vector[0:num_moons - 200]
	y_train = y_col_vector[num_moons - 200:]
	
	tf.reset_default_graph()
	
	x = tf.placeholder(tf.float32, shape=(None, num_classes + 1), name='x') # each x feature is a single row vector w/ 3 cols
	y = tf.placeholder(tf.float32, shape=(None, 1), name='y') # essentialy just a single value (1 or 0)
	
	y_probability, loss, training_op, loss_summary, init, saver = logistic_regression(x, y)

	file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
	
	n_epochs = 1000
	batch_size = 50
	n_batches = int(np.ceil((num_moons - 200)/batch_size))
	
	checkpoint_path = "./tmp/my_logreg_model.ckpt"
	checkpoint_epoch_path = checkpoint_path + ".epoch"
	final_model_path = "./my_logreg_model"
	
	with tf.Session() as sess:
		if os.path.isfile(checkpoint_epoch_path):
			# if the checkpoint file exists, restore the model and load the epoch number
			with open(checkpoint_epoch_path, "rb") as f:
				start_epoch = int(f.read())
			print("Training was interrupted. Continuing at epoch", start_epoch)
			saver.restore(sess, checkpoint_path)
		else:
			start_epoch = 0
			sess.run(init)
		
		for epoch in range(start_epoch, n_epochs):
			for batch_index in range(n_batches):
				x_batch, y_batch = random_batch(x_train, y_train, batch_size)
				sess.run(training_op, feed_dict={x: x_batch, y: y_batch})
			loss_val, summary_str = sess.run([loss, loss_summary], feed_dict={x: x_test, y: y_test})
			file_writer.add_summary(summary_str, epoch)
			if epoch % 500 == 0:
				print("Epoch:", epoch, "\tLoss:", loss_val)
				saver.save(sess, checkpoint_path)
				with open(checkpoint_epoch_path, "wb") as f:
					f.write(b"%d" % (epoch + 1))
				
		saver.save(sess, final_model_path)
		y_proba_val = y_probability.eval(feed_dict={x: x_test, y: y_test})
		os.remove(checkpoint_epoch_path)
		y_prediction = (y_proba_val >= .7)
	
		print('precision: ' + str(precision_score(y_test, y_prediction)))
		print('recall: ' + str(recall_score(y_test, y_prediction)))
		
	
	
if __name__ == '__main__':
	run()