import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def model(data, num_nodes_l1, num_nodes_l2, num_nodes_l3, num_classes):
	h1_layer = {'weights' : tf.Variable(tf.random_normal([784, num_nodes_l1])), 'biases' : tf.Variable(tf.random_normal([num_nodes_l1]))}

	h2_layer = {'weights' : tf.Variable(tf.random_normal([num_nodes_l1, num_nodes_l2])), 'biases' : tf.Variable(tf.random_normal([num_nodes_l2]))}
	
	h3_layer = {'weights' : tf.Variable(tf.random_normal([num_nodes_l2, num_nodes_l3])), 'biases' : tf.Variable(tf.random_normal([num_nodes_l3]))}
	
	output_layer = {'weights' : tf.Variable(tf.random_normal([num_nodes_l3, num_classes])), 'biases' : tf.Variable(tf.random_normal([num_classes]))}
	
	l1 = tf.add(tf.matmul(data, h1_layer['weights']), h1_layer['biases'])
	l1 = tf.nn.relu(l1)
	
	l2 = tf.add(tf.matmul(l1, h2_layer['weights']), h2_layer['biases'])
	l2 = tf.nn.relu(l2)
	
	l3 = tf.add(tf.matmul(l2, h3_layer['weights']), h3_layer['biases'])
	l3 = tf.nn.relu(l3)
	
	output = tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])
	
	return output
	
def train(mnist, x, y, num_nodes_l1, num_nodes_l2, num_nodes_l3, num_classes, batch_size):
	prediction = model(x, num_nodes_l1, num_nodes_l2, num_nodes_l3, num_classes)
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
	optimize = tf.train.AdamOptimizer().minimize(loss)
	
	epochs = 10
	
	with tf.Session() as session:
		session.run(tf.global_variables_initializer())
		
		for e in range(1, epochs):
			epoch_loss = 0
			for _ in range(int(mnist.train.num_examples/batch_size)):
				ep_x,ep_y = mnist.train.next_batch(batch_size)
				_, c = session.run([optimize, loss], feed_dict={x : ep_x, y : ep_y})
				epoch_loss += c
				
			print('Epoch #: ' + str(e) + ' | Loss: ' + str(epoch_loss))
			
		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y,1))
		
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print('Accuracy: ' + str(accuracy.eval({x : mnist.test.images, y : mnist.test.labels})))
	
def run():
	mnist = input_data.read_data_sets('/tmp/data', one_hot=True)

	num_nodes_l1 = 500
	num_nodes_l2 = 500
	num_nodes_l3 = 500

	num_classes = 10 # Numbers 0-9
	batch_size = 100 


	x = tf.placeholder('float', [None, 784])
	y = tf.placeholder('float', [None, 10])
	
	train(mnist, x, y, num_nodes_l1, num_nodes_l2, num_nodes_l3, num_classes, batch_size)

if __name__ == '__main__':
	run()