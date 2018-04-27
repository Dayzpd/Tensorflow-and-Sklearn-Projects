import numpy as np
from functools import partial
from PIL import Image
import tensorflow as tf
import requests
import os
from io import BytesIO
import matplotlib.pyplot as plt


	
def run():
	model_file = './data/tensorflow_inception_graph.pb'
	
	graph = tf.Graph()
	sess = tf.InteractiveSession(graph=graph)
	
	#Opening Inception model
	with tf.gfile.FastGFile(model_file, 'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
	t_input = tf.placeholder(np.float32, name='input') # input tensor of size 32 bits
	imagenet_mean = 117.0 # We remove this value from each image to help w/ feature learning
	t_preprocessed = tf.expand_dims(t_input - imagenet_mean, 0)
	tf.import_graph_def(graph_def, {'input' : t_preprocessed})
	
	# loading all Convolutional layers
	layers = [op.name for op in graph.get_operations() if op.type=='Conv2D' and 'import/' in op.name]
	
	# Each layer outputs features and we'll store them in feature_nums
	feature_nums = [int(graph.get_tensor_by_name(name + ':0').get_shape()[-1]) for name in layers]
	
	print('# layers: ' + str(len(layers)))
	print('# feature channels: ' + str(sum(feature_nums)))
	
	# Helper functions for TF Graph visualization
	#pylint: disable=unused-variable
	def strip_consts(graph_def, max_const_size=32):
		"""Strip large constant values from graph_def."""
		strip_def = tf.GraphDef()
		for n0 in graph_def.node:
			n = strip_def.node.add() #pylint: disable=maybe-no-member
			n.MergeFrom(n0)
			if n.op == 'Const':
				tensor = n.attr['value'].tensor
				size = len(tensor.tensor_content)
				if size > max_const_size:
					tensor.tensor_content = "<stripped %d bytes>"%size
		return strip_def
	  
	def rename_nodes(graph_def, rename_func):
		res_def = tf.GraphDef()
		for n0 in graph_def.node:
			n = res_def.node.add() #pylint: disable=maybe-no-member
			n.MergeFrom(n0)
			n.name = rename_func(n.name)
			for i, s in enumerate(n.input):
				n.input[i] = rename_func(s) if s[0]!='^' else '^'+rename_func(s[1:])
		return res_def
	  
	def showarray(a):
		a = np.uint8(np.clip(a, 0, 1)*255)
		plt.imshow(a)
		plt.show()
		
	def visstd(a, s=0.1):
		'''Normalize the image range for visualization'''
		return (a-a.mean())/max(a.std(), 1e-4)*s + 0.5

	def T(layer):
		'''Helper for getting layer output tensor'''
		return graph.get_tensor_by_name("import/%s:0"%layer)

	def render_naive(t_obj, img0, iter_n=20, step=1.0):
		t_score = tf.reduce_mean(t_obj) # defining the optimization objective
		t_grad = tf.gradients(t_score, t_input)[0] # behold the power of automatic differentiation!
		
		img = img0.copy()
		for _ in range(iter_n):
			g, _ = sess.run([t_grad, t_score], {t_input:img})
			# normalizing the gradient, so the same step size should work 
			g /= g.std()+1e-8         # for different layers and networks
			img += g*step
		showarray(visstd(img))
		
	def tffunc(*argtypes):
		'''Helper that transforms TF-graph generating function into a regular one.
		See "resize" function below.
		'''
		placeholders = list(map(tf.placeholder, argtypes))
		def wrap(f):
			out = f(*placeholders)
			def wrapper(*args, **kw):
				return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))
			return wrapper
		return wrap

	def resize(img, size):
		img = tf.expand_dims(img, 0)
		return tf.image.resize_bilinear(img, size)[0,:,:,:]
	resize = tffunc(np.float32, np.int32)(resize)

	def calc_grad_tiled(img, t_grad, tile_size=512):
		'''Compute the value of tensor t_grad over the image in a tiled way.
		Random shifts are applied to the image to blur tile boundaries over 
		multiple iterations.'''
		sz = tile_size
		h, w = img.shape[:2]
		sx, sy = np.random.randint(sz, size=2)
		img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
		grad = np.zeros_like(img)
		for y in range(0, max(h-sz//2, sz),sz):
			for x in range(0, max(w-sz//2, sz),sz):
				sub = img_shift[y:y+sz,x:x+sz]
				g = sess.run(t_grad, {t_input:sub})
				grad[y:y+sz,x:x+sz] = g
		return np.roll(np.roll(grad, -sx, 1), -sy, 0)
	
	def render_deepdream(t_obj, img0, t_input, iter_n=10, step=1.5, octave_n=10, octave_scale=1.4):
		t_score = tf.reduce_mean(t_obj)
		t_grad = tf.gradients(t_score, t_input)[0]
		
		img = img0
		octaves =[]
		for _ in range(octave_n - 1):
			hw = img.shape[:2]
			lo = resize(img, np.int32(np.float32(hw)/octave_scale))
			hi = img - resize(lo, hw)
			img = lo
			octaves.append(hi)
			
		for octave in range(octave_n):
			if octave > 0:
				hi = octaves[-octave]
				img = resize(img, hi.shape[:2]) + hi
			for _ in range(iter_n):
				g = calc_grad_tiled(img, t_grad)
				img += g*(step / (np.abs(g).mean() + 1e-7))
				
		return img	
			
	#Pick a layer from the model and a feature channel
	# http://storage.googleapis.com/deepdream/visualz/tensorflow_inception/index.html
	# http://storage.googleapis.com/deepdream/visualz/tensorflow_inception/mixed4d_3x3_bottleneck_pre_relu.html
	layer = 'mixed5a_pool_reduce_pre_relu'
	channel = 24
	
	img0 = Image.open('./img/pilatus800.jpg')
	img0 = np.float32(img0)
	
	# Apply Gradient Ascent which is basically the same as gradient descent expect we find the global max.
	# It's done by applying gradient descent to the negative of the loss function. 
	img = render_deepdream(graph.get_tensor_by_name("import/%s:0"%layer)[:,:,:,24], img0, t_input)
	
	showarray(img/255.0)
	
if __name__ == '__main__':
	run()