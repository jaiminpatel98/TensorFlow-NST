'''
Description: Utility methods for Neural Transfer
'''
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (10, 10)
mpl.rcParams['axes.grid'] = False
import numpy as np
import IPython.display
from PIL import Image

import time
import functools

import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.python.keras.preprocessing import image as kp_image
from tensorflow.python.keras import models
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import backend as k
tf.compat.v1.enable_eager_execution()

def load_img(img_path):
	''' 
	Resizes image and converts to array
	'''
	max_dim = 1024
	img = Image.open(img_path)
	long = max(img.size)
	scale = max_dim/long
	img = img.resize((round(img.size[0]*scale), round(img.size[1]*scale)), Image.ANTIALIAS)
	img = kp_image.img_to_array(img)
	
	#Broadcast image array to have batch dim
	img = np.expand_dims(img, axis=0)
	return img

def imshow(img, title=None):
	'''
	Removes normalize image processing in order to display with plt
	'''
	# Remove batch dim
	out = np.squeeze(img, axis=0)
	# Normalize
	out = out.astype('uint8')
	plt.imshow(out)
	if title is not None:
		plt.title(title)
	plt.imshow(out)

def process_img(img_path):
	'''
	Process img for VGG-19 input
	'''
	img = load_img(img_path)
	img = tf.keras.applications.vgg19.preprocess_input(img)
	return img

def deprocess_img(img):
	'''
	Converting a VGG-19 img back to standard img-array
	'''
	tmp = img.copy()
	# Make sure img is of 3-dimension
	if len(tmp.shape) == 4:
		tmp = np.squeeze(tmp, 0)
		assert len(tmp.shape) == 3, ('Deprocess input image must be of dimension [1, H, W, C] or [H, W, C]')
	if len(tmp.shape) != 3:
		raise ValueError('Invalid image input')
	# Invert VGG-19 processing step
	tmp[:, :, 0] += 103.939
	tmp[:, :, 1] += 116.779
	tmp[:, :, 2] += 123.78
	tmp = tmp[:, :, ::-1]
	# Set values in range 0, 255
	tmp = np.clip(tmp, 0, 255).astype('uint8')
	return tmp

def vgg_model(c_layers, s_layers):
	'''
	Loads VGG-19 model trained on imagenet dataset.
	Retrieves desired intermediate layers for style and content.
	Returns new model taking an input image and returning intermediate layer outputs.
	'''
	# Load pretrained imagenet VGG-19 network
	vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
	vgg.trainable = False
	# Get output layers corresponding to function params
	s_out = [vgg.get_layer(name).output for name in s_layers]
	c_out = [vgg.get_layer(name).output for name in c_layers]
	model_out = s_out + c_out
	return models.Model(vgg.input, model_out)

def gram_matrix(input_t):
	c = int(input_t.shape[-1])
	a = tf.reshape(input_t, [-1, c])
	n = tf.shape(a)[0]
	gram = tf.matmul(a, a, transpose_a=True)
	return gram / tf.cast(n, tf.float32)

def content_loss_function(base, target):
	return tf.reduce_mean(tf.square(base - target))

def style_loss_function(base, target):
	h, w, c = base.get_shape().as_list()
	gram_style = gram_matrix(base)
	return tf.reduce_mean(tf.square(gram_style - target))

def feature_map(model, c_path, s_path, c_layers, s_layers, num_c_layers, num_s_layers):
	# Process images
	c_img = process_img(c_path)
	s_img = process_img(s_path)
	# Compute content and style features
	c_out = model(c_img)
	s_out = model(s_img)
	# Get feature maps from model
	c_feat = [c_layer[0] for c_layer in c_out[num_s_layers:]]
	s_feat = [s_layer[0] for s_layer in s_out[:num_s_layers]]
	return c_feat, s_feat

def total_loss(model, loss_weights, base_img, g_s_feats, c_feats, num_s_layers, num_c_layers):
	s_weight, c_weight = loss_weights
	
	model_out = model(base_img)

	s_out_feat = model_out[:num_s_layers]
	c_out_feat = model_out[num_s_layers:]
	
	s_score = 0
	c_score = 0

	weight_per_s_layer = 1.0 / float(num_s_layers)
	for target_s, comb_s in zip(g_s_feats, s_out_feat):
		s_score += weight_per_s_layer * style_loss_function(comb_s[0], target_s)

	weight_per_c_layer = 1.0 / float(num_c_layers)
	for target_c, comb_c in zip(c_feats, c_out_feat):
		c_score += weight_per_c_layer * content_loss_function(comb_c[0], target_c)

	s_score *= s_weight
	c_score *= c_weight

	loss = s_score + c_score
	return loss, s_score, c_score

def compute_gradient(c):
	with tf.GradientTape() as tape: 
		loss = total_loss(**c)
	t_loss = loss[0]
	return tape.gradient(t_loss, c['base_img']), loss

def run_nst(c_path, s_path, c_layers, s_layers, num_s_layers, num_c_layers, i=1000, c_weight=1e3, s_weight=1e-2):
	model = vgg_model(c_layers, s_layers)
	for layer in model.layers:
		layer.trainable = False

	c_feats, s_feats = feature_map(model, c_path, s_path, c_layers, s_layers, num_c_layers, num_s_layers)
	g_s_feats = [gram_matrix(s_feat) for s_feat in s_feats]

	base_img = process_img(c_path)
	base_img = tfe.Variable(base_img, dtype=tf.float32)

	optimizer = tf.train.AdamOptimizer(learning_rate=7, beta1=0.99, epsilon=1e-1)

	i_count = 1

	best_loss, best_img = float('inf'), None

	loss_weights = (s_weight, c_weight)

	c = {
	'model': model,
	'loss_weights': loss_weights,
	'base_img': base_img,
	'g_s_feats': g_s_feats,
	'c_feats': c_feats,
	'num_s_layers': num_s_layers,
	'num_c_layers': num_c_layers
	}

	rows = 2
	cols = 5
	interval = i / (rows*cols)
	global_start = time.time()

	norm_means = np.array([103.939, 116.779, 123.68])
	min_vals = -norm_means
	max_vals = 255 - norm_means

	imgs = []
	for j in range(i):
		start_time = time.time()
		grads, loss = compute_gradient(c)
		loss, s_score, c_score = loss
		optimizer.apply_gradients([(grads, base_img)])
		clipped = tf.clip_by_value(base_img, min_vals, max_vals)
		base_img.assign(clipped)

		if loss < best_loss:
			best_loss = loss
			best_img = deprocess_img(base_img.numpy())

		if j % interval == 0:
			plot_img = base_img.numpy()
			plot_img = deprocess_img(plot_img)
			imgs.append(plot_img)
			IPython.display.clear_output(wait=True)
			IPython.display.display_png(Image.fromarray(plot_img))
			print('Iteration: {}'.format(j))
			print('Total Loss: {:.4e}, '
				'Style Loss: {:.4e},'
				'Content Loss: {:.4e}'
				'Time: {:.4f}s'.format(loss, s_score, c_score, time.time() - start_time))
	print('Total Time: {:.4f}s'.format(time.time() - global_start))
	IPython.display.clear_output(wait=True)
	plt.figure(figsize=(14, 20))
	#Image.fromarray(best_img)
	for k, img in enumerate(imgs):
		plt.subplot(rows, cols, k+1)
		plt.imshow(img)
		plt.xticks([])
		plt.yticks([])
	return best_img, best_loss

def result(best_img, c_path, s_path, show_large_final=True):
	plt.figure(figsize=(10, 5))
	content = load_img(c_path)
	style = load_img(s_path)

	plt.subplot(1, 2, 1)
	imshow(content, 'Content Image')

	plt.subplot(1, 2, 2)
	imshow(style, 'Style Image')

	if show_large_final:
		plt.figure(figsize=(10,10))
		plt.imshow(best_img)
		plt.title('Output Image')
		plt.show()
