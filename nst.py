'''
Description: Neural Style Transfer Model
'''
from utility import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.misc
mpl.rcParams['figure.figsize'] = (10, 10)
mpl.rcParams['axes.grid'] = False
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
# print("Eager execution: {}".format(tf.executing_eagerly()))

# Global values
img_dir = '/images'
cimg_path = 'images/london.jpg'
simg_path = 'images/reddit.jpg'

''' For personal understanding of image loading: Splits image into 3 channels, then an extra dim to wrap the 3 channels
img = Image.open(cimg_path)
img = kp_image.img_to_array(img)
img = np.expand_dims(img, axis=0)
'''

# Visualize content and style images
'''plt.figure(figsize=(10, 10))
content = load_img(cimg_path).astype('uint8')
style = load_img(simg_path).astype('uint8')
plt.subplot(1, 2, 1)
imshow(content, 'Content Image')
plt.subplot(1, 2, 2)
imshow(style, 'Style Image')
#plt.show()'''

''' Understanding a VGG-19 processed input image: Normalizes each Blue-Green-Red channel by mean [103.939, 116.779, 123.68]
img = load_img(cimg_path)
img = tf.keras.applications.vgg19.preprocess_input(img)
print(img.shape)
print(img.shape)
'''

''' Understanding the VGG-19 architecture: 
vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
vgg.summary()
'''

# Content layer used for feature map - room for experimentation
c_layers = ['block5_conv2']
# Style layers used for reconstruction
s_layers = ['block1_conv1',
			'block2_conv1',
			'block3_conv1',
			'block4_conv1',
			'block5_conv1']
num_c_layers = len(c_layers)
num_s_layers = len(s_layers)

#base_img = process_img(cimg_path)
#base_img = tfe.Variable(base_img, dtype=tf.float32)
#print(base_img)

best, best_loss = run_nst(cimg_path, simg_path, c_layers, s_layers, num_s_layers, num_c_layers, i=3000, c_weight=1e3, s_weight=1e1)

result(best, cimg_path, simg_path)

#scipy.misc.toimage(best, cmin=0.0, cmax=255.0).save('out.jpg')