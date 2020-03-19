import tensorflow as tf
import numpy as np
from models.generative.ops import *
from models.generative.activations import *
from models.generative.normalization import *

display = True

def discriminator_resnet(images, layers, spectral, activation, reuse, init='xavier', regularizer=None, normalization=None, attention=None, down='downscale', label=None, label_t='cat', infoGAN=False, c_dim=None, name='discriminator'):
	net = images
	channels = [32, 64, 128, 256, 512, 1024]
	# For resolution 448
	# channels = [16, 32, 64, 128, 256, 512, 1024]
	if display:
		print('DISCRIMINATOR INFORMATION:')
		print('Channels: ', channels[:layers])
		print('Normalization: ', normalization)
		print('Activation: ', activation)
		print('Attention:  ', attention)
		print()


	print(channels)

	with tf.variable_scope(name, reuse=reuse):

		# Discriminator with conditional projection.
		if label is not None:
			batch_size, label_dim = label.shape.as_list()
			embedding_size = channels[-1]
			# Categorical Embedding.
			print(label_t)
			if label_t == 'cat':
				# emb = embedding(shape=(label_dim, embedding_size), init=init, regularizer=regularizer, power_iterations=1)
				emb = embedding(shape=(label_dim, embedding_size), init=init, power_iterations=1)
				index = tf.argmax(label, axis=-1)
				label_emb = tf.nn.embedding_lookup(emb, index)
			# Linear conditioning, using NN to produce embedding.
			else:
				inter_dim = int((label_dim+net.shape.as_list()[-1])/2)
				net_label = dense(inputs=label, out_dim=inter_dim, spectral=spectral, init='xavier', regularizer=None, scope='label_nn_1')
				if normalization is not None: net_label = normalization(inputs=net_label, training=True)
				net_label = activation(net_label)
				label_emb = dense(inputs=net_label, out_dim=embedding_size, spectral=spectral, init='xavier', regularizer=None, scope='label_nn_2')

		for layer in range(layers):
			# ResBlock.
			net = residual_block(inputs=net, filter_size=3, stride=1, padding='SAME', scope=layer, is_training=True, normalization=normalization, use_bias=True, 
								 spectral=spectral, init=init, regularizer=regularizer, activation=activation)
			# Attention layer. 
			if attention is not None and net.shape.as_list()[1]==attention:
				net = attention_block(net, spectral=True, init=init, regularizer=regularizer, scope=layers)
			
			# Down.
			net = convolutional(inputs=net, output_channels=channels[layer], filter_size=4, stride=2, padding='SAME', conv_type=down, spectral=spectral, init=init, regularizer=regularizer, scope=layer)
			if normalization is not None: net = normalization(inputs=net, training=True)
			net = activation(net)
			
		# Flatten.
		net = tf.layers.flatten(inputs=net)

		# Dense.
		net = dense(inputs=net, out_dim=channels[-1], spectral=spectral, init=init, regularizer=regularizer, scope=1)				
		if normalization is not None: net = normalization(inputs=net, training=True)
		net = activation(net)

		# Dense
		logits_net = dense(inputs=net, out_dim=1, spectral=spectral, init=init, regularizer=regularizer, scope=2)		
		if label is not None: 
			inner_prod = tf.reduce_sum(net * label_emb, axis=-1, keepdims=True)
			logits = logits_net + inner_prod
			output = sigmoid(logits)
		else:
			logits = logits_net
			output = sigmoid(logits)

		if infoGAN:
			mean_c_x = dense(inputs=net, out_dim=c_dim, spectral=spectral, init=init, regularizer=regularizer, scope=3)
			logs2_c_x = dense(inputs=net, out_dim=c_dim, spectral=spectral, init=init, regularizer=regularizer, scope=4)
			return output, logits, mean_c_x, logs2_c_x 

	print()
	return output, logits


def discriminator_resnet_incr(images, layers, spectral, activation, reuse, init='xavier', regularizer=None, normalization=None, attention=None, down='downscale', label=None, label_t='cat', infoGAN=False, c_dim=None, name='discriminator'):
	net = images
	channels = [32, 64, 128, 256, 512, 1024]
	if display:
		print('DISCRIMINATOR INFORMATION:')
		print('Channels: ', channels[:layers])
		print('Normalization: ', normalization)
		print('Activation: ', activation)
		print('Attention:  ', attention)
		print()

	with tf.variable_scope(name, reuse=reuse):

		layer = 0
		net = convolutional(inputs=net, output_channels=channels[layer], filter_size=3, stride=1, padding='SAME', conv_type='convolutional', spectral=spectral, init=init, regularizer=regularizer, scope=layer)

		for layer in range(layers):
			# ResBlock.
			net = residual_block(inputs=net, filter_size=3, stride=1, padding='SAME', scope=layer, is_training=True, normalization=normalization, use_bias=True, spectral=spectral, init=init, regularizer=regularizer, activation=activation)
			
			# Attention layer. 
			if attention is not None and net.shape.as_list()[1]==attention:
				net = attention_block(net, spectral=True, init=init, regularizer=regularizer, scope=layers)
			
			# Down.
			layer_channel = layer+1
			if layer == layers - 1:
				layer_channel = -2
			net = convolutional(inputs=net, output_channels=channels[layer_channel], filter_size=4, stride=2, padding='SAME', conv_type=down, spectral=spectral, init=init, regularizer=regularizer, scope=layer+1)
			if normalization is not None: net = normalization(inputs=net, training=True)
			net = activation(net)
			
		# Flatten.
		net = tf.layers.flatten(inputs=net)

		shape = int(np.product(net.shape.as_list()[1:3])/4)
		# # Dense.
		# net = dense(inputs=net, out_dim=shape, spectral=spectral, init=init, regularizer=regularizer, scope=1)				
		# if normalization is not None: net = normalization(inputs=net, training=True)
		# net = activation(net)

		# Dense.
		net = dense(inputs=net, out_dim=channels[-1], spectral=spectral, init=init, regularizer=regularizer, scope=2)				
		if normalization is not None: net = normalization(inputs=net, training=True)
		net = activation(net)

		# Dense
		logits = dense(inputs=net, out_dim=1, spectral=spectral, init=init, regularizer=regularizer, scope=3)	
		output = sigmoid(logits)

	print()
	return output, logits



def encoder_resnet(images, z_dim, layers, spectral, activation, reuse, init='xavier', regularizer=None, normalization=None, attention=None, down='downscale'):
	net = images
	channels = [32, 64, 128, 256, 512, 1024]
	if display:
		print('ENCODER INFORMATION:')
		print('Channels: ', channels[:layers])
		print('Normalization: ', normalization)
		print('Activation: ', activation)
		print('Attention:  ', attention)
		print()

	with tf.variable_scope('encoder', reuse=reuse):

		for layer in range(layers+1):
			# ResBlock.
			net = residual_block(inputs=net, filter_size=3, stride=1, padding='SAME', scope=layer, is_training=True, normalization=normalization, use_bias=True, spectral=spectral, init=init, regularizer=regularizer, activation=activation)
			
			# Attention layer. 
			if attention is not None and net.shape.as_list()[1]==attention: net = attention_block(net, spectral=True, init=init, regularizer=regularizer, scope=layers)
			
			# Down.
			net = convolutional(inputs=net, output_channels=channels[layer], filter_size=4, stride=2, padding='SAME', conv_type=down, spectral=spectral, init=init, regularizer=regularizer, scope=layer)
			if normalization is not None: net = normalization(inputs=net, training=True, scope=layer)
			net = activation(net)
			
		# Flatten.
		net = tf.layers.flatten(inputs=net)

		# Dense.
		net = dense(inputs=net, out_dim=channels[-1], spectral=spectral, init=init, regularizer=regularizer, scope=1)				
		if normalization is not None: net = normalization(inputs=net, training=True)
		net = activation(net)

		# Dense
		w_latent = dense(inputs=net, out_dim=z_dim, spectral=spectral, init=init, regularizer=regularizer, scope=2)		

	print()
	return w_latent


def encoder_resnet_incr(images, z_dim, layers, spectral, activation, reuse, init='xavier', regularizer=None, normalization=None, attention=None, down='downscale', name='encoder'):
	net = images
	channels = [32, 64, 128, 256, 512, 1024]
	if display:
		print('ENCODER INFORMATION:')
		print('Channels: ', channels[:layers])
		print('Normalization: ', normalization)
		print('Activation: ', activation)
		print('Attention:  ', attention)
		print()

	with tf.variable_scope(name, reuse=reuse):

		layer = 0
		net = convolutional(inputs=net, output_channels=channels[layer], filter_size=3, stride=1, padding='SAME', conv_type='convolutional', spectral=spectral, init=init, regularizer=regularizer, scope=layer)

		for layer in range(layers):
			# ResBlock.
			net = residual_block(inputs=net, filter_size=3, stride=1, padding='SAME', scope=layer, is_training=True, normalization=normalization, use_bias=True, spectral=spectral, init=init, regularizer=regularizer, activation=activation)
			# Attention layer. 
			if attention is not None and net.shape.as_list()[1]==attention: net = attention_block(net, spectral=True, init=init, regularizer=regularizer, scope=layers)
			
			# Down.
			layer_channel = layer+1
			if layer == layers - 1:
				layer_channel = -2
			net = convolutional(inputs=net, output_channels=channels[layer_channel], filter_size=4, stride=2, padding='SAME', conv_type=down, spectral=spectral, init=init, regularizer=regularizer, scope=layer+1)
			if normalization is not None: net = normalization(inputs=net, training=True)
			net = activation(net)
			
		# Flatten.
		net = tf.layers.flatten(inputs=net)

		# shape = int(np.product(net.shape.as_list()[1:3])/4)
		# # Dense.
		# net = dense(inputs=net, out_dim=shape, spectral=spectral, init=init, regularizer=regularizer, scope=1)				
		# if normalization is not None: net = normalization(inputs=net, training=True)
		# net = activation(net)

		# Dense.
		net = dense(inputs=net, out_dim=channels[-1], spectral=spectral, init=init, regularizer=regularizer, scope=2)				
		if normalization is not None: net = normalization(inputs=net, training=True)
		net = activation(net)

		# Dense
		w_latent = dense(inputs=net, out_dim=z_dim, spectral=spectral, init=init, regularizer=regularizer, scope=3)	


	print()
	return w_latent


def discriminator(images, layers, spectral, activation, reuse, normalization=None):
	net = images
	channels = [32, 64, 128, 256, 512, 1024]
	
	if display:
		print('Discriminator Information.')
		print('Channels: ', channels[:layers])
		print('Normalization: ', normalization)
		print('Activation: ', activation)
		print()
	with tf.variable_scope('discriminator', reuse=reuse):
		# Padding = 'Same' -> H_new = H_old // Stride

		for layer in range(layers):
			# Down.
			net = convolutional(inputs=net, output_channels=channels[layer], filter_size=5, stride=2, padding='SAME', conv_type='convolutional', spectral=spectral, scope=layer+1)
			if normalization is not None: net = normalization(inputs=net, training=True)
			net = activation(net)

		# Flatten.
		net = tf.layers.flatten(inputs=net)
		
		# Dense.
		net = dense(inputs=net, out_dim=channels[-1], spectral=spectral, scope=1)				
		if normalization is not None: net = normalization(inputs=net, training=True)
		net = activation(net)
		
		# Dense
		logits = dense(inputs=net, out_dim=1, spectral=spectral, scope=2)				
		output = sigmoid(logits)

	print()
	return output, logits

