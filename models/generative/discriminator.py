import tensorflow as tf
import numpy as np
from models.generative.ops import *
from models.generative.activations import *
from models.generative.normalization import *

display = True

def discriminator_resnet(images, layers, spectral, activation, reuse, init='xavier', regularizer=None, normalization=None, attention=None, down='downscale', label=None, label_t='cat', name='discriminator'):
	net = images
	channels = [32, 64, 128, 256, 512, 1024]
	if display:
		print('DISCRIMINATOR INFORMATION:')
		print('Channels: ', channels[:layers])
		print('Normalization: ', normalization)
		print('Activation: ', activation)
		print('Attention:  ', attention)
		print()

	# Trials B. Test to improve performance on discriminator: layers += 1

	with tf.variable_scope(name, reuse=reuse):

		# Discriminator with conditional projection.
		if label is not None:
			batch_size, label_dim = label.shape.as_list()
			embedding_size = channels[-1]
			# Categorical Embedding.
			print(label_t)
			if label_t == 'cat':
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


	print()
	return output, logits


def discriminator_resnet_mask_class(images, layers, spectral, activation, reuse, init='xavier', regularizer=None, normalization=None, attention=None, down='downscale', label=None, name='discriminator'):
	net = images
	# channels = [32, 64, 128, 256, 512, 1024, 2048]
	channels = [32, 64, 128, 256, 512, 1024]

	if display:
		print('DISCRIMINATOR INFORMATION:', name)
		print('Total  Channels: ', channels)
		print('Chosen Channels: ', channels[:layers])
		print('Normalization: ', normalization)
		print('Activation: ', activation)
		print('Attention:  ', attention)
		print()

	with tf.variable_scope(name, reuse=reuse):

		# Discriminator with conditional projection.
		batch_size, label_dim = label.shape.as_list()
		embedding_size = channels[-1]

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

		feature_space = dense(inputs=net, out_dim=channels[-1], spectral=spectral, init=init, regularizer=regularizer, scope=1)				
		net = activation(feature_space)

		# Dense Classes
		class_logits = dense(inputs=net, out_dim=label_dim, spectral=spectral, init=init, regularizer=regularizer, scope=3)		
		class_logits = tf.nn.log_softmax(class_logits)
		# One encoding for label input
		logits = class_logits*label
		logits = tf.reduce_sum(logits, axis=-1)
		output = sigmoid(logits)

		
	print()
	return output, logits, feature_space


def discriminator_resnet_class(images, layers, spectral, activation, reuse, l_dim, init='xavier', regularizer=None, normalization=None, attention=None, down='downscale', name='discriminator'):
	net = images
	# channels = [32, 64, 128, 256, 512, 1024, 2048]
	channels = [32, 64, 128, 256, 512, 1024]

	if display:
		print('DISCRIMINATOR INFORMATION:', name)
		print('Total  Channels: ', channels)
		print('Chosen Channels: ', channels[:layers])
		print('Normalization: ', normalization)
		print('Activation: ', activation)
		print('Attention:  ', attention)
		print()

	with tf.variable_scope(name, reuse=reuse):

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
		feature_space = dense(inputs=net, out_dim=channels[-1], spectral=spectral, init=init, regularizer=regularizer, scope=2)		
		net = activation(feature_space)		

		# Dense
		logits = dense(inputs=net, out_dim=1, spectral=spectral, init=init, regularizer=regularizer, scope=3)		
		output = sigmoid(logits)

		net = dense(inputs=net, out_dim=channels[-1], spectral=spectral, init=init, regularizer=regularizer, scope=4)		
		net = activation(net)	

		# Dense Classes
		class_logits = dense(inputs=net, out_dim=l_dim, spectral=spectral, init=init, regularizer=regularizer, scope=5)		

	print()
	return output, logits, feature_space, class_logits


def discriminator_resnet_class2(images, layers, spectral, activation, reuse, l_dim, init='xavier', regularizer=None, normalization=None, attention=None, down='downscale', name='discriminator'):
	net = images
	# channels = [32, 64, 128, 256, 512, 1024, 2048]
	channels = [32, 64, 128, 256, 512, 1024]

	# New
	layers = layers + 1

	if display:
		print('DISCRIMINATOR INFORMATION:', name)
		print('Total  Channels: ', channels)
		print('Chosen Channels: ', channels[:layers])
		print('Normalization: ', normalization)
		print('Activation: ', activation)
		print('Attention:  ', attention)
		print()

	with tf.variable_scope(name, reuse=reuse):

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
			
		# New
		# Flatten.
		net = tf.layers.flatten(inputs=net)

		# Dense.
		feature_space = dense(inputs=net, out_dim=channels[-1], spectral=spectral, init=init, regularizer=regularizer, scope=2)		
		net = activation(feature_space)		

		# Dense
		logits = dense(inputs=net, out_dim=1, spectral=spectral, init=init, regularizer=regularizer, scope=3)		
		output = sigmoid(logits)

		net = dense(inputs=net, out_dim=channels[-1], spectral=spectral, init=init, regularizer=regularizer, scope=4)		
		net = activation(net)	

		# Dense Classes
		class_logits = dense(inputs=net, out_dim=l_dim, spectral=spectral, init=init, regularizer=regularizer, scope=5)			

	print()
	return output, logits, feature_space, class_logits


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

		# Dense.
		net = dense(inputs=net, out_dim=channels[-1], spectral=spectral, init=init, regularizer=regularizer, scope=2)				
		if normalization is not None: net = normalization(inputs=net, training=True)
		net = activation(net)

		# Dense
		logits = dense(inputs=net, out_dim=1, spectral=spectral, init=init, regularizer=regularizer, scope=3)	
		output = sigmoid(logits)

	print()
	return output, logits


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


def discriminator_encoder(enconding, layers, spectral, activation, reuse, init='xavier', regularizer=None, normalization=None, name='dis_encoding'):
	net = enconding
	channels = [150, 100, 50, 25, 12]
	# channels = [200, 150, 100, 50, 24]
	if display:
		print('DISCRIMINATOR-ENCODER INFORMATION:')
		print('Channels: ', channels[:layers])
		print('Normalization: ', normalization)
		print('Activation: ', activation)
		print()

	with tf.variable_scope(name, reuse=reuse):
		for layer in range(layers):

			# Residual Dense layer.
			net = residual_block_dense(inputs=net, scope=layer, is_training=True, normalization=normalization, spectral=spectral, activation=activation, init=init, regularizer=regularizer, display=True)

			# Dense layer downsample dim.
			net = dense(inputs=net, out_dim=channels[layer], spectral=spectral, init=init, regularizer=regularizer, scope=layer)				
			if normalization is not None: net = normalization(inputs=net, training=True)
			net = activation(net)

		# Dense
		logits_net = dense(inputs=net, out_dim=1, spectral=spectral, init=init, regularizer=regularizer, scope=layer+1)		
		output = sigmoid(logits_net)

	print()
	return output, logits_net

