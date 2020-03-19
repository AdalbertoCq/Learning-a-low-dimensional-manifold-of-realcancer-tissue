import os
import tensorflow as tf
import numpy as np
import h5py
import random
import shutil
import tensorflow.contrib.gan as tfgan
from models.generative.utils import *
from data_manipulation.utils import *

# Gather real samples from train and test sets for FID and other scores.
def real_samples(data, data_output_path, num_samples=5000):
	path = os.path.join(data_output_path, 'evaluation')
	path = os.path.join(path, 'real')
	path = os.path.join(path, data.dataset)
	path = os.path.join(path, data.marker)
	res = 'h%s_w%s_n%s' % (data.training.patch_h, data.training.patch_w, data.training.n_channels)
	path = os.path.join(path, res)
	img_train = os.path.join(path, 'img_train')
	img_test = os.path.join(path, 'img_test')
	if not os.path.isdir(path):
		os.makedirs(path)
		os.makedirs(img_train)
		os.makedirs(img_test)

	batch_size = data.training.batch_size
	images_shape =  [num_samples] + data.test.shape[1:]

	hdf5_path_train = os.path.join(path, 'hdf5_%s_%s_images_train_real.h5' % (data.dataset, data.marker))
	hdf5_path_test = os.path.join(path, 'hdf5_%s_%s_images_test_real.h5' % (data.dataset, data.marker))
	
	if os.path.isfile(hdf5_path_train):
		print('H5 File Image Train already created.')
		print('\tFile:', hdf5_path_train)
	else:
		hdf5_img_train_real_file = h5py.File(hdf5_path_train, mode='w')
		train_storage = hdf5_img_train_real_file.create_dataset(name='images', shape=images_shape, dtype=np.float32)
		
		print('H5 File Image Train.')
		print('\tFile:', hdf5_path_train)

		possible_samples = len(data.training.images)
		random_samples = list(range(possible_samples))
		random.shuffle(random_samples)

		ind = 0
		for index in random_samples[:num_samples]:
			train_storage[ind] = data.training.images[index]
			plt.imsave('%s/real_train_%s.png' % (img_train, ind), data.training.images[index])
			ind += 1
		print('\tNumber of samples:', ind)

	if os.path.isfile(hdf5_path_test):
		print('H5 File Image Test already created.')
		print('\tFile:', hdf5_path_test)
	else:
		hdf5_img_test_real_file = h5py.File(hdf5_path_test, mode='w')
		test_storage = hdf5_img_test_real_file.create_dataset(name='images', shape=images_shape, dtype=np.float32)

		print('H5 File Image Test')
		print('\tFile:', hdf5_path_test)

		possible_samples = len(data.test.images)
		random_samples = list(range(possible_samples))
		random.shuffle(random_samples)

		ind = 0
		for index in random_samples[:num_samples]:
			test_storage[ind] = data.test.images[index]
			plt.imsave('%s/real_test_%s.png' % (img_test, ind), data.test.images[index])
			ind += 1
		print('\tNumber of samples:', ind)

	return hdf5_path_train, hdf5_path_test

# Extract Inception-V1 features from images in HDF5.
def inception_tf_feature_activations(hdf5s, input_shape, batch_size):
	images_input = tf.placeholder(dtype=tf.float32, shape=[None] + input_shape, name='images')
	images = 2*images_input
	images -= 1
	images = tf.image.resize_bilinear(images, [299, 299])
	out_incept_v3 = tfgan.eval.run_inception(images=images, output_tensor='pool_3:0')

	hdf5s_features = list()
	with tf.Session() as sess:
		for hdf5_path in hdf5s:
			hdf5_feature_path = hdf5_path.replace('_images_','_features_')
			if os.path.isfile(hdf5_feature_path):
				print('H5 File Feature already created.')
				print('\tFile:', hdf5_feature_path)
				hdf5s_features.append(hdf5_feature_path)
				continue
			hdf5_img_file = h5py.File(hdf5_path, mode='r')
			images_storage = hdf5_img_file['images']
			if 'images_prime' in hdf5_img_file:
				images_prime_storage = hdf5_img_file['images_prime']
			num_samples = images_storage.shape[0]
			batches = int(num_samples/batch_size)

			features_shape = (num_samples, 2048)
			if os.path.isfile(hdf5_feature_path):
				os.remove(hdf5_feature_path)
			hdf5_features_file = h5py.File(hdf5_feature_path, mode='w')
			features_storage = hdf5_features_file.create_dataset(name='features', shape=features_shape, dtype=np.float32)
			if 'images_prime' in hdf5_img_file: 
				features_prime_storage = hdf5_features_file.create_dataset(name='features_prime', shape=features_shape, dtype=np.float32)
			hdf5s_features.append(hdf5_feature_path)

			print('Starting features extraction...')
			print('\tImage File:', hdf5_path)
			print('\t\tImage type: images')
			ind = 0
			for batch_num in range(batches):
				batch_images = images_storage[batch_num*batch_size:(batch_num+1)*batch_size]
				if 'test' in hdf5_path or 'train' in hdf5_path:
					batch_images = batch_images/255.
				activations = sess.run(out_incept_v3, {images_input: batch_images})
				features_storage[batch_num*batch_size:(batch_num+1)*batch_size] = activations
				ind += batch_size
			print('\tFeature File:', hdf5_feature_path)
			print('\tNumber of samples:', ind)

			if 'images_prime' in hdf5_img_file: 
				print('Starting features extraction...')
				print('\tImage File:', hdf5_path)
				print('\t\tImage type: images recon')
				ind = 0
				for batch_num in range(batches):
					batch_images = images_prime_storage[batch_num*batch_size:(batch_num+1)*batch_size]
					if 'test' in hdf5_path or 'train' in hdf5_path:
						batch_images = batch_images/255.
					activations = sess.run(out_incept_v3, {images_input: batch_images})
					features_prime_storage[batch_num*batch_size:(batch_num+1)*batch_size] = activations
					ind += batch_size
				print('\tFeature File:', hdf5_feature_path)
				print('\tNumber of samples:', ind)

	return hdf5s_features

# Generate random samples from a model, it also dumps a sprite image width them.
def generate_samples_epoch(session, model, data_shape, epoch, evaluation_path, num_samples=5000, batches=50):
	epoch_path = os.path.join(evaluation_path, 'epoch_%s' % epoch)
	check_epoch_path = os.path.join(epoch_path, 'checkpoints')
	checkpoint_path = os.path.join(evaluation_path, '../checkpoints')
	
	os.makedirs(epoch_path)
	shutil.copytree(checkpoint_path, check_epoch_path)

	if model.conditional:
		runs = ['postive', 'negative']
	else:
		runs = ['unconditional']

	for run in  runs:

		hdf5_path = os.path.join(epoch_path, 'hdf5_epoch_%s_gen_images_%s.h5' % (epoch, run))
		
		# H5 File.
		img_shape = [num_samples] + data_shape
		hdf5_file = h5py.File(hdf5_path, mode='w')
		storage = hdf5_file.create_dataset(name='images', shape=img_shape, dtype=np.float32)

		ind = 0
		while ind < num_samples:
			if model.conditional:
				label_input = model.label_input
				if 'postive' in run:
					labels = np.ones((batches, 1))
				else:
					labels = np.zeros((batches, 1))
				labels = tf.keras.utils.to_categorical(y=labels, num_classes=2)
			else:
				label_input=None
				labels=None
			gen_samples, _ = show_generated(session=session, z_input=model.z_input, z_dim=model.z_dim, output_fake=model.output_gen, label_input=label_input, labels=labels, n_images=batches, show=False)

			for i in range(batches):
				if ind == num_samples:
					break
				storage[ind] = gen_samples[i, :, :, :]
				ind += 1

# Generate sampeles from PathologyGAN, no encoder.
def generate_samples_from_checkpoint(model, data, data_out_path, checkpoint, num_samples=5000, batches=50):
	path = os.path.join(data_out_path, 'evaluation')
	path = os.path.join(path, model.model_name)
	path = os.path.join(path, data.dataset)
	path = os.path.join(path, data.marker)
	res = 'h%s_w%s_n%s' % tuple(data.test.shape[1:])
	path = os.path.join(path, res)
	img_path = os.path.join(path, 'generated_images')
	if not os.path.isdir(path):
		os.makedirs(path)
		os.makedirs(img_path)

	hdf5_path = os.path.join(path, 'hdf5_%s_%s_images_%s.h5' % (data.dataset, data.marker, model.model_name))
	
	# Lazy access to one set of images, not used at all, just filling tensorflows complains.
	for batch_images, batch_labels in data.training:
		break
	
	if not os.path.isfile(hdf5_path):
		# H5 File specifications and creation.
		img_shape = [num_samples] + data.test.shape[1:]
		latent_shape = [num_samples] + [model.z_dim]
		hdf5_file = h5py.File(hdf5_path, mode='w')
		img_storage = hdf5_file.create_dataset(name='images', shape=img_shape, dtype=np.float32)
		z_storage = hdf5_file.create_dataset(name='z_latent', shape=latent_shape, dtype=np.float32)
		if 'PathologyGAN' in model.model_name:
			w_storage = hdf5_file.create_dataset(name='w_latent', shape=latent_shape, dtype=np.float32)
		print('Generated Images path:', img_path)
		print('H5 File path:', hdf5_path)

		saver = tf.train.Saver()
		with tf.Session() as session:

			# Initializer and restoring model.
			session.run(tf.global_variables_initializer())
			saver.restore(session, checkpoint)

			ind = 0
			while ind < num_samples:
				
				# Image and latent generation for PathologyGAN.
				if model.model_name == 'BigGAN':
					z_latent_batch = np.random.normal(size=(batches, model.z_dim))
					feed_dict = {model.z_input:z_latent_batch, model.real_images:batch_images}
					gen_img_batch = session.run([model.output_gen], feed_dict=feed_dict)[0]

				# Image and latent generation for StylePathologyGAN.
				else:
					z_latent_batch = np.random.normal(size=(batches, model.z_dim))
					feed_dict = {model.z_input_1: z_latent_batch, model.real_images:batch_images}
					w_latent_batch = session.run([model.w_latent_out], feed_dict=feed_dict)[0]
					w_latent_in = np.tile(w_latent_batch[:,:, np.newaxis], [1, 1, model.layers+1])
					feed_dict = {model.w_latent_in:w_latent_in, model.real_images:batch_images}
					gen_img_batch = session.run([model.output_gen], feed_dict=feed_dict)[0]

				# Fill in storage for latent and image.
				for i in range(batches):
					if ind == num_samples:
						break
					img_storage[ind] = gen_img_batch[i, :, :, :]
					z_storage[ind] = z_latent_batch[i, :]
					if 'PathologyGAN' in model.model_name:
						w_storage[ind] = w_latent_batch[i, :]
					plt.imsave('%s/gen_%s.png' % (img_path, ind), gen_img_batch[i, :, :, :])
					ind += 1
		print(ind, 'Generated Images')
	else:
		print('H5 File already created.')
		print('H5 File Generated Samples')
		print('\tFile:', hdf5_path)

	return hdf5_path

# Generate and encode samples from PathologyGAN, with an encoder.
def generate_encode_samples_from_checkpoint(model, data, data_out_path, checkpoint, num_samples=5000, batches=50):
	path = os.path.join(data_out_path, 'evaluation')
	path = os.path.join(path, model.model_name)
	path = os.path.join(path, data.dataset)
	path = os.path.join(path, data.marker)
	res = 'h%s_w%s_n%s' % tuple(data.test.shape[1:])
	path = os.path.join(path, res)
	img_path = os.path.join(path, 'generated_images')
	if not os.path.isdir(path):
		os.makedirs(path)
	if not os.path.isdir(img_path):
		os.makedirs(img_path)

	hdf5_path = os.path.join(path, 'hdf5_%s_%s_images_%s.h5' % (data.dataset, data.marker, model.model_name))
	
	# Lazy access to one set of images, not used at all, just filling tensorflows complains.
	for batch_images, batch_labels in data.training:
		break
	
	if not os.path.isfile(hdf5_path):
		# H5 File specifications and creation.
		img_shape = [num_samples] + data.test.shape[1:]
		latent_shape = [num_samples] + [model.z_dim]
		hdf5_file = h5py.File(hdf5_path, mode='w')
		z_storage = hdf5_file.create_dataset(name='z_latent', shape=latent_shape, dtype=np.float32)
		# Generated images.
		img_storage = hdf5_file.create_dataset(name='images', shape=img_shape, dtype=np.float32)
		w_storage = hdf5_file.create_dataset(name='w_latent', shape=latent_shape, dtype=np.float32)
		# Reconstructed generated images.
		img_prime_storage = hdf5_file.create_dataset(name='images_prime', shape=img_shape, dtype=np.float32)
		w_prime_storage = hdf5_file.create_dataset(name='w_latent_prime', shape=latent_shape, dtype=np.float32)
		print('Generated Images path:', img_path)
		print('H5 File path:', hdf5_path)

		saver = tf.train.Saver()
		with tf.Session() as session:

			# Initializer and restoring model.
			session.run(tf.global_variables_initializer())
			saver.restore(session, checkpoint)

			ind = 0
			while ind < num_samples:
				
				# W latent.
				z_latent_batch = np.random.normal(size=(batches, model.z_dim))
				feed_dict = {model.z_input_1: z_latent_batch}
				w_latent_batch = session.run([model.w_latent_out], feed_dict=feed_dict)[0]
				w_latent_in = np.tile(w_latent_batch[:,:, np.newaxis], [1, 1, model.layers+1])

				# Generate images from W latent space.
				feed_dict = {model.w_latent_in:w_latent_in}
				gen_img_batch = session.run([model.output_gen], feed_dict=feed_dict)[0]

				# Encode generated images into W' latent space.
				feed_dict = {model.real_images_2:gen_img_batch}
				w_latent_prime_batch = session.run([model.w_latent_e_out], feed_dict=feed_dict)[0]
				w_latent_prime_in = np.tile(w_latent_prime_batch[:,:, np.newaxis], [1, 1, model.layers+1])

				# Generate images from W' latent space.
				feed_dict = {model.w_latent_in:w_latent_prime_in}
				gen_img_prime_batch = session.run([model.output_gen], feed_dict=feed_dict)[0]

				# Fill in storage for latent and image.
				for i in range(batches):
					if ind == num_samples:
						break
					z_storage[ind] = z_latent_batch[i, :]
					# Generated.
					img_storage[ind] = gen_img_batch[i, :, :, :]
					w_storage[ind] = w_latent_batch[i, :]
					# Reconstructed.
					img_prime_storage[ind] = gen_img_prime_batch[i, :, :, :]
					w_prime_storage[ind] = w_latent_prime_batch[i, :]

					# Saving images
					plt.imsave('%s/gen_%s.png' % (img_path, ind), gen_img_batch[i, :, :, :])
					plt.imsave('%s/gen_recon_%s.png' % (img_path, ind), gen_img_prime_batch[i, :, :, :])
					ind += 1
		print(ind, 'Generated Images')
	else:
		print('H5 File already created.')
		print('H5 File Generated Samples')
		print('\tFile:', hdf5_path)

	return hdf5_path

# Encode real images and regenerate from PathologyGAN, with an encoder.
def real_encode_samples_from_checkpoint(model, data, data_out_path, checkpoint, real_hdf5, num_samples=5000, batches=50):
	path = os.path.join(data_out_path, 'evaluation')
	path = os.path.join(path, model.model_name)
	path = os.path.join(path, data.dataset)
	path = os.path.join(path, data.marker)
	res = 'h%s_w%s_n%s' % tuple(data.test.shape[1:])
	path = os.path.join(path, res)
	img_path = os.path.join(path, 'real_images')
	if not os.path.isdir(path):
		os.makedirs(path)
	if not os.path.isdir(img_path):
		os.makedirs(img_path)

	if not os.path.isfile(real_hdf5):
		print('Real image H5 file does not exist:', real_hdf5)
		exit()
	real_images = read_hdf5(real_hdf5, 'images')

	hdf5_path = os.path.join(path, 'hdf5_%s_%s_real_images_%s.h5' % (data.dataset, data.marker, model.model_name))
	
	# Lazy access to one set of images, not used at all, just filling tensorflows complains.
	for batch_images, batch_labels in data.training:
		break
	
	if not os.path.isfile(hdf5_path):
		# H5 File specifications and creation.
		img_shape = [num_samples] + data.test.shape[1:]
		latent_shape = [num_samples] + [model.z_dim]
		hdf5_file = h5py.File(hdf5_path, mode='w')
		# Real images.
		img_storage = hdf5_file.create_dataset(name='images', shape=img_shape, dtype=np.float32)
		w_storage = hdf5_file.create_dataset(name='w_latent', shape=latent_shape, dtype=np.float32)
		# Reconstructed generated images.
		img_prime_storage = hdf5_file.create_dataset(name='images_prime', shape=img_shape, dtype=np.float32)
		w_prime_storage = hdf5_file.create_dataset(name='w_latent_prime', shape=latent_shape, dtype=np.float32)
		print('Generated Images path:', img_path)
		print('H5 File path:', hdf5_path)

		saver = tf.train.Saver()
		with tf.Session() as session:

			# Initializer and restoring model.
			session.run(tf.global_variables_initializer())
			saver.restore(session, checkpoint)

			ind = 0
			while ind < num_samples:

				# Real images.
				if (ind + batches) < len(real_images):
					real_img_batch = real_images[ind: ind+batches, :, :, :]/255.
				else:
					real_img_batch = real_images[ind:, :, :, :]/255.

				# Encode real images into W latent space.
				feed_dict = {model.real_images_2:real_img_batch}
				w_latent_batch = session.run([model.w_latent_e_out], feed_dict=feed_dict)[0]
				w_latent_in = np.tile(w_latent_batch[:,:, np.newaxis], [1, 1, model.layers+1])

				# Generate images from W latent space.
				feed_dict = {model.w_latent_in:w_latent_in}
				recon_img_batch = session.run([model.output_gen], feed_dict=feed_dict)[0]

				# Encode reconstructed images into W' latent space.
				feed_dict = {model.real_images_2:recon_img_batch}
				w_latent_prime_batch = session.run([model.w_latent_e_out], feed_dict=feed_dict)[0]
				w_latent_prime_in = np.tile(w_latent_prime_batch[:,:, np.newaxis], [1, 1, model.layers+1])

				# Fill in storage for latent and image.
				for i in range(batches):
					if ind == num_samples:
						break
					# Real Images.
					img_storage[ind] = real_img_batch[i, :, :, :]
					w_storage[ind] = w_latent_batch[i, :]
					
					# Reconstructed images.
					img_prime_storage[ind] = recon_img_batch[i, :, :, :]
					w_prime_storage[ind] = w_latent_prime_batch[i, :]

					# Saving images
					plt.imsave('%s/real_%s.png' % (img_path, ind), real_img_batch[i, :, :, :])
					plt.imsave('%s/real_recon_%s.png' % (img_path, ind), recon_img_batch[i, :, :, :])
					ind += 1
		print(ind, 'Generated Images')
	else:
		print('H5 File already created.')
		print('H5 File Generated Samples')
		print('\tFile:', hdf5_path)

	return hdf5_path

# Encode real images for prognosis.
def real_encode_prognosis_from_checkpoint(model, data, data_out_path, checkpoint, real_hdf5, type_img='train_img', batches=50):
	path = os.path.join(data_out_path, 'evaluation')
	path = os.path.join(path, model.model_name)
	path = os.path.join(path, data.dataset)
	path = os.path.join(path, data.marker)
	res = 'h%s_w%s_n%s' % tuple(data.test.shape[1:])
	path = os.path.join(path, res)
	img_path = os.path.join(path, 'real_images_prognosis')
	if not os.path.isdir(path):
		os.makedirs(path)
	if not os.path.isdir(img_path):
		os.makedirs(img_path)

	if not os.path.isfile(real_hdf5):
		print('Real image H5 file does not exist:', real_hdf5)
		exit()
	real_images = read_hdf5(real_hdf5, type_img)
	num_samples = len(real_images)
	
	dataset = real_hdf5.split('/hdf5_')[1]
	dataset = dataset.split('_')[0]
	hdf5_path = os.path.join(path, 'hdf5_%s_%s_real_images_prognosis_%s.h5' % (dataset, data.marker, model.model_name))

	# Lazy access to one set of images, not used at all, just filling tensorflows complains.
	for batch_images, batch_labels in data.training:
		break
		
	if not os.path.isfile(hdf5_path):
		# H5 File specifications and creation.
		img_shape = [num_samples] + data.test.shape[1:]
		latent_shape = [num_samples] + [model.z_dim]
		hdf5_file = h5py.File(hdf5_path, mode='w')
		# Real images.
		img_storage = hdf5_file.create_dataset(name='images', shape=img_shape, dtype=np.float32)
		w_storage = hdf5_file.create_dataset(name='w_latent', shape=latent_shape, dtype=np.float32)
		# Reconstructed generated images.
		img_prime_storage = hdf5_file.create_dataset(name='images_prime', shape=img_shape, dtype=np.float32)

		print('Generated Images path:', img_path)
		print('H5 File path:', hdf5_path)

		saver = tf.train.Saver()
		with tf.Session() as session:

			# Initializer and restoring model.
			session.run(tf.global_variables_initializer())
			saver.restore(session, checkpoint)

			print('Number of Real Images:', num_samples)
			print('Starting encoding...')
			ind = 0
			while ind < num_samples:

				# Real images.
				if (ind + batches) < len(real_images):
					real_img_batch = real_images[ind: ind+batches, :, :, :]/255.
				else:
					real_img_batch = real_images[ind:, :, :, :]/255.

				# Encode real images into W latent space.
				feed_dict = {model.real_images_2:real_img_batch}
				w_latent_batch = session.run([model.w_latent_e_out], feed_dict=feed_dict)[0]
				w_latent_in = np.tile(w_latent_batch[:,:, np.newaxis], [1, 1, model.layers+1])

				# Generate images from W latent space.
				feed_dict = {model.w_latent_in:w_latent_in}
				recon_img_batch = session.run([model.output_gen], feed_dict=feed_dict)[0]

				# Fill in storage for latent and image.
				for i in range(batches):
					if ind == num_samples:
						break

					# Real Images.
					img_storage[ind] = real_img_batch[i, :, :, :]
					w_storage[ind] = w_latent_batch[i, :]
					
					# Reconstructed images.
					img_prime_storage[ind] = recon_img_batch[i, :, :, :]

					# Saving images
					plt.imsave('%s/real_%s.png' % (img_path, ind), real_img_batch[i, :, :, :])
					plt.imsave('%s/real_recon_%s.png' % (img_path, ind), recon_img_batch[i, :, :, :])
					ind += 1
		print(ind, 'Encoded Images')
	else:
		print('H5 File already created.')
		print('H5 File Generated Samples')
		print('\tFile:', hdf5_path)

	return hdf5_path, num_samples