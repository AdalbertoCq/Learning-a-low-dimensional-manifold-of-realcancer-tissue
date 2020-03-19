import tensorflow as tf
import numpy as np


def mode_score(p_xi_y, p_ui_y, batch_size, epsilon=1e-20):
	batch_scores = list()
	batches = int(p_xi_y.shape.as_list([0])/batch_size)
	for i in range(batches):
		if batches-1 == i: 
			p_xi_y_batch = p_xi_y[i*batch_size: , :]
			p_ui_y_batch = p_ui_y[i*batch_size: , :]
		else:
			p_xi_y_batch = p_xi_y[i*batch_size: (i+1)*batch_size, :]
			p_ui_y_batch = p_ui_y[i*batch_size: (i+1)*batch_size, :]
		# Marginal label distribution over all batch samples.
		p_y_batch = tf.reduce_mean(p_xi_y_batch, axis=0)
		p_yu_batch = tf.reduce_mean(p_ui_y_batch, axis=0)
		kl_dist = p_xi_y_batch * (tf.log(p_xi_y_batch+epsilon) - tf.log(p_y_batch+epsilon))
		kl_dist_y = p_y_batch * (tf.log(p_y_batch+epsilon) - tf.log(p_yu_batch+epsilon))
		kl_dist_y = tf.reduce_sum(kl_dist_y)
		is_batch = tf.exp(tf.reduce_mean(kl_dist, axis=-1) - kl_dist_y)
		batch_scores.append(is_batch)
	return np.mean(batch_scores), np.std(batch_scores)
