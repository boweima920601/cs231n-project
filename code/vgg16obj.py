from __future__ import division
import numpy as np
import tensorflow as tf
import math
import os
import matplotlib.pyplot as plt
import sys
# %matplotlib inline
import logging
import datetime
import time
logging.basicConfig(level=logging.INFO)
FLAGS = tf.app.flags.FLAGS

class Model:
	def __init__(self):
		tf.reset_default_graph()

		self.X = tf.placeholder(tf.float32, [None, 224, 224, 3])
		self.y = tf.placeholder(tf.int64, [None])
		self.is_training = tf.placeholder(tf.bool)
		self.setup_log()
		self.setup_system()



	def setup_system(self):
		self.y_out = self.vgg16(self.X, self.y, drop_rate=FLAGS.dropout, is_training=self.is_training)

		total_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.y_out)
		self.mean_loss = tf.reduce_mean(total_loss)

		global_step = tf.Variable(0, trainable=False)
		learning_rate = tf.train.exponential_decay(FLAGS.start_learning_rate, global_step,
												   1000, 0.9, staircase=True)
		# batch normalization in tensorflow requires this extra dependency
		# extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		# with tf.control_dependencies(extra_update_ops):
		#     train_step = optimizer.minimize(self.mean_loss, global_step=global_step)
		optimizer = tf.train.AdamOptimizer(learning_rate)
		self.train_step = optimizer.minimize(self.mean_loss, global_step=global_step)

		self.saver = tf.train.Saver()

		self.correct_prediction = tf.equal(tf.argmax(self.y_out, 1), self.y)
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

	def setup_log(self):
		description = raw_input('Please add description for the model: ')
		start_time = '{:%m%d_%H%M%S}'.format(datetime.datetime.now())
		self.index_file = 'logs/index_log' + start_time + '.txt'
		self.epoch_file = 'logs/epoch_log' + start_time + '.txt'
		remark = description + '\n' + ' learn rate: {} \n batch size: {} \n data_size: {} \n dropout: {} \n'.format(FLAGS.start_learning_rate, FLAGS.batch_size, FLAGS.data_size, FLAGS.dropout)

		with open (self.index_file, 'a') as f_index:
			f_index.write(remark)
		with open (self.epoch_file, 'a') as f_epoch:
			f_epoch.write(remark)

	def vgg16(self, X, y, drop_rate=0.5, is_training=None):
		conv1 = X
		for i in range(2):
			conv1 = tf.layers.conv2d(conv1, filters=64, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
		pool1 = tf.layers.max_pooling2d(conv1, pool_size=[2, 2], strides=2)

		conv3 = pool1
		for i in range(2):
			conv3 = tf.layers.conv2d(conv3, filters=128, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
		pool2 = tf.layers.max_pooling2d(conv3, pool_size=[2, 2], strides=2)

		conv5 = pool2
		for i in range(3):
			conv5 = tf.layers.conv2d(conv5, filters=256, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
		pool3 = tf.layers.max_pooling2d(conv5, pool_size=[2, 2], strides=2)

		conv8 = pool3
		for i in range(3):
			conv8 = tf.layers.conv2d(conv8, filters=512, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
		pool4 = tf.layers.max_pooling2d(conv8, pool_size=[2, 2], strides=2)

		conv11 = pool4
		for i in range(3):
			conv11 = tf.layers.conv2d(conv11, filters=512, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
		pool5 = tf.layers.max_pooling2d(conv11, pool_size=[2, 2], strides=2)

		pool5_flat = tf.reshape(pool5, [-1, 7 * 7 * 512])
		fc1 = tf.layers.dense(pool5_flat, units=4096, activation = tf.nn.relu)
		fc2 = tf.layers.dense(fc1, units=4096, activation=tf.nn.relu)

		dropout1 = tf.layers.dropout(fc2, rate=drop_rate, training=is_training)
		logits = tf.layers.dense(dropout1, units=10)
	#     logits = tf.layers.dense(fc2, units=10)

		return logits

	def run_model(self, session, Xd, yd, epochs=1, batch_size=64, training_now=False, use_save=False, plot_losses=False):

		# shuffle indicies
		train_indicies = np.arange(Xd.shape[0])
		np.random.shuffle(train_indicies)

		variables = [self.mean_loss, self.correct_prediction, self.accuracy]
		if training_now:
			variables[-1] = self.train_step

		iter_cnt = 0
		for e in range(epochs):
			index = 0
			# keep track of losses and accuracy
			correct = 0
			losses = []
			total_loss = 0
			for i in range(int(math.ceil((Xd.shape[0] / batch_size)))):
				start_idx = (i * batch_size) % Xd.shape[0]
				idx = train_indicies[start_idx: start_idx + batch_size]
				feed_dict = {self.X: Xd[idx,:], self.y: yd[idx], self.is_training: training_now}
				actual_batch_size = yd[idx].shape[0]

				# have tensorflow compute loss and correct predictions
				# and (if given) perform a training step
				loss, corr, _ = session.run(variables, feed_dict=feed_dict)
				losses.append(loss)
				correct += np.sum(corr)
				total_loss += loss * actual_batch_size
				iter_cnt += actual_batch_size
				index += actual_batch_size

				if training_now and index % FLAGS.print_every == 0:
					index_res = '{}  loss  {}  accuracy  {}  E{}'.format(index, loss, np.sum(corr) / actual_batch_size, e + 1)
					print(index_res)
					if index % FLAGS.log_every == 0:
						time_now = '   {:%m%d_%H%M%S}\n'.format(datetime.datetime.now())
						with open (self.index_file, 'a') as f_index:
							f_index.write(index_res)
							f_index.write(time_now)

			total_correct = correct / Xd.shape[0]
			total_loss /= Xd.shape[0]
			print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}"\
				  .format(total_loss, total_correct, e+1))
			if plot_losses:
				plt.plot(losses)
				plt.grid(True)
				plt.title('Epoch {} Loss'.format(e + 1))
				plt.xlabel('minibatch number')
				plt.ylabel('minibatch loss')
				plt.show()
			if training_now and use_save:
				self.saver.save(session, os.path.join('.', 'saves/model'), global_step=e + 1)
			if not training_now:
				time_now = '   {:%m%d_%H%M%S}\n'.format(datetime.datetime.now())
				print('!!!!!!!!!!!!!!!!!')
				epoch_res = '\nEpoch # {}\nLoss: {}\nf1, em on dataset: {}\nf1, em on valid_dataset:{}\n'.format(e, val_loss, train_eval, val_eval)
				print(epoch_res)
				with open (self.epoch_file, 'a') as f_epoch:
					f_epoch.write(epoch_res)
					f_epoch.write(time_now)