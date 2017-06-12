from __future__ import division

import matplotlib as matplotlib

matplotlib.use('Agg')

import numpy as np
import tensorflow as tf
import math
import os
import matplotlib.pyplot as plt
import sys
# %matplotlib inline
import logging
import time
import datetime
import pandas as pd

logging.basicConfig(level=logging.INFO)
FLAGS = tf.app.flags.FLAGS

mean_pixel = [103.94, 116.78, 123.68]

class Model:
	def __init__(self):
		tf.reset_default_graph()

		self.X = tf.placeholder(tf.float32, [None, 224, 224, 3])
		self.y = tf.placeholder(tf.int64, [None])
		self.is_training = tf.placeholder(tf.bool)
		self.setup_log()
		self.setup_system()

	# Sets up the graph
	def setup_system(self):
		self.y_out = self.vgg16(self.X, drop_rate=FLAGS.dropout, reg=FLAGS.reg, is_training=self.is_training)

		# correct_scores = tf.gather_nd(self.y_out, tf.stack((tf.range(self.X.shape[0]), self.y), axis=1))
		a = tf.range(10, dtype=tf.int64)
		b = tf.stack((a, self.y), axis=1)
		correct_scores = tf.gather_nd(self.y_out, b)

		loss = tf.reduce_mean(correct_scores)
		grads = tf.gradients(loss, self.X)[0]
		self.grads = tf.reduce_max(abs(grads), axis=-1)


		total_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.y_out)
		self.mean_loss = tf.reduce_mean(total_loss)

		global_step = tf.Variable(0, trainable=False)
		learning_rate = tf.train.exponential_decay(FLAGS.start_learning_rate, global_step,
												   FLAGS.data_size / FLAGS.batch_size, 0.9, staircase=True)
		# batch normalization in tensorflow requires this extra dependency
		# extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		# with tf.control_dependencies(extra_update_ops):
		#     train_step = optimizer.minimize(self.mean_loss, global_step=global_step)
		optimizer = tf.train.AdamOptimizer(learning_rate)
		self.train_step = optimizer.minimize(self.mean_loss, global_step=global_step)

		self.saver = tf.train.Saver()

		self.correct_prediction = tf.equal(tf.argmax(self.y_out, 1), self.y)
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

		self.softmax_y = tf.nn.softmax(self.y_out)

	# Creates log files and adds description
	def setup_log(self):
		description = input('Please add description for the model: ')
		start_time = '{:%m%d_%H%M%S}'.format(datetime.datetime.now())
		self.index_file = 'logs/index_log' + start_time + '.txt'
		self.epoch_file = 'logs/epoch_log' + start_time + '.txt'
		remark = description + '\n' + ' learn rate: {} \n regularization: {} \n batch size: {} \n data_size: {} \n dropout: {} \n'.format(FLAGS.start_learning_rate, FLAGS.reg, FLAGS.batch_size, FLAGS.data_size, FLAGS.dropout)

		with open (self.index_file, 'a') as f_index:
			f_index.write(remark)
		with open (self.epoch_file, 'a') as f_epoch:
			f_epoch.write(remark)

	# Creates net structure
	def vgg16(self, X, drop_rate=0.5, reg = 1e-2, is_training=None):
		reg_func = lambda t: reg * tf.nn.l2_loss(t)
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
		# fc1 = tf.layers.dense(pool5_flat, units=4096, activation = tf.nn.relu, kernel_regularizer = reg_func)
		# fc2 = tf.layers.dense(fc1, units=4096, activation=tf.nn.relu, kernel_regularizer = reg_func)
		fc1 = tf.layers.dense(pool5_flat, units=4096, activation = tf.nn.relu, kernel_regularizer = reg_func)
		dropout1 = tf.layers.dropout(fc1, rate=drop_rate, training=is_training)
		fc2 = tf.layers.dense(dropout1, units=4096, activation=tf.nn.relu, kernel_regularizer = reg_func)
		dropout2 = tf.layers.dropout(fc2, rate=drop_rate, training=is_training)
		logits = tf.layers.dense(dropout2, units=10, kernel_regularizer = reg_func)
	#     logits = tf.layers.dense(fc2, units=10)

		return logits

	# Runs an epoch
	def run_epoch(self, session, Xd, yd, e, variables, indicies, batch_size=64,
				training_now=True, use_save=True, plot_losses=True):
		index = 0
		correct = 0
		losses = []
		total_loss = 0
		for i in range(int(math.ceil((Xd.shape[0] / batch_size)))):
			start_idx = (i * batch_size) % Xd.shape[0]
			idx = indicies[start_idx: start_idx + batch_size]
			feed_dict = {self.X: Xd[idx,:], self.y: yd[idx], self.is_training: training_now}
			actual_batch_size = yd[idx].shape[0]

			# Computes loss and correct predictions
			# and (if given) perform a training step
			if training_now:
				loss, corr, _ = session.run(variables, feed_dict=feed_dict)
			else:
				loss, corr, _ = session.run(variables, feed_dict=feed_dict)
			losses.append(loss)
			correct += np.sum(corr)
			total_loss += loss * actual_batch_size
			index += actual_batch_size

			if training_now and index % FLAGS.print_every == 0:
				index_res = '{}\tloss: {:.5g}\taccuracy: {:.5g}\tE{}'.format(index, loss, np.sum(corr) / actual_batch_size, e + 1)
				print(index_res)
				if index % FLAGS.log_every == 0:
					time_now = '   {:%m%d_%H%M%S}\n'.format(datetime.datetime.now())
					with open (self.index_file, 'a') as f_index:
						f_index.write(index_res)
						f_index.write(time_now)

		total_correct = correct / Xd.shape[0]
		total_loss /= Xd.shape[0]
		if training_now:
			if plot_losses:
				plt.plot(losses)
				plt.grid(True)
				plt.title('Epoch {} Loss'.format(e + 1))
				plt.xlabel('minibatch number')
				plt.ylabel('minibatch loss')
				plt.show()
			if use_save:
				self.saver.save(session, os.path.join('.', 'saves/model'), global_step=e + 1)
		else:
			time_now = '   {:%m%d_%H%M%S}\n'.format(datetime.datetime.now())
			print('!!!!!!!!!!!!!!!!!')
			epoch_res = '\nEpoch # {}\nVal Loss: {:.5g}\nVal Accuracy:{:.5g}\n'.format(e + 1, total_loss, total_correct)
			print(epoch_res)
			with open (self.epoch_file, 'a') as f_epoch:
				f_epoch.write(epoch_res)
				f_epoch.write(time_now)

	# Runs the model
	def run_model(self, session, dataset=None, epochs=1, batch_size=64, use_save=True, plot_losses=False, testing=False, get_vis=False):
		if get_vis:
			plt.rcParams['figure.figsize'] = (10.0, 10.0) # set default size of plots
			plt.rcParams['image.interpolation'] = 'nearest'
			# plt.rcParams['image.cmap'] = 'gray'
			X_train, y_train, X_val, y_val = dataset
			# mask = np.arange(5)
			mask = [0, 76, 150, 238, 319, 402, 475, 558, 630, 687]

			mask = np.asarray(mask)
			Xm = X_train[mask]
			ym = y_train[mask]

			feed_dict = {self.X: Xm, self.y: ym, self.is_training: False}
			saliency = session.run(self.grads, feed_dict)

			for i in range(mask.size):
				plt.subplot(2, mask.size, i + 1)
				plt.imshow(Xm[i])
				# plt.imshow(deprocess_image(Xm[i]))
				plt.axis('off')
				plt.title(ym[i])
				# plt.title(class_names[ym[i]])
				plt.subplot(2, mask.size, mask.size + i + 1)
				plt.title(mask[i])
				plt.imshow(saliency[i], cmap=plt.cm.hot)
				plt.axis('off')
				plt.gcf().set_size_inches(10, 4)
			plt.savefig('pic.png')
			return

		if testing:
			X_test = np.load('../data/test_data.npy')
			# normalize images by subtracting mean
			for c in range(3):
				X_test[:, :, :, c] = X_test[:, :, :, c] - mean_pixel[c]

			output_y = np.zeros((X_test.shape[0], 10))
			print('test set length:{}'.format(X_test.shape[0]))
			indicies = np.arange(X_test.shape[0])
			for i in range(int(math.ceil((X_test.shape[0] / batch_size)))):
				start_idx = (i * batch_size) % X_test.shape[0]
				idx = indicies[start_idx: start_idx + batch_size]
				# print(X_test[idx,:].shape)
				feed_dict = {self.X: X_test[idx,:], self.is_training: False}

				# Computes loss and correct predictions
				# and (if given) perform a training step
				temp = session.run([self.softmax_y], feed_dict=feed_dict)
				output_y[start_idx: start_idx + batch_size, :] = temp[0]
				print('finished {}'.format(start_idx + batch_size))

			rows = np.load('../data/test_data_id.npy')
			cols = ['c0', 'c1', 'c2', 'c3','c4', 'c5', 'c6', 'c7', 'c8', 'c9']
			# result = pd.DataFrame(output_y, index = rows, columns = cols)
			result = pd.DataFrame(output_y, index = rows, columns = cols)
			now = str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))
			result.to_csv('test_result_%s.csv' % now, index=True, header=True, sep=',')


			# X_val = np.load(os.path.join('..', 'data', 'train_data_' + str(22424) + '.npy'))
			# y_val = np.load(os.path.join('..', 'data', 'train_label_' + str(22424) + '.npy'))
			# X_val = X_val[20441:]
			# y_val = y_val[20441:]
			# output_val_y = np.zeros(X_val.shape[0])
			# print('small validation set length:{}'.format(X_val.shape[0]))
			# indicies = np.arange(X_val.shape[0])
			# for i in range(int(math.ceil((X_val.shape[0] / batch_size)))):
			#   start_idx = (i * batch_size) % X_val.shape[0]
			#   idx = indicies[start_idx: start_idx + batch_size]
			#   feed_dict = {self.X: X_val[idx,:], self.is_training: False}
			#   temp = session.run([self.softmax_y], feed_dict=feed_dict)[0]
			#   output_val_y[start_idx: start_idx + batch_size] = np.argmax(temp, axis = 1)
			#   print('finished val {}'.format(start_idx + batch_size))
			#
			# val_result = pd.DataFrame({'y_val':y_val, 'output_val_y': output_val_y})
			# now = str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))
			# val_result.to_csv('val_result_%s.csv' % now, header=True, sep=',')
			return


		X_train, y_train, X_val, y_val = dataset
		# shuffle indicies
		train_indicies = np.arange(X_train.shape[0])
		np.random.seed(0)
		np.random.shuffle(train_indicies)
		val_indicies = np.arange(X_val.shape[0])

		train_vars = [self.mean_loss, self.correct_prediction, self.train_step]
		val_vars = [self.mean_loss, self.correct_prediction, self.accuracy]

		for e in range(epochs):
			# Train
			self.run_epoch(session, X_train, y_train, e, train_vars, train_indicies, batch_size=batch_size,
				training_now=True, use_save=True, plot_losses=plot_losses)
			# Evaluate
			if FLAGS.eval_every_epoch:
				self.run_epoch(session, X_val, y_val, e, val_vars, val_indicies, batch_size=batch_size,
					training_now=False, use_save=False, plot_losses=plot_losses)

		if not FLAGS.eval_every_epoch:
			self.run_epoch(session, X_val, y_val, e, val_vars, val_indicies, batch_size=batch_size,
					training_now=False, use_save=False, plot_losses=plot_losses)
