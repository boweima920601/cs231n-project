import tensorflow as tf
import logging
import sys
import os
import numpy as np
import math
from vgg16obj import Model
logging.basicConfig(level=logging.INFO)

# Tune all the hyper params here
tf.app.flags.DEFINE_float("start_learning_rate", 1e-5, "Learning rate.")
tf.app.flags.DEFINE_float("dropout", 0.5, "Fraction of units randomly dropped")
tf.app.flags.DEFINE_float("reg", 5e-4, "L2 regularization to each layer")
tf.app.flags.DEFINE_integer("batch_size", 64, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 10, "Number of epochs to train.")

tf.app.flags.DEFINE_integer("data_size", 22424, "The number of training samples")
tf.app.flags.DEFINE_boolean("is_debug", False, "Use smaller dataset for debug")
tf.app.flags.DEFINE_boolean("use_save", True, "Save model into checkpoint")

tf.app.flags.DEFINE_integer("print_every", 64 * 5, "How many iterations to do per print.")
tf.app.flags.DEFINE_integer("log_every", 64 * 5, "How many iterations to do per log.")
tf.app.flags.DEFINE_boolean("eval_every_epoch", True, "Whether evaluate after each epoch")

tf.app.flags.DEFINE_boolean("is_testing", False, "Whether we are testing")

FLAGS = tf.app.flags.FLAGS

mean_pixel = [103.94, 116.78, 123.68]

# Inits the model and reads checkpoints if possible
def initialize_model(session, saver, train_dir):
	ckpt = tf.train.get_checkpoint_state(train_dir)
	v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
	if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
		logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
		saver.restore(session, ckpt.model_checkpoint_path)
	else:
		logging.info("Created model with fresh parameters.")
		session.run(tf.global_variables_initializer())
		pretrained_vgg16 = np.load('../data/vgg16_weights.npz')
		keys = sorted(pretrained_vgg16.keys())
		for i, k in enumerate(keys[:-2]): # exclude the last fc layer
		# for i, k in enumerate(keys[:-4]): # exclude the last three fc layer
			# print (i, k, np.shape(pretrained_vgg16[k]))
			session.run(tf.trainable_variables()[i].assign(pretrained_vgg16[k]))
		logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))

# Loads the data
def load_data(debug=True, data_size=1000):
	if not debug:
		data_size = FLAGS.data_size
	X_train = np.load(os.path.join('..', 'data', 'train_data_' + str(data_size) + '.npy'))
	y_train = np.load(os.path.join('..', 'data', 'train_label_' + str(data_size) + '.npy'))
	train_indicies = np.arange(X_train.shape[0])

	# normalize images by subtracting mean
	for c in range(3):
		X_train[:, :, :, c] = X_train[:, :, :, c] - mean_pixel[c]
	# np.random.shuffle(train_indicies)
	# driver_list = [0,725,1548,2424,3299,4377,5614,6847,8073,9269,10117,10768,11373,11964,
	# 12688,13523,14534,15324,16244,16984,17778,18587,19407,20441,20787,21601,22424]
	# val_list = np.random.choice(26, 3, replace = False)
	# in_val = []
	# for i in val_list:
	# 	in_val = np.append(in_val, train_indicies[driver_list[i]:driver_list[i+1]])
	in_val = train_indicies[20441:]
	in_train = train_indicies[:20441]
	# in_val = list(map(int, in_val))
	# in_train = list(map(int, in_train))
	X_val = X_train[in_val]
	y_val = y_train[in_val]
	X_train = X_train[in_train]
	y_train = y_train[in_train]
	return X_train, y_train, X_val, y_val

def main(_):
	# Creates directory for checkpoint and logs
	if not os.path.exists('saves'):
		os.makedirs('saves')
	if not os.path.exists('logs'):
		os.makedirs('logs')

	if not FLAGS.is_testing:
		dataset = load_data(FLAGS.is_debug)
		X_train, y_train, X_val, y_val = dataset
		print('data size:')
		print(sys.getsizeof(X_train), sys.getsizeof(y_train), sys.getsizeof(X_val), sys.getsizeof(y_val))
		print('data shape:')
		print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)

	# Creates the model
	model = Model()

	with tf.Session() as sess:
		# Inits the model
		initialize_model(sess, model.saver, './saves/')

		if not FLAGS.is_testing:
			print('Training')

			# Runs the model
			model.run_model(sess, dataset, epochs=FLAGS.epochs, batch_size=FLAGS.batch_size, use_save=True, plot_losses=False)
		if FLAGS.is_testing:
			model.run_model(sess, batch_size=FLAGS.batch_size, testing=True)


if __name__ == "__main__":
	tf.app.run()
