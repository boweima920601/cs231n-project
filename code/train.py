import tensorflow as tf
import logging
import sys
import os
import numpy as np
import math
from vgg16obj import Model

logging.basicConfig(level=logging.INFO)

tf.app.flags.DEFINE_float("start_learning_rate", 1e-5, "Learning rate.")
tf.app.flags.DEFINE_float("dropout", 0.5, "Fraction of units randomly dropped")
tf.app.flags.DEFINE_integer("batch_size", 40, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 1, "Number of epochs to train.")

tf.app.flags.DEFINE_integer("print_every", 20, "How many iterations to do per print.")
tf.app.flags.DEFINE_integer("log_every", 20, "How many iterations to do per log.")

tf.app.flags.DEFINE_string("data_dir", "data", "SQuAD directory (default ./data/squad)")
tf.app.flags.DEFINE_string("train_dir", "train", "Training directory to save the model parameters (default: ./train).")
tf.app.flags.DEFINE_integer("data_size", 100, "The number of training samples")
tf.app.flags.DEFINE_integer("is_debug", True, "Use smaller dataset for debug")
tf.app.flags.DEFINE_integer("use_save", True, "Save model into checkpoint")

FLAGS = tf.app.flags.FLAGS


def initialize_model(session, saver, train_dir):
	ckpt = tf.train.get_checkpoint_state(train_dir)
	v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
	if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
		logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
		saver.restore(session, ckpt.model_checkpoint_path)
	else:
		logging.info("Created model with fresh parameters.")
		session.run(tf.global_variables_initializer())
		logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))

def load_data(debug=True, data_size=1000):
	if debug:
		X_train = np.load(os.path.join('..', FLAGS.data_dir, 'train_data_' + str(data_size) + '.npy'))
		y_train = np.load(os.path.join('..', FLAGS.data_dir, 'train_label_' + str(data_size) + '.npy'))
	else:
		X_train = np.load(os.path.join('..', FLAGS.data_dir, 'train_data.' + str(data_size) + '.npy'))
		y_train = np.load(os.path.join('..', FLAGS.data_dir, 'train_label.' + str(data_size) + '.npy'))
	train_indicies = np.arange(X_train.shape[0])
	np.random.shuffle(train_indicies)
	num_training = int(math.ceil(X_train.shape[0] * 0.9))
	in_train = train_indicies[:num_training]
	in_val = train_indicies[num_training:]
	X_val = X_train[in_val]
	y_val = y_train[in_val]
	X_train = X_train[in_train]
	y_train = y_train[in_train]
	return X_train, y_train, X_val, y_val

def main(_):
	if not os.path.exists('saves'):
		os.makedirs('saves')
	if not os.path.exists('logs'):
		os.makedirs('logs')

	dataset = load_data(FLAGS.is_debug, FLAGS.data_size)
	X_train, y_train, X_val, y_val = dataset
	print(sys.getsizeof(X_train), sys.getsizeof(y_train), sys.getsizeof(X_val), sys.getsizeof(y_val))
	print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)

	model = Model()

	with tf.Session() as sess:
		initialize_model(sess, model.saver, '.')
		print('Training')
		model.run_model(sess, X_train, y_train, epochs=FLAGS.epochs, batch_size=FLAGS.batch_size, training_now=True, use_save=True, plot_losses=True)
		print('Validation')
		model.run_model(sess, X_val, y_val, epochs=1, batch_size=FLAGS.batch_size, training_now=False, use_save=False, plot_losses=False)

if __name__ == "__main__":
	tf.app.run()