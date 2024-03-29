from __future__ import division
import numpy as np
import tensorflow as tf
import math
import os
import matplotlib.pyplot as plt
import sys
%matplotlib inline
import logging
import datetime
import time
logging.basicConfig(level=logging.INFO)

def load_data(debug=True, data_size=1000):
    if debug:
        X_train = np.load(os.path.join('.', 'data', 'train_data_' + data_size + '.npy'))
        y_train = np.load(os.path.join('.', 'data', 'train_label_' + data_size + '.npy'))
    else:
        X_train = np.load(os.path.join('.', 'data', 'train_data.' + data_size + '.npy'))
        y_train = np.load(os.path.join('.', 'data', 'train_label.' + data_size + '.npy'))
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

data_size = 1000
X_train, y_train, X_val, y_val = load_data(debug=True, data_size)
print(sys.getsizeof(X_train), sys.getsizeof(y_train), sys.getsizeof(X_val), sys.getsizeof(y_val))
print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)

def vgg16(X, y, drop_rate=0.5, is_training=None):
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


def run_model(session, predict, loss_val, Xd, yd,
              epochs=1, batch_size=64, print_every=100,
              training=None, plot_losses=False, saver=None):
    # have tensorflow compute accuracy
    correct_prediction = tf.equal(tf.argmax(predict, 1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # shuffle indicies
    train_indicies = np.arange(Xd.shape[0])
    np.random.shuffle(train_indicies)

    training_now = training is not None

    # setting up variables we want to compute (and optimizing)
    # if we have a training function, add that to things we compute
    variables = [loss_val, correct_prediction, accuracy]
    if training_now:
        variables[-1] = training

    # counter
    iter_cnt = 0
    for e in range(epochs):
        # keep track of losses and accuracy
        correct = 0
        losses = []
        total_loss = 0
        # make sure we iterate over the dataset once
        for i in range(int(math.ceil((Xd.shape[0] / batch_size)))):
            # generate indicies for the batch
            start_idx = (i * batch_size) % Xd.shape[0]
            idx = train_indicies[start_idx: start_idx + batch_size]

            feed_dict = {X: Xd[idx,:], y: yd[idx], is_training: training_now}
            actual_batch_size = yd[idx].shape[0]

            # have tensorflow compute loss and correct predictions
            # and (if given) perform a training step
            loss, corr, _ = session.run(variables, feed_dict=feed_dict)

            # aggregate performance stats
            losses.append(loss)
            correct += np.sum(corr)
            total_loss += loss * actual_batch_size

            # print every now and then
            if training_now and (iter_cnt % print_every) == 0:
                print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}"\
                      .format(iter_cnt, loss, np.sum(corr) / actual_batch_size))
            iter_cnt += 1
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
        if saver:
            saver.save(session, os.path.join('.', 'saves/model'), global_step=e + 1)
    return total_loss,total_correct

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

tf.reset_default_graph()

X = tf.placeholder(tf.float32, [None, 224, 224, 3])
y = tf.placeholder(tf.int64, [None])
is_training = tf.placeholder(tf.bool)
y_out = vgg16(X, y, drop_rate=0.5, is_training=is_training)

total_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=y_out)
mean_loss = tf.reduce_mean(total_loss)

# params
starter_learning_rate = 1e-5
use_saves = True
batch_size = 64
epoch_num = 1

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           1000, 0.9, staircase=True)
optimizer = tf.train.AdamOptimizer(learning_rate)
train_step = optimizer.minimize(mean_loss, global_step=global_step)

# batch normalization in tensorflow requires this extra dependency
# extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
# with tf.control_dependencies(extra_update_ops):
#     train_step = optimizer.minimize(mean_loss, global_step=global_step)
saver = tf.train.Saver()
if not os.path.exists('saves'):
    os.makedirs('saves')
if not os.path.exists('logs'):
    os.makedirs('logs')

start_time = '{:%m%d_%H%M%S}'.format(datetime.datetime.now())
index_file = 'logs/index_log' + start_time + '.txt'
epoch_file = 'logs/epoch_log' + start_time + '.txt'
description = input('Please add description for the model: ')
remark = description + '\n' + 'learn rate: {} \n batch size: {} \n data_size: {} \n dropout: {} \n'.format(starter_learning_rate, batch_size, data_size, drop_rate)

with tf.Session() as sess:
    if use_saves:
        initialize_model(sess, saver, '.')
    else:
        sess.run(tf.global_variables_initializer())
    print('Training')
    run_model(sess, y_out, mean_loss, X_train, y_train, epoch_num, batch_size, 
              print_every=100, training=train_step, plot_losses=True, saver=saver)
    print('Validation')
    run_model(sess, y_out, mean_loss, X_val, y_val, epochs=1, batch_size=batch_size)
