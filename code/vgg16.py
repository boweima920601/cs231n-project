import numpy as np
import tensorflow as tf
from tf.layers import conv2d, batch_normalization, max_pooling2d, dense, dropout
from tf.nn import relu

def vgg16(X, y, is_training):
    conv1 = conv2d(X, filter = 64, kernel_size = [3, 3], padding = 'same', activation = relu)
    conv2 = conv2d(conv1, filter = 64, kernel_size = [3, 3], padding = 'same', activation = relu)
    pool1 = max_pooling2d(conv2, pool_size = [2, 2], strides = 2)

    conv3 = conv2d(pool1, filter = 128, kernel_size = [3, 3], padding = 'same', activation = relu)
    conv4 = conv2d(conv3, filter = 128, kernel_size = [3, 3], padding = 'same', activation = relu)
    pool2 = max_pooling2d(conv4, pool_size = [2, 2], strides = 2)

    conv5 = conv2d(pool2, filter = 256, kernel_size = [3, 3], padding = 'same', activation = relu)
    conv6 = conv2d(conv5, filter = 256, kernel_size = [3, 3], padding = 'same', activation = relu)
    conv7 = conv2d(conv6, filter = 256, kernel_size = [3, 3], padding = 'same', activation = relu)
    pool3 = max_pooling2d(conv7, pool_size = [2, 2], strides = 2)

    conv8 = conv2d(pool3, filter = 512, kernel_size = [3, 3], padding = 'same', activation = relu)
    conv9 = conv2d(conv8, filter = 512, kernel_size = [3, 3], padding = 'same', activation = relu)
    conv10 = conv2d(conv9, filter = 512, kernel_size = [3, 3], padding = 'same', activation = relu)
    pool4 = max_pooling2d(conv10, pool_size = [2, 2], strides = 2)

    conv11 = conv2d(pool4, filter = 512, kernel_size = [3, 3], padding = 'same', activation = relu)
    conv12 = conv2d(conv11, filter = 512, kernel_size = [3, 3], padding = 'same', activation = relu)
    conv13 = conv2d(conv12, filter = 512, kernel_size = [3, 3], padding = 'same', activation = relu)
    pool5 = max_pooling2d(conv13, pool_size = [2, 2], strides = 2)

    pool5_flat = tf.reshape(pool5, [-1, ?????])
    fc1 = dense(pool5_flat, units=???, activation = relu)
    fc2 = dense(fc1, units=???, activation = relu)
    fc3 = dense(fc2, units=???, activation = relu)
    
    # dropout1 = dropout(fc3, rate=0.5, training=is_training)
    # logits = dense(dropout1, units=10)
    logits = dense(fc3, units=10)

    return logits
