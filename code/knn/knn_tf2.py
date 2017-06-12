from __future__ import print_function

import numpy as np
import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

K = 4
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# In this example, we limit mnist data
Xtr, Ytr = mnist.train.next_batch(55000)  # whole training set
Xte, Yte = mnist.test.next_batch(10000)  # whole test set

# tf Graph Input
xtr = tf.placeholder("float", [None, 784])
ytr = tf.placeholder("float", [None, 10])
xte = tf.placeholder("float", [784])

# Euclidean Distance
distance = tf.negative(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(xtr, xte)), reduction_indices=1)))
# Prediction: Get min distance neighbors
values, indices = tf.nn.top_k(distance, k=K, sorted=False)

nearest_neighbors = []
for i in range(K):
    nearest_neighbors.append(tf.argmax(ytr[indices[i]], 0))

neighbors_tensor = tf.stack(nearest_neighbors)
y, idx, count = tf.unique_with_counts(neighbors_tensor)
pred = tf.slice(y, begin=[tf.argmax(count, 0)], size=tf.constant([1], dtype=tf.int64))[0]

accuracy = 0.

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # loop over test data
    for i in range(len(Xte)):
        # Get nearest neighbor
        nn_index = sess.run(pred, feed_dict={xtr: Xtr, ytr: Ytr, xte: Xte[i, :]})
        # Get nearest neighbor class label and compare it to its true label
        print("Test", i, "Prediction:", nn_index,
             "True Class:", np.argmax(Yte[i]))
        #Calculate accuracy
        if nn_index == np.argmax(Yte[i]):
            accuracy += 1. / len(Xte)
    print("Done!")
    print("Accuracy:", accuracy)