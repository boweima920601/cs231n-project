from __future__ import print_function

import numpy as np
import tensorflow as tf

K = 11

x_train_holder = tf.placeholder("float", [None, 3 * 3 * 512])

x_test_holder = tf.placeholder("float", [3 * 3 * 512])

# Euclidean Distance
distance = tf.negative(tf.reduce_sum(tf.square(x_train_holder - x_test_holder), reduction_indices=1))

# Prediction: Get min distance neighbors
values, indices = tf.nn.top_k(distance, k=K, sorted=False)


M = np.load('../data/pool6_result.npy')

init = tf.global_variables_initializer()

# 79726
# M = M0[: 79726, :]
# del M0

with tf.Session() as sess:
    sess.run(init)

    idx_m = np.zeros((M.shape[0], K))
    dist_m = np.zeros((M.shape[0], K))

    # loop over test data
    for i in range(M.shape[0]):
        # Get nearest neighbor
        v, idx = sess.run([values, indices], feed_dict={x_train_holder: M, x_test_holder: M[i, :]})

        idx_m[i, :] = idx
        dist_m[i, :] = v
        print("i", i)
        print("idx", idx)
        print("v", v)
    np.save('index1.npy', idx_m)
    np.save('dist_m1.npy', dist_m)
    print("Done!")