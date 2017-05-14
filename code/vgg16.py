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


def run_model(session, predict, loss_val, Xd, yd,
              epochs=1, batch_size=64, print_every=100,
              training=None, plot_losses=False):
    # have tensorflow compute accuracy
    correct_prediction = tf.equal(tf.argmax(predict,1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # shuffle indicies
    train_indicies = np.arange(Xd.shape[0])
    np.random.shuffle(train_indicies)

    training_now = training is not None

    # setting up variables we want to compute (and optimizing)
    # if we have a training function, add that to things we compute
    variables = [mean_loss,correct_prediction,accuracy]
    if training_now:
        variables[-1] = training

    # counter
    iter_cnt = 0
    for e in range(epochs):
        # keep track of losses and accuracy
        correct = 0
        losses = []
        # make sure we iterate over the dataset once
        for i in range(int(math.ceil(Xd.shape[0]/batch_size))):
            # generate indicies for the batch
            start_idx = (i*batch_size)%X_train.shape[0]
            idx = train_indicies[start_idx:start_idx+batch_size]

            # create a feed dictionary for this batch
            feed_dict = {X: Xd[idx,:],
                         y: yd[idx],
                         is_training: training_now }
            # get batch size
            actual_batch_size = yd[i:i+batch_size].shape[0]

            # have tensorflow compute loss and correct predictions
            # and (if given) perform a training step
            loss, corr, _ = session.run(variables,feed_dict=feed_dict)

            # aggregate performance stats
            losses.append(loss*actual_batch_size)
            correct += np.sum(corr)

            # print every now and then
            if training_now and (iter_cnt % print_every) == 0:
                print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}"\
                      .format(iter_cnt,loss,np.sum(corr)/actual_batch_size))
            iter_cnt += 1
        total_correct = correct/Xd.shape[0]
        total_loss = np.sum(losses)/Xd.shape[0]
        print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}"\
              .format(total_loss,total_correct,e+1))
        if plot_losses:
            plt.plot(losses)
            plt.grid(True)
            plt.title('Epoch {} Loss'.format(e+1))
            plt.xlabel('minibatch number')
            plt.ylabel('minibatch loss')
            plt.show()
    return total_loss,total_correct


tf.reset_default_graph()

X = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.int64, [None])
is_training = tf.placeholder(tf.bool)
y_out = my_model(X,y,is_training)

total_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = y_out)
mean_loss = tf.reduce_mean(total_loss)

global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 1e-3
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           800, 0.9, staircase=True)
optimizer = tf.train.AdamOptimizer(learning_rate)

# batch normalization in tensorflow requires this extra dependency
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(extra_update_ops):
    train_step = optimizer.minimize(mean_loss, global_step=global_step)


with tf.Session() as sess:
    with tf.device("/cpu:0"): #"/cpu:0" or "/gpu:0"
        sess.run(tf.global_variables_initializer())
        print('Training')
        run_model(sess,y_out,mean_loss,X_train,y_train,1,64,100,train_step,True)
        print('Validation')
        run_model(sess,y_out,mean_loss,X_val,y_val,1,64)
