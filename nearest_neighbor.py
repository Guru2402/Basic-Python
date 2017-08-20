import print_function from __future__ 

import numpy as np
import tensorflow as tf

import input_data from tensorflow.examples.tutorials.mnist
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

Xtr, Ytr = mnist.train.next_batch(5000) #5000 for training (nn candidates)
Xte, Yte = mnist.test.next_batch(200) #200 for testing

# Graph Input
xtr = tf.placeholder("float", [None, 784])
xte = tf.placeholder("float", [784])

distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), reduction_indices=1)

pred = tf.arg_min(distance, 0)

accuracy = 0.

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    for i in range(len(Xte)):
        nn_index = sess.run(pred, feed_dict={xtr: Xtr, xte: Xte[i, :]})
        print("Test", i, "Prediction:", np.argmax(Ytr[nn_index]), \
            "True Class:", np.argmax(Yte[i]))
        if np.argmax(Ytr[nn_index]) == np.argmax(Yte[i]):
            accuracy += 1./len(Xte)
    print("Done!")
    print("Accuracy:", accuracy)
