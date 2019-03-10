import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd

from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


def main(_):
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, W) + b

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    optimizer = tf.train.AdagradOptimizer(learning_rate=0.05).minimize(cross_entropy)  # AdagradOptimizer(learning_rate=0.005).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) * 100


    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # for logging into tesnorboard
    tf.summary.scalar("MSE Loss", cross_entropy)
    tf.summary.scalar("Accuracy", accuracy)
    train_writer = tf.summary.FileWriter('./log/train', graph=tf.get_default_graph())
    test_writer = tf.summary.FileWriter('./log/test', graph=tf.get_default_graph())
    # Train


    for ep in range(50000):
        batch_xs, batch_ys = mnist.train.next_batch(128)


        if ep % 400 == 0:
            train_loss = sess.run([optimizer, cross_entropy], feed_dict={x: batch_xs, y_: batch_ys})
            train_summaries = tf.Summary(value=[

                tf.Summary.Value(tag="MSE Loss", simple_value=train_loss[1])])
            train_writer.add_summary(train_summaries, ep)
        elif ep % 500 == 0:
            test_loss, acc = sess.run([cross_entropy, accuracy], feed_dict={x: batch_xs, y_: batch_ys})
            test_summaries = tf.Summary(value=[
                                                tf.Summary.Value(tag="Accuracy", simple_value=acc),
                                                tf.Summary.Value(tag="MSE Loss", simple_value=test_loss)])
            test_writer.add_summary(test_summaries, ep)

        print('iteration {} train loss {}'.format(ep, train_loss[1]))

    # for ep in range(20000):
    #     batch_xs, batch_ys = mnist.train.next_batch(256)
    #     sess.run([optimizer], feed_dict={x: batch_xs, y_: batch_ys})
    #     loss = sess.run([cross_entropy], feed_dict={x: batch_xs, y_: batch_ys})
    #     if ep % 100 == 0:
    #         train_summaries = tf.Summary(value=[
    #
    #             tf.Summary.Value(tag="MSE Loss", simple_value=loss[0])])
    #         train_writer.add_summary(train_summaries, ep)
    #     elif ep % 150 == 0:
    #         test_x, test_y = mnist.train.next_batch(256)
    #         test_loss, acc = sess.run([cross_entropy, accuracy], feed_dict={x: test_x, y_: test_y})
    #         test_summaries = tf.Summary(value=[
    #             tf.Summary.Value(tag="Accuracy", simple_value=acc),
    #             tf.Summary.Value(tag="MSE Loss", simple_value=test_loss)])
    #         test_writer.add_summary(test_summaries, ep)
    #
    #     print('iteration {}'.format(ep))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
