# -*- coding: utf-8 -*-

import numpy as np
import sys
import os
import argparse
import logging

import tensorflow as tf

from cnn_architecture_inception_v4 import cnn_architecture

tf.logging.set_verbosity(tf.logging.INFO)

# Set default flags for the output directories
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    flag_name='checkpoint_path', default_value='',
    docstring='Checkpoint path')
tf.app.flags.DEFINE_string(
    flag_name='output_checkpoint_path', default_value='',
    docstring='Checkpoint path')
tf.app.flags.DEFINE_integer(flag_name='image_size',
                            default_value=200, docstring="Image size")


def load_and_save_ckpt():

    # Create placeholders
    X = tf.placeholder(dtype=tf.float32, shape=(
        None, FLAGS.image_size, FLAGS.image_size, 3), name='input_image')

    # Load net architecture
    net_final = cnn_architecture(X, is_training=False)

    # Add softmax layer
    prediction = tf.argmax(net_final, name="prediction")

    saver = tf.train.Saver()

    # Open session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, FLAGS.checkpoint_path)
        saver.save(sess, FLAGS.output_checkpoint_path)


if __name__ == "__main__":
    load_and_save_ckpt()
