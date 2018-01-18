# -*- coding: utf-8 -*-

import numpy as np
import cv2
import tensorflow as tf
import sys

from demographics_architecture import image_size, cnn_architecture

tf.logging.set_verbosity(tf.logging.INFO)

# Set default flags for the output directories
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    flag_name='checkpoint_path', default_value='',
    docstring='Checkpoint path')
tf.app.flags.DEFINE_string(flag_name="network_name",
                           default_value="inception_v4", docstring="Network architecture to use")


# MNIST sample images
IMAGE_URLS = [
    '../dataset/faces/v0.2_Angry_1.jpg',
    '../dataset/faces/v0.2_Angry_2.jpg',
    '../dataset/faces/v0.2_Angry_3.jpg'
]


def predict_from_list():

    # Create placeholders
    X = tf.placeholder(dtype=tf.float32, shape=(
        None, image_size, image_size, 3), name='X')

    # Load net architecture
    Y_age_pred, Y_eth_pred, Y_gender_pred = cnn_architecture(
        X, is_training=False, network_name=FLAGS.network_name)

    saver = tf.train.Saver()

    # Open session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Restore weights
        # if tf.gfile.Exists(FLAGS.checkpoint_path):
        #     saver.restore(sess, FLAGS.checkpoint_path)
        # else:
        #     tf.logging.error("Checkpoint file {} not found".format(FLAGS.checkpoint_path))
        #     sys.exit(0)
        saver.restore(sess, FLAGS.checkpoint_path)

        # Make predictions
        for img_path in IMAGE_URLS:
            tf.logging.info("Predict {}".format(img_path))
            # Read and preprocess image
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB).astype(np.float)
            img /= 255.0
            img = np.expand_dims(cv2.resize(img, (image_size, image_size)),axis=0)

            Y_age_pred_v, Y_gender_pred_v, Y_eth_v = sess.run(
                [Y_age_pred, Y_gender_pred, Y_eth_pred], feed_dict={X: img})

            Y_age_pred_v = np.argmax(Y_age_pred_v,axis=1)[0]
            Y_gender_pred_v = np.argmax(Y_gender_pred_v,axis=1)[0]
            Y_eth_v = np.argmax(Y_eth_v,axis=1)[0]
            tf.logging.info("{} - Age: {} Gender: {} Ethniticy: {}".format(
                img_path, Y_age_pred_v, Y_gender_pred_v, Y_eth_v))



if __name__ == "__main__":
    predict_from_list()