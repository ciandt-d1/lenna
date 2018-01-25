# -*- coding: utf-8 -*-

import os

import numpy as np
import pandas as pd
import tensorflow as tf
from google.cloud import storage
FLAGS = tf.app.flags.FLAGS
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png')


def split_storage_name(full_path):
    """Split full_path to return bucket name and file path"""

    full_path_split = "".join(full_path.split(
        "gs://")[1:])  # Remove 'gs://' prefix
    bucket_and_filename = full_path_split.split("/")
    bucket_name = bucket_and_filename[0]
    filename = "/".join(bucket_and_filename[1:])
    tf.logging.debug(filename)
    return bucket_name, filename


def on_storage(full_path):
    return full_path.startswith("gs://")


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    tf.logging.info('Blob {} downloaded to {}.'.format(
        source_blob_name,
        destination_file_name))
    tf.logging.info('Full path of {} {}'.format(
        destination_file_name, os.path.abspath(destination_file_name)))


def read_csv(full_filename):
    """Return metadata as pandas DataFrame whether on storage or locally"""
    if on_storage(full_filename):
        tf.logging.info("Reading metadata from storage")
        local_filename = "tmp.csv"
        bucket_name, filename = split_storage_name(full_filename)
        download_blob(bucket_name, filename, local_filename)
        metadata = pd.read_csv(os.path.abspath(local_filename))
        tf.logging.info("HEAD: {}".format(metadata.head()))
        # delete file immediately after reading it
        os.remove(os.path.abspath(local_filename))
    else:
        tf.logging.info("Reading metadata locally")
        metadata = pd.read_csv(full_filename)

    return metadata


def list_tfrecord(regex):
    """Return list of files given a regex"""

    list_op = tf.train.match_filenames_once(regex)
    init_ops = (tf.global_variables_initializer(),
                tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_ops)
        files = sess.run(list_op)

    return files


def get_dataset_len(tfrecord_list):
    """Approximate dataset length by the length of the first x-th tfrecords"""
    options = tf.python_io.TFRecordOptions(
        tf.python_io.TFRecordCompressionType.GZIP)
    return len(tfrecord_list) * np.mean(
        list(sum(1 for _ in tf.python_io.tf_record_iterator(path, options)) for path in tfrecord_list[0:10]))


def configure_optimizer(learning_rate):
    """Configures the optimizer used for training.
    Args:
        learning_rate: A scalar or `Tensor` learning rate.
    Returns:
        An instance of an optimizer.
    Raises:
        ValueError: if FLAGS.optimizer is not recognized.
    """
    if FLAGS.optimizer == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(
            learning_rate,
            rho=FLAGS.adadelta_rho,
            epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(
            learning_rate,
            initial_accumulator_value=FLAGS.adagrad_initial_accumulator_value)
    elif FLAGS.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(
            learning_rate,
            beta1=FLAGS.adam_beta1,
            beta2=FLAGS.adam_beta2,
            epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(
            learning_rate,
            learning_rate_power=FLAGS.ftrl_learning_rate_power,
            initial_accumulator_value=FLAGS.ftrl_initial_accumulator_value,
            l1_regularization_strength=FLAGS.ftrl_l1,
            l2_regularization_strength=FLAGS.ftrl_l2)
    elif FLAGS.optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(
            learning_rate,
            momentum=FLAGS.momentum,
            name='Momentum')
    elif FLAGS.optimizer == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate,
            decay=FLAGS.rmsprop_decay,
            momentum=FLAGS.rmsprop_momentum,
            epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError('Optimizer [%s] was not recognized', FLAGS.optimizer)
    return optimizer


def configure_learning_rate(num_samples_per_epoch, global_step):
    """Configures the learning rate.
    Args:
        num_samples_per_epoch: The number of samples in each epoch of training.
        global_step: The global_step tensor.
    Returns:
        A `Tensor` representing the learning rate.
    Raises:
        ValueError: if
    """
    decay_steps = int(num_samples_per_epoch / FLAGS.batch_size *
                      FLAGS.num_epochs_per_decay)
    if FLAGS.sync_replicas:
        decay_steps /= FLAGS.replicas_to_aggregate

    if FLAGS.learning_rate_decay_type == 'exponential':
        return tf.train.exponential_decay(FLAGS.learning_rate,
                                          global_step,
                                          decay_steps,
                                          FLAGS.learning_rate_decay_factor,
                                          staircase=True,
                                          name='exponential_decay_learning_rate')
    elif FLAGS.learning_rate_decay_type == 'fixed':
        return tf.constant(FLAGS.learning_rate, name='fixed_learning_rate')
    elif FLAGS.learning_rate_decay_type == 'polynomial':
        return tf.train.polynomial_decay(FLAGS.learning_rate,
                                         global_step,
                                         decay_steps,
                                         FLAGS.end_learning_rate,
                                         power=1.0,
                                         cycle=False,
                                         name='polynomial_decay_learning_rate')
    else:
        raise ValueError('learning_rate_decay_type [%s] was not recognized',
                         FLAGS.learning_rate_decay_type)


def get_variables_to_train():
    """Returns a list of variables to train.
    Returns:
      A list of variables to train by the optimizer.
    """
    if FLAGS.trainable_scopes is None:
        return tf.trainable_variables()
    else:
        scopes = [scope.strip() for scope in FLAGS.trainable_scopes.split(',')]

    variables_to_train = []
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)
    return variables_to_train


def get_variables_to_restore():
    """Returns a list of variables to restore.
    Returns:
      A list of variables to be restored from the checkpoint.
    """
    exclusions = []
    if FLAGS.checkpoint_exclude_scopes:
        exclusions = [scope.strip()
                      for scope in FLAGS.checkpoint_exclude_scopes.split(',')]

    tf.logging.info("Exclusions: {}".format(exclusions))
    variables_to_restore = []
    for var in tf.contrib.framework.get_model_variables():
        #tf.logging.info("Var: {}".format(var))
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):   
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)
    return variables_to_restore
