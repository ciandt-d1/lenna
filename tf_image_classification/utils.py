# -*- coding: utf-8 -*-
"""
.. module:: utils
   :platform: Unix
   :synopsis: Utilities

.. moduleauthor:: Rodrigo Pereira <rodrigofp@ciandt.com>

"""

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from google.cloud import storage

from tensorflow.python.framework import ops, dtypes
from tensorflow.python.ops import array_ops, variables

import io

FLAGS = tf.app.flags.FLAGS
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png')


def split_storage_name(full_path):   
    """ Split file path on storage (Google Cloud Storage) to return bucket name and file path

        Args:
            ``full_path`` (string): file path to split

        Returns:
            ``bucket_name`` (string): bucket name
            ``filename`` (string): basename
    """

    full_path_split = "".join(full_path.split(
        "gs://")[1:])  # Remove 'gs://' prefix
    bucket_and_filename = full_path_split.split("/")
    bucket_name = bucket_and_filename[0]
    filename = "/".join(bucket_and_filename[1:])
    tf.logging.debug(filename)
    return bucket_name, filename


def on_storage(full_path):
    """ Check whether or not a file is located on Google Cloud Storage

        Args:
            ``full_path`` (string): file path

        Returns:
            ``flag`` (bool)
            
    """
    return full_path.startswith("gs://")


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """ Downloads a blob from Google Cloud Storage
    
        Args:
            ``bucket_name`` (string)
            ``source_blob_name`` (string)
            ``destination_file_name`` (string) 
        
    """
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
    """ Return dataset metadata as `pandas.DataFrame <https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html>`_
    
        Args:
            ``full_filename`` (string): csv file where each row posses the image names

        Return:
            ``metadata`` (`pandas.DataFrame <https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html>`_): dataset metadata
    
    """
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
    """ Return list of files given a regex
    
        Args:
            ``regex`` (string): File regex. It must follow Google Cloud Storage `Wild Card Convension <https://cloud.google.com/storage/docs/gsutil/addlhelp/WildcardNames>`_

        Returns:
            ``files`` (list): files that match ``regex``

    """

    list_op = tf.train.match_filenames_once(regex)
    init_ops = (tf.global_variables_initializer(),
                tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_ops)
        files = sess.run(list_op)

    return files


def get_dataset_len(tfrecord_list):
    """ Approximate dataset length by the length of the first x-th tfrecords
    
        If you don't know the exact length of you dataset, this helper function will help you.

        Args:
            ``tfrecord_list`` (list): list of all tf-records to estimate dataset length

        Returns:
            Estimated dataset length
    """
    options = tf.python_io.TFRecordOptions(
        tf.python_io.TFRecordCompressionType.GZIP)
    return len(tfrecord_list) * np.mean(
        list(sum(1 for _ in tf.python_io.tf_record_iterator(path, options)) for path in tfrecord_list[0:20]))


def configure_optimizer(learning_rate):
    """ Configures the optimizer used for training.

        To choose the optimzer use tf.flags.FLAGS provided by :mod:~tf_image_classification.train_estimator

        Available optimizers:

        * `Adadelta <https://www.tensorflow.org/api_docs/python/tf/train/AdadeltaOptimizer>`_
        * `Adagrad <https://www.tensorflow.org/api_docs/python/tf/train/AdagradOptimizer>`_
        * `Adam <https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer>`_
        * `Ftrl <https://www.tensorflow.org/api_docs/python/tf/train/FtrlOptimizer>`_
        * `Momentum <https://www.tensorflow.org/api_docs/python/tf/train/MomentumOptimizer>`_
        * `RMSProp <https://www.tensorflow.org/api_docs/python/tf/train/RMSPropOptimizer>`_
        * `SGD <https://www.tensorflow.org/api_docs/python/tf/train/GradientDescentOptimizer>`_    
    
        Args:
            learning_rate: A scalar(float) or `tf.Tensor <https://www.tensorflow.org/api_docs/python/tf/Tensor>`_. You can configure  
        
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
    """ Configures the learning rate.

        To choose the learning rate type use tf.flags.FLAGS provided by :mod:~tf_image_classification.train_estimator

        Available learning rate:

        * Fixed. Just a scalar
        * `Exponential decay <https://www.tensorflow.org/api_docs/python/tf/train/exponential_decay>`_
        * `Polynomial decay <https://www.tensorflow.org/api_docs/python/tf/train/polynomial_decay>`_

        Args:
            ``num_samples_per_epoch`` (int) : The number of samples in each epoch of training.
            ``global_step`` (`tf.Tensor <https://www.tensorflow.org/api_docs/python/tf/Tensor>`_): The global_step tensor.
        
        Returns:
            A `tf.Tensor <https://www.tensorflow.org/api_docs/python/tf/Tensor>`_ representing the learning rate.
        
        Raises:
            ValueError: learning_rate not defined
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
    """ Returns a list of variables to train.
    
        Returns:
            A list of variables to train.
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
    """ Returns a list of variables to restore.
    
        Returns:
            A list of variables to be restored from the checkpoint.
    """
    exclusions = []
    if FLAGS.checkpoint_exclude_scopes:
        exclusions = [scope.strip()
                      for scope in FLAGS.checkpoint_exclude_scopes.split(',')]

    tf.logging.info("Exclusions: {}".format(exclusions))
    variables_to_restore = []    
    all_variables = set(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) + tf.get_collection(tf.GraphKeys.MODEL_VARIABLES))
            
    for var in all_variables: 
        
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):   
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)
    return variables_to_restore



def _createLocalVariable(name, shape, collections=None,
                                validate_shape=True,
                                dtype=dtypes.float32):
        """ Creates a new local variable."""
        
        collections = list(collections or [])
        collections += [ops.GraphKeys.LOCAL_VARIABLES]
        return variables.Variable(
            initial_value=array_ops.zeros(shape, dtype=dtype),
            name=name,
            trainable=False,
            collections=collections,
            validate_shape=validate_shape)

def streaming_confusion_matrix(name, label, prediction, num_classes=None):
    """ Compute a streaming confusion matrix
    
        Args:
            ``label`` (string): True labels
            ``prediction`` (string): Predicted labels            
            ``num_classes`` (int): Number of labels for the confusion matrix


        Returns:
            ``percentConfusionMatrix`` (`tf.Tensor <https://www.tensorflow.org/api_docs/python/tf/Tensor>`_): Confusion matrix tensor
            ``updateOp`` (`tf.Operation <https://www.tensorflow.org/api_docs/python/tf/Operation>`_): Op that update confusion matrix through batches
    """
    # Compute a per-batch confusion
    batch_confusion_name = "batch_confusion_"+name
    batch_confusion = tf.confusion_matrix(label, prediction,
                                            num_classes=num_classes,
                                            name=batch_confusion_name)

    count = _createLocalVariable(None, (), dtype=tf.int32)
    confusion_name = 'confusion_matrix_'+name
    confusion = _createLocalVariable(confusion_name, [num_classes, num_classes], dtype=tf.int32)

    # Create the update op for doing a "+=" accumulation on the batch
    countUpdate = count.assign(count + tf.reduce_sum(batch_confusion))
    confusionUpdate = confusion.assign(confusion + batch_confusion)

    updateOp = tf.group(confusionUpdate, countUpdate)

    #percentConfusion = 100 * tf.truediv(confusion, count)
    #return percentConfusion, updateOp
    return tf.identity(confusion), updateOp



def draw_confusion_matrix(cm,title,labels,output_path, is_large=False):
    """ Draw confusion matrix

        Args:
            ``cm`` (numpy array):
            ``title`` (string): plot title
            ``labels`` (list): list of labels to be plot on axis ticks
            ``output_path`` (string): where to save output figure
            ``is_large`` (bool): if *True*, labels will not be displayed.


    """
    plt.figure()
    if not is_large:
        ax = sns.heatmap(cm, xticklabels=labels,
                            yticklabels=labels, cmap='coolwarm', annot=True, robust=True, cbar=False,fmt='g').get_figure()                    
    else:
        
        # Normalize values - TODO: optimize it
        cm_norm = np.zeros(cm.shape)
        cm_sum = np.sum(cm,axis=1)
        for i,v in enumerate(cm_sum):    
            cm_norm[i] = cm[i].astype(np.float)/float(v)

        plt.subplots(figsize=(12,10))
        ax = sns.heatmap(cm_norm, xticklabels=[],yticklabels=[], cmap='coolwarm', robust=False, cbar=True,fmt='g').get_figure()
    
    plt.title(title)
    plt.xlabel("Prediciton")
    plt.ylabel("Ground Truth")
    plt.tight_layout()
    output_file = io.BytesIO()
    ax.savefig(output_file)
    output_bytes = output_file.getvalue()

    with tf.gfile.GFile(output_path,'wb') as f:
        f.write(output_bytes)