# -*- coding: utf-8 -*-

import os

import numpy as np
import pandas as pd
import tensorflow as tf
from google.cloud import storage

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png')


def split_storage_name(full_path):
    """Split full_path to return bucket name and file path"""

    full_path_split = "".join(full_path.split("gs://")[1:])  # Remove 'gs://' prefix
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
    tf.logging.info('Full path of {} {}'.format(destination_file_name, os.path.abspath(destination_file_name)))


def read_csv(full_filename):
    """Return metadata as pandas DataFrame whether on storage or locally"""
    if on_storage(full_filename):
        tf.logging.info("Reading metadata from storage")
        local_filename = "tmp.csv"
        bucket_name, filename = split_storage_name(full_filename)
        download_blob(bucket_name, filename, local_filename)
        metadata = pd.read_csv(os.path.abspath(local_filename))
        tf.logging.info("HEAD: {}".format(metadata.head()))
        os.remove(os.path.abspath(local_filename))  # delete file immediately after reading it
    else:
        tf.logging.info("Reading metadata locally")
        metadata = pd.read_csv(full_filename)

    return metadata


def list_tfrecord(regex):
    """Return list of files given a regex"""

    list_op = tf.train.match_filenames_once(regex)
    init_ops = (tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_ops)
        files = sess.run(list_op)

    return files


def get_dataset_len(tfrecord_list):
    """Approximate dataset length by the length of the first x-th tfrecords"""
    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    return len(tfrecord_list) * np.mean(
        list(sum(1 for _ in tf.python_io.tf_record_iterator(path, options)) for path in tfrecord_list[0:10]))
