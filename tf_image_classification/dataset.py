# -*- coding: utf-8 -*-

import tensorflow as tf


def load_and_preproc_from_file(filename, label, width, height, preproc_fn, class_dict):
    """
    Load images from bytes to tensors

    Args:
        image_bytes: Image tensor as byte string
        width: output width
        height: output height
        preproc_fn: preprocessing function
        class_dict: list of dictionary that contains two keys: 'name'(tensor) and 'depth'(int for one-hot encoding)
        :param class_dict:
        :param preproc_fn:
        :param height:
        :param width:
        :param label:
        :param filename:
    """

    image_bytes = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_bytes, channels=3)

    if preproc_fn is not None:
        image_decoded = preproc_fn(image_decoded, width, height)

    labels = {}
    for _class in class_dict:
        one_hot_label = tf.one_hot(tf.to_int32(
            label), depth=_class['depth'], dtype=tf.float32)
        labels[_class['name']] = one_hot_label

    return image_decoded, labels


def load_and_preproc(image_bytes, width, height, preproc_fn, class_dict):
    """
    Load images from bytes to tensors

    Args:
        image_bytes: Image tensor as byte string
        width: output width
        height: output height
        preproc_fn: preprocessing function
        class_dict: list of dictionary that contains two keys: 'name'(tensor) and 'depth'(int for one-hot encoding)
    """

    image_decoded = tf.image.decode_jpeg(image_bytes, channels=3)

    if preproc_fn is not None:
        image_decoded = preproc_fn(image_decoded, height, width)

    labels = {}
    for _class in class_dict:
        if _class['one-hot']:
            one_hot_label = tf.one_hot(tf.to_int32(
                _class['tensor']), depth=_class['depth'], dtype=tf.float32)
            labels[_class['name']] = one_hot_label
        else:
            labels[_class['name']] = _class['tensor']

    return image_decoded, labels


def get_batch_loader_tfrecord(metadata, batch_size, epochs, preproc_fn, class_dict, image_size, batch_prefech=5):
    """
    Return op to provide batches through training

    Args:
    metadata: pandas Dataframe or tfrecord
    batch_size: int
    is_tfrecord: bool
    """

    def _parse_function(example_proto):
        """ Parse data from tf.Example. All labels are decoded as float32"""

        parser_dict = {"image_bytes": tf.FixedLenFeature(
            (), tf.string, default_value="")}
        for d in class_dict:
            parser_dict[d['name']] = tf.FixedLenFeature(
                (), tf.float32, default_value=-1)
        parsed_features = tf.parse_single_example(example_proto, parser_dict)

        return parsed_features

    def _build_dict_and_preproc(parsed_features):
        """ Decode data to batch loader """

        class_dict_tensor = []
        for d in class_dict:
            class_dict_tensor.append(
                {'name': d['name'], 'depth': d['depth'], 'one-hot': d['one-hot'], 'tensor': parsed_features[d['name']]})

        return load_and_preproc(parsed_features['image_bytes'], image_size, image_size, preproc_fn, class_dict_tensor)

    dataset = tf.data.TFRecordDataset(metadata, compression_type="GZIP")
    dataset = dataset.map(_parse_function)
    dataset = dataset.map(_build_dict_and_preproc)

    if not tf.flags.FLAGS.debug:
        dataset = dataset.shuffle(buffer_size=batch_prefech * batch_size)

    dataset = dataset.repeat(epochs) #TODO: optimize these steps
    
    return dataset.batch(batch_size)


def get_batch_loader_csv(metadata, batch_size, epochs, preproc_fn, class_dict, image_size, batch_prefech=5):
    """
    Return op to provide batches through training

    Args:
    metadata: pandas Dataframe or tfrecord
    batch_size: int
    is_tfrecord: bool
    """

    dataset = tf.data.Dataset.from_tensor_slices(
        (metadata['URIs'].tolist(), metadata['labels'].factorize()[0]))
    dataset = dataset.map(lambda filename, label: load_and_preproc_from_file(
        filename, label, image_size, image_size, preproc_fn, class_dict))

    if not tf.flags.FLAGS.debug:
        dataset = dataset.shuffle(buffer_size=batch_prefech * batch_size)

    dataset = dataset.repeat(epochs)
    dataset = dataset.batch(batch_size)
    batch_iter = dataset.make_initializable_iterator()

    return batch_iter
