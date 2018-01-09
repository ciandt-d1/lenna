# -*- coding: utf-8 -*-
import abc

import tensorflow as tf

import dataset
from hooks import IteratorInitializerHook


# from collections import namedtuple
# LabelDepth = namedtuple("LabelDepth","name depth")

class EstimatorSpec(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        super(EstimatorSpec, self).__init__()
        # List of dict that contains two keys, name(label name) and depth(number of classes)
        self.class_dict = []
        self.load_checkpoint_hook = None

    def get_model_fn(self, network_name, endpoint, checkpoint_path):
        """ Implement here your model function """
        pass

    def metric_ops(self, labels, predictions):
        """ Implement here you metrics ops """
        pass

    def get_preproc_fn(self, is_training, network_name):
        """ Implement here your preproc function """
        pass

    def input_fn(self, batch_size, metadata, class_dict, epochs, image_size, preproc_fn):
        """ 
        Input function to provide data to estimator model

        Args:
            batch_size: int
                Batch size
            metadata: any
                List of tfrecord paths handled by the batch loader

        Returns:
            Input Function to be consumed by the Estimator API and any hook
        """
        iterator_initializer_hook = IteratorInitializerHook()

        def _input_fn():
            """Returns input and target tensors"""

            with tf.name_scope('Data_Loader'):
                iterator = dataset.get_batch_loader_tfrecord(
                    metadata=metadata, batch_size=batch_size, epochs=epochs, image_size=image_size, preproc_fn=preproc_fn,
                    class_dict=class_dict)

                next_example, next_label = iterator.get_next()

                iterator_initializer_hook.iterator_initializer_func = \
                    lambda sess: sess.run(iterator.initializer)

                return next_example, next_label

        return _input_fn, iterator_initializer_hook
