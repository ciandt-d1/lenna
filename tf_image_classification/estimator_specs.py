# -*- coding: utf-8 -*-
"""
.. module:: estimator_specs
   :platform: Unix
   :synopsis: Definition of Estimator Specification class for Image Classification

.. moduleauthor:: Rodrigo Pereira <rodrigofp@ciandt.com>


"""


import abc

import tensorflow as tf
import dataset


class EstimatorSpec(object):
    """Estimator Specification base class for image classification

    Inherit this class to create your own estimator for image classification.
    Basically you need to implement the methods :func:`~tf_image_classification.estimator_specs.EstimatorSpec.get_model_fn`, :func:`~tf_image_classification.estimator_specs.EstimatorSpec.metric_ops` and :func:`~tf_image_classification.estimator_specs.EstimatorSpec.get_preproc_fn`

    .. note::

       If you follow the schema presented on the `MiniMNIST` example, you don't need to overwrite the method :func:`input_fn`.

    Recall that the attribute ``class_dict`` is set as an list.
    It defines how the labels from your dataset should be decoded.
    You may have multiple labels when dealing with multi-task problems.
    `Take a read on it <https://www.kdnuggets.com/2016/07/multi-task-learning-tensorflow-part-1.html>`_

    **Example from MiniMNIST:**
    
    >>> self.class_dict = [{'name': 'class_id',  'depth': 10, 'one-hot': True}]

    In this example we have just one label, called ``class_id`` . It **must** match with what you have saved on tf-records.


    """
    __metaclass__ = abc.ABCMeta


    def __init__(self):        
        
        super(EstimatorSpec, self).__init__()
        
        self.class_dict = []

    def get_model_fn(self, *args):
        """Implement your model function here.
            It **must** return a function that builds your model as a `tf.estimator.EstimatorSpec <https://www.tensorflow.org/api_docs/python/tf/estimator/EstimatorSpec>`_
            
            The model function **must** follow the signature (`features`, `labels`, `mode`, `params`) as shown on `Estimator Tutorial <https://www.tensorflow.org/get_started/custom_estimators#write_a_model_function>`_
            
            It is here that your network architecture is defined, both for training, evaluation and inference.
            
            * When on training mode ( mode == `tf.estimator.ModeKeys.TRAIN` ) it is expected the graph to be built up to the loss definition.
            
            * When on evaluation mode ( mode == `tf.estimator.ModeKeys.EVAL` ) it is expected the graph to be built up to the metrics definition.
            
            * When on inference mode ( mode == `tf.estimator.ModeKeys.PREDICT` ) it is expected the graph to be built up to the logits/predictions definition.

        Args:
           ``args`` : Any argument you want to pass to your model function

        Returns:
            Function that builds the model_fn which returns a `tf.estimator.EstimatorSpec <https://www.tensorflow.org/api_docs/python/tf/estimator/EstimatorSpec>`_


        **Example from MiniMNIST:**
        
        >>> def get_model_fn(self):
        ...     
        ...     # Recall to follow the signature below
        ...     def model_fn(features, labels, mode, params):
        ... 
        ...         is_training = mode == ModeKeys.TRAIN
        ... 
        ...         # Define model's architecture
        ...         logits = cnn_architecture(
        ...             features, is_training=is_training, weight_decay=params.weight_decay)
        ...         
        ...         prediction = tf.argmax(logits, axis=1, name="prediction")
        ...         prediction_dict = {"class_id": prediction}
        ... 
        ...         # Loss, training and eval operations are not needed during inference.
        ...         loss = None
        ...         train_op = None
        ...         eval_metric_ops = {}
        ... 
        ...         if mode != ModeKeys.INFER:
        ... 
        ...             # IT IS VERY IMPORTANT TO RETRIEVE THE REGULARIZATION LOSSES
        ...             reg_loss = tf.losses.get_regularization_loss()
        ... 
        ...             # This summary is automatically caught by the Estimator API
        ...             tf.summary.scalar("Regularization_Loss", tensor=reg_loss)
        ... 
        ...             loss = tf.losses.softmax_cross_entropy(
        ...                 labels['class_id'], logits)
        ...             tf.summary.scalar("XEntropy_Loss", tensor=loss)
        ... 
        ...             total_loss = loss + reg_loss
        ... 
        ...             learning_rate = utils.configure_learning_rate(
        ...                 params.num_samples_per_epoch, tf.train.get_global_step())
        ...             optimizer = utils.configure_optimizer(learning_rate)
        ...             vars_to_train = utils.get_variables_to_train()
        ...             tf.logging.info("Variables to train: {}".format(vars_to_train))
        ...             
        ...             if is_training:
        ...                 # You DO must get this collection in order to perform updates on batch_norm variables
        ...                 update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        ...                 with tf.control_dependencies(update_ops):
        ...                     train_op = optimizer.minimize(
        ...                         loss=total_loss, global_step=tf.train.get_global_step(), var_list=vars_to_train)
        ... 
        ...             eval_metric_ops = self.metric_ops(labels, prediction_dict)
        ...         return tf.estimator.EstimatorSpec(
        ...             mode=mode,
        ...             predictions=prediction_dict,
        ...             loss=total_loss,
        ...             train_op=train_op,
        ...             eval_metric_ops=eval_metric_ops)
        ... 
        ...     return model_fn            

        """
        pass

    def metric_ops(self, labels, predictions):
        """ Implement here the metrics you want to collect
        
        You **do** need to use the streaming metrics from `tf.metrics <https://www.tensorflow.org/api_docs/python/tf/metrics>`_ ,
        however it is also possible to create your own since it returns the tuple ( `metric_val` , `metric_update_op` ).
        See :func:`util.streaming_confusion_matrix` for an example
         
        
        Args:
            ``labels`` (`tf.Tensor <https://www.tensorflow.org/api_docs/python/tf/Tensor>`_): tensor with the ground-truth labels for each example
            
            ``prediction`` (`tf.Tensor <https://www.tensorflow.org/api_docs/python/tf/Tensor>`_): tensor with the prediction labels for each example

        Returns:
            metrics (dict): Dictionary mapping the metric name (it will be both logged and displayed on TensorBoard) and the metric itself


        **Example from MiniMNIST:**

        >>> def metric_ops(self, labels, predictions):    
        ...     ground_truth = tf.argmax(labels['class_id'], axis=1)
        ...     prediction = predictions['class_id']
        ...     conf_matrix = utils.streaming_confusion_matrix(
        ...         name='conf_matrix', label=ground_truth, prediction=prediction, num_classes=10)
        ... 
        ...     return {
        ...         'Accuracy': tf.metrics.accuracy(
        ...             labels=ground_truth,
        ...             predictions=prediction,
        ...             name='accuracy'),
        ...         'Precision': tf.metrics.precision(
        ...             labels=ground_truth,
        ...             predictions=prediction,
        ...             name='precision'),
        ...         'Recall': tf.metrics.recall(
        ...             labels=ground_truth,
        ...             predictions=prediction,
        ...             name='recall'),
        ...         'Confusion_Matrix': conf_matrix,
        ...     }

        """
        pass

    def get_preproc_fn(self, is_training, **kargs):
        """ Implement here your preprocessing function.

        Create your preprocessing function. It is expected that the input tensor contains a RGB or BGR image.
        
        Args:
            ``is_training`` (bool): Whether or not the preprocessing will be executed during training or not. This may be useful if you want to do online data augmentation
            
            ``kargs`` (dict): any extra argument you want to pass to your preprocessing function
        
        Returns:
            A function that preprocess input images

        **Example from MiniMNIST:**

        >>> def get_preproc_fn(self, is_training):
        ... 
        ...     def _preproc(image, width, height):
        ...         image_resize = tf.image.resize_images(
        ...             tf.to_float(image), [height, width])
        ...         image_norm = tf.divide(image_resize, 255.0)
        ... 
        ...         return image_norm
        ...     return _preproc
        """
        pass

    def input_fn(self, batch_size, metadata, class_dict, is_tfrecord, epochs, image_size, preproc_fn):
        """Input function to provide data to estimator model

        Args:
            ``batch_size`` (int): Batch size

            ``metadata`` (list): List of tfrecord paths handled by the batch loader

            ``class_dict`` (list of dict): It defines how the labels from your dataset should be decoded.

            ``is_tfrecord`` (bool): Whether the dataset list ``metadata`` is a **csv** or **tf-records**

            ``epochs`` (int): Number of epochs to provide data

            ``image_size`` (int): Image dimension to resize on preprocessing function

            ``preproc_fn`` (function): Image preprocessing function

        Returns:
            Input function that returns a tf.data.Dataset object to be consumed by the Estimator API


        .. note::

            This method is already implemented, but you may overwrite as you wish.

        
        **Code**

        >>> def _input_fn():
        ...     with tf.name_scope('Data_Loader'):
        ...         if is_tfrecord:
        ...             batch = dataset.get_batch_loader_tfrecord(
        ...                 metadata=metadata, batch_size=batch_size, epochs=epochs, image_size=image_size, preproc_fn=preproc_fn,
        ...                 class_dict=class_dict)
        ...         else:
        ...             batch = dataset.get_batch_loader_csv(
        ...                 metadata=metadata, batch_size=batch_size, epochs=epochs, image_size=image_size, preproc_fn=preproc_fn,
        ...                 class_dict=class_dict)
        ... 
        ...         return batch
        ... 
        ... return _input_fn
        """

        def _input_fn():
            """Returns input and target tensors"""

            with tf.name_scope('Data_Loader'):
                if is_tfrecord:
                    batch = dataset.get_batch_loader_tfrecord(
                        metadata=metadata, batch_size=batch_size, epochs=epochs, image_size=image_size, preproc_fn=preproc_fn,
                        class_dict=class_dict)
                else:
                    batch = dataset.get_batch_loader_csv(
                        metadata=metadata, batch_size=batch_size, epochs=epochs, image_size=image_size, preproc_fn=preproc_fn,
                        class_dict=class_dict)

                return batch

        return _input_fn
