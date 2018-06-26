# -*- coding: utf-8 -*-

import tensorflow as tf
from lenna import estimator_specs, train_estimator, utils


from cnn_architecture_inception_v4 import cnn_architecture

import sys
import os

#####################################
# Evaluation FLAGS
#####################################
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_boolean(
    name="evaluate", default=False, help="Evaluation mode")
tf.app.flags.DEFINE_string(
    name="labels", default=None, help="File which contains Mini_MNIST labels")
tf.app.flags.DEFINE_string(name='output_cm_folder',
                           default=None, help='Folder to save confusion matrices')


class MiniMNIST(estimator_specs.EstimatorSpec):

    def __init__(self):
        self.class_dict = [
            {'name': 'class_id',  'depth': 10, 'one-hot': True}]
    
    def preproc_fn(self, image_bytes):

        image_decoded = tf.image.decode_jpeg(
            image_bytes, channels=3, name='image_decoded')        
        image_resize = tf.image.resize_images(tf.to_float(
            image_decoded), [FLAGS.image_size, FLAGS.image_size])
        image_norm = tf.divide(image_resize, 255.0)

        return image_norm

    def get_serving_fn(self):
        return tf.estimator.export.build_raw_serving_input_receiver_fn({'raw_bytes': tf.placeholder(dtype=tf.string, shape=[None])})

    def get_model_fn(self):
        """ 
        Build model function to return network architecture

        Args:
            network_name: string
                Network name to build upon. See base_architectures.py to check the availables networks                        

        Returns:
            Model Function to be consumed by the Estimator API
        """

        def model_fn(features, labels, mode, params):
            """
            Returns network architecture

            Args:
                features: Input Tensor.
                labels: Target Tensor. Used for training and evaluation.
                mode: TRAIN, EVAL, PREDICT. See tensorflow.contrib.learn for more info.
                params: hyperparameters used by the optimizer.

            Returns:
                (EstimatorSpec): Model to be run by Estimator.
            """

            image_batch = tf.map_fn(
                self.preproc_fn, features['raw_bytes'], dtype=tf.float32)
            tf.logging.info('Input shape : {}'.format(image_batch.get_shape()))

            is_training = mode == tf.estimator.ModeKeys.TRAIN

            with tf.variable_scope("MiniMNIST"):
                net = tf.layers.conv2d(image_batch, filters=10, kernel_size=(
                    5, 5), kernel_regularizer=tf.contrib.layers.l2_regularizer(params.weight_decay))
                net = tf.layers.conv2d(net, filters=10, kernel_size=(
                    5, 5), kernel_regularizer=tf.contrib.layers.l2_regularizer(params.weight_decay))
                net = tf.layers.conv2d(net, filters=10, kernel_size=(
                    5, 5), kernel_regularizer=tf.contrib.layers.l2_regularizer(params.weight_decay))
                net = tf.layers.max_pooling2d(net, 2, strides=2)
                net = tf.layers.flatten(net)
                logits = tf.layers.dense(net, 10, activation=None, name='logits',
                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(params.weight_decay))

            prediction = tf.argmax(logits, axis=1, name="prediction")
            prediction_dict = {"class_id": prediction}

            # Loss, training and eval operations are not needed during inference.
            total_loss = None
            loss = None
            train_op = None
            eval_metric_ops = {}
            export_outputs = None

            if mode != tf.estimator.ModeKeys.PREDICT:

                # IT IS VERY IMPORTANT TO RETRIEVE THE REGULARIZATION LOSSES
                reg_loss = tf.losses.get_regularization_loss()

                # This summary is automatically caught by the Estimator API
                tf.summary.scalar("Regularization_Loss", tensor=reg_loss)

                loss = tf.losses.softmax_cross_entropy(
                    labels['class_id'], logits)
                tf.summary.scalar("XEntropy_Loss", tensor=loss)

                total_loss = loss + reg_loss

                learning_rate = utils.configure_learning_rate(
                    params.num_samples_per_epoch, tf.train.get_global_step())
                optimizer = utils.configure_optimizer(learning_rate)
                vars_to_train = utils.get_variables_to_train()
                tf.logging.info("Variables to train: {}".format(vars_to_train))

                if is_training:
                    # You DO must get this collection in order to perform updates on batch_norm variables
                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                    with tf.control_dependencies(update_ops):
                        train_op = optimizer.minimize(
                            loss=total_loss, global_step=tf.train.get_global_step(), var_list=vars_to_train)

                eval_metric_ops = self.metric_ops(labels, prediction_dict)

            else:
                # read labels file to output predictions as string
                export_outputs = {}

                labels = tf.convert_to_tensor(
                    [l.strip() for l in tf.gfile.GFile(FLAGS.labels).readlines()])

                predicted_label = tf.gather(labels, prediction)
                scores = tf.reduce_max(tf.nn.softmax(logits), axis=1)                
                export_outputs = {'predicted_label': tf.estimator.export.ClassificationOutput(
                    scores=scores, classes=predicted_label)}

            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=prediction_dict,
                loss=total_loss,
                train_op=train_op,
                eval_metric_ops=eval_metric_ops,
                export_outputs=export_outputs)

        return model_fn

    def metric_ops(self, labels, predictions):
        """Return a dict of the evaluation Ops.
        Args:
            labels (Tensor): Labels tensor for training and evaluation.
            predictions (Tensor): Predictions Tensor.
        Returns:
            Dict of metric results keyed by name.
        """
        ground_truth = tf.argmax(labels['class_id'], axis=1)
        prediction = predictions['class_id']
        conf_matrix = utils.streaming_confusion_matrix(
            name='conf_matrix', label=ground_truth, prediction=prediction, num_classes=10)

        return {
            'Accuracy': tf.metrics.accuracy(
                labels=ground_truth,
                predictions=prediction,
                name='accuracy'),
            'Precision': tf.metrics.precision(
                labels=ground_truth,
                predictions=prediction,
                name='precision'),
            'Recall': tf.metrics.recall(
                labels=ground_truth,
                predictions=prediction,
                name='recall'),
            'Confusion_Matrix': conf_matrix,
        }


if __name__ == '__main__':
    mini_mnist = MiniMNIST()

    if FLAGS.evaluate:
        labels = tf.gfile.GFile(FLAGS.labels).readlines()

        metrics = train_estimator.evaluate(mini_mnist)
        tf.logging.info("Metrics: {}".format(metrics))

        output_path = os.path.join(FLAGS.output_cm_folder, 'mini_mnist.png')
        utils.draw_confusion_matrix(
            cm=metrics['Confusion_Matrix'], title='Mini_MNIST', labels=labels, output_path=output_path)

    else:
        train_estimator.train(mini_mnist)
