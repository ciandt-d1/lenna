# -*- coding: utf-8 -*-

import tensorflow as tf
from tf_image_classification import estimator_specs, train_estimator, utils
from tf_image_classification.hooks import LoadCheckpointHook
#from cnn_architecture_mobile_net import cnn_architecture
from cnn_architecture_inception_v4 import cnn_architecture
from tensorflow.contrib.learn import ModeKeys
import sys


class MiniMNIST(estimator_specs.EstimatorSpec):

    def __init__(self):
        self.class_dict = [{'name': 'class_id',  'depth': 10}]

    def get_preproc_fn(self, is_training):

        def _preproc(image, width, height):
            image_resize = tf.image.resize_images(
                tf.to_float(image), [height, width])
            image_norm = tf.divide(image_resize, 255.0)
            image_norm = tf.subtract(image_norm, 0.5)
            image_norm = tf.multiply(image_norm, 2)

            return image_norm
        return _preproc

    def get_model_fn(self, checkpoint_path):
        """ 
        Build model function to return network architecture

        Args:
            network_name: string
                Network name to build upon. See base_architectures.py to check the availables networks            
            checkpoint_path: string
                Checkpoint to load.

        Returns:
            Model Function to be consumed by the Estimator API
        """
        self.load_checkpoint_hook = LoadCheckpointHook()

        def model_fn(features, labels, mode, params):
            """
            Returns network architecture

            Args:
                features: Input Tensor.
                labels: Target Tensor. Used for training and evaluation.
                mode: TRAIN, EVAL, INFER. See tensorflow.contrib.learn for more info.
                params: hyperparameters used by the optimizer.

            Returns:
                (EstimatorSpec): Model to be run by Estimator.
            """

            is_training = mode == ModeKeys.TRAIN
            
            # Define model's architecture
            logits = cnn_architecture(
                features, is_training=is_training,weight_decay=params.weight_decay)

            if is_training:                
                if tf.gfile.IsDirectory(checkpoint_path):
                    _checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
                else:
                    _checkpoint_path = checkpoint_path
                    if not tf.gfile.Exists(_checkpoint_path):
                        _checkpoint_path = None
                
                if _checkpoint_path is not None:                    
                    vars_to_restore = utils.get_variables_to_restore()
                    tf.logging.info("Variables to restore from {} : {}".format(_checkpoint_path,vars_to_restore))

                    self.load_checkpoint_hook.load_checkpoint_initializer_func = tf.contrib.framework.assign_from_checkpoint_fn(
                        model_path=_checkpoint_path, var_list=vars_to_restore, ignore_missing_vars=True)
                else:
                    tf.logging.warning(
                        "Checkpoint {} not found, so not loaded".format(_checkpoint_path))


            prediction = tf.argmax(logits, axis=1, name="prediction")
            prediction_dict = {"class_id": prediction}

            
            # Loss, training and eval operations are not needed during inference.
            loss = None
            train_op = None
            eval_metric_ops = {}

            if mode != ModeKeys.INFER:

                # IT IS VERY IMPORTANT TO RETRIEVE THE REGULARIZATION LOSSES
                reg_loss = tf.losses.get_regularization_loss()

                # This summary is automatically caught by the Estimator API
                tf.summary.scalar("Regularization_Loss",tensor=reg_loss)
                
                loss = tf.losses.softmax_cross_entropy(labels['class_id'], logits)
                tf.summary.scalar("XEntropy_Loss",tensor=loss)


                total_loss = loss + reg_loss
                # learning_rate_decay_op = tf.train.exponential_decay(params.learning_rate, tf.train.get_global_step(),
                #                                                     120, 0.75, staircase=True)

                # train_op = tf.train.AdamOptimizer(learning_rate=learning_rate_decay_op,
                #                                   beta1=params.beta1, beta2=params.beta2,
                #                                   epsilon=params.epsilon).minimize(loss=total_loss, global_step=tf.train.get_global_step())

                learning_rate = utils.configure_learning_rate(params.num_samples_per_epoch,tf.train.get_global_step())
                optimizer = utils.configure_optimizer(learning_rate)
                vars_to_train = utils.get_variables_to_train()
                train_op = optimizer.minimize(loss=total_loss, global_step=tf.train.get_global_step(),var_list=vars_to_train)
                #tf.summary.scalar("learning_rate",tensor=params.learning_rate)

                eval_metric_ops = self.metric_ops(labels, prediction_dict)
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=prediction_dict,
                loss=total_loss,
                train_op=train_op,
                eval_metric_ops=eval_metric_ops)

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
        

        return {
            'Accuracy': tf.metrics.accuracy(
                labels=ground_truth,
                predictions=prediction,
                name='accuracy')            
        }


if __name__ == '__main__':
    mini_mnist = MiniMNIST()
    train_estimator.train(mini_mnist)
