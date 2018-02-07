# -*- coding: utf-8 -*-

#import base_architectures as ba
from nets import nets_factory as slim_nets
import tensorflow as tf
#from tensorflow.contrib import slim


def cnn_architecture(inputs, is_training, weight_decay):
    """
    Return network architecture

    Args:
        inputs: Tensor
            Input Tensor
        is_training: bool
            Whether the network will be used to train or not. Used for dropout operation

    Returns:
        Logits for each demographic network branch
    """
    
    net_fn = slim_nets.get_network_fn(
        name="inception_v4", num_classes=None, is_training=is_training, weight_decay=weight_decay)

    _, endpoints = net_fn(inputs)
    net_final = endpoints['Mixed_7d']
    net_final = tf.layers.flatten(net_final)

    with tf.variable_scope("MiniMNIST"):
        net_final = tf.layers.batch_normalization(net_final,epsilon=1e-3,momentum=0.99,name='MiniMNIST_Batchnorm_1',training=is_training)
        net_final = tf.layers.dense(net_final, 100, activation=tf.nn.relu, name='fc1', trainable=True,kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
        net_final = tf.layers.dropout(net_final,rate=0.6,name="dropout_1",training=is_training)
        net_final = tf.layers.batch_normalization(net_final,epsilon=1e-3,momentum=0.99,name='MiniMNIST_Batchnorm_2',training=is_training)
        net_logits = tf.layers.dense(net_final, 10, activation=None, name='logits', trainable=True,kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))

    return net_logits
