# -*- coding: utf-8 -*-

#import base_architectures as ba
from nets import nets_factory as slim_nets
import tensorflow as tf


def cnn_architecture(inputs, is_training):
    """
    Return network architecture

    Args:
        inputs: Tensor
            Input Tensor
        is_training: bool
            Whether the network will be used to train or not. Used for dropout operation

    Returns:
        Network Logits
    """

    net_fn = slim_nets.get_network_fn(
        name="mobilenet_v1_025", num_classes=None, is_training=is_training, weight_decay=0.0)

    _, endpoints = net_fn(inputs)
    net_final = endpoints['AvgPool_1a']
    net_final = tf.layers.flatten(net_final)

    with tf.variable_scope("MiniMNIST"):
        net_final = tf.layers.dense(net_final, 100, activation=None, name='fc1', trainable=True)
        net_final = tf.layers.dropout(net_final,rate=0.6,name="dropout_1",training=is_training)
        net_logits = tf.layers.dense(net_final, 10, activation=None, name='logits', trainable=True)

    return net_logits
