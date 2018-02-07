# -*- coding: utf-8 -*-

import tensorflow as tf

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
    
    
    with tf.variable_scope("MiniMNIST"):
        net_final = tf.layers.conv2d(inputs=inputs,filters=32,kernel_size=(3,3), activation=tf.nn.relu,trainable=True,kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),name='Conv_1')                
        net_final = tf.layers.conv2d(inputs=net_final,filters=64,kernel_size=(3,3), activation=tf.nn.relu,trainable=True,kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),name='Conv_2')
        net_final = tf.layers.batch_normalization(net_final,epsilon=1e-3,momentum=0.99,name='MiniMNIST_Batchnorm_1',training=is_training)
        
        net_final = tf.layers.max_pooling2d(inputs=net_final,pool_size=(2,2),strides=(1,1),name='MaxPool_1')        
        net_final = tf.layers.dropout(net_final,rate=0.25,name="Dropout_1",training=is_training)
        net_final = tf.layers.flatten(net_final)

        net_final = tf.layers.dense(net_final, 128, activation=tf.nn.relu, name='fc1', trainable=True,kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
        net_final = tf.layers.dropout(net_final,rate=0.5,name="Dropout_2",training=is_training)
        net_logits = tf.layers.dense(net_final, 10, activation=None, name='logits', trainable=True,kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))

    return net_logits
