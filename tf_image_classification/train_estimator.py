# -*- coding: utf-8 -*-
from __future__ import division

import math
import tensorflow as tf
import utils

slim = tf.contrib.slim

tf.logging.set_verbosity(tf.logging.INFO)

# Set default flags for the output directories
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    flag_name='model_dir', default_value='./models',
    docstring='Output directory for model and training stats.')
tf.app.flags.DEFINE_string(flag_name="checkpoint_path", default_value=None,
                           docstring="Checkpoint path to load pre-trained weights")
tf.app.flags.DEFINE_string(
    flag_name="train_metadata", default_value="",
    docstring="Trainset metadata")
tf.app.flags.DEFINE_string(
    flag_name="eval_metadata", default_value="",
    docstring="Evalset metadata")
tf.app.flags.DEFINE_integer(flag_name="batch_size", default_value=1,
                            docstring="Batch size")
tf.app.flags.DEFINE_integer(flag_name="train_steps", default_value=20,
                            docstring="Train steps")
tf.app.flags.DEFINE_integer(flag_name="image_size", default_value=299,
                            docstring="Image size")
tf.app.flags.DEFINE_integer(flag_name="eval_freq", default_value=5,
                            docstring="Frequency to perfom evalutaion")
tf.app.flags.DEFINE_integer(flag_name="eval_throttle_secs", default_value=120,
                            docstring="Evaluation every 'eval_throttle_secs' seconds")
tf.app.flags.DEFINE_boolean(flag_name="debug", default_value=False, docstring="Debug mode")

# tf.app.flags.DEFINE_float(flag_name="learning_rate",
#                           default_value=1e-3, docstring="Learning Rate")
# tf.app.flags.DEFINE_float(flag_name="beta1",
#                           default_value=0.9, docstring="First order momentum for Adam optimizer")
# tf.app.flags.DEFINE_float(flag_name="beta2",
#                           default_value=0.999, docstring="Second order momentum for Adam optimizer")
# tf.app.flags.DEFINE_float(flag_name="epsilon",
#                           default_value=1e-8, docstring="Epsilon to avoid division by zero on Adam optimizer")



######################
# Optimization Flags #
######################

tf.app.flags.DEFINE_float(
    'weight_decay', 0.00004, 'The weight decay on the model weights.')

tf.app.flags.DEFINE_string(
    'optimizer', 'rmsprop',
    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')

tf.app.flags.DEFINE_float(
    'adadelta_rho', 0.95,
    'The decay rate for adadelta.')

tf.app.flags.DEFINE_float(
    'adagrad_initial_accumulator_value', 0.1,
    'Starting value for the AdaGrad accumulators.')

tf.app.flags.DEFINE_float(
    'adam_beta1', 0.9,
    'The exponential decay rate for the 1st moment estimates.')

tf.app.flags.DEFINE_float(
    'adam_beta2', 0.999,
    'The exponential decay rate for the 2nd moment estimates.')

tf.app.flags.DEFINE_float(
    'opt_epsilon', 1.0, 'Epsilon term for the optimizer.')

tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
                          'The learning rate power.')

tf.app.flags.DEFINE_float(
    'ftrl_initial_accumulator_value', 0.1,
    'Starting value for the FTRL accumulators.')

tf.app.flags.DEFINE_float(
    'ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')

tf.app.flags.DEFINE_float(
    'ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')

tf.app.flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')

tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')

tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

#######################
# Learning Rate Flags #
#######################

tf.app.flags.DEFINE_string(
    'learning_rate_decay_type',
    'exponential',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"')

tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')

tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.0001,
    'The minimal end learning rate used by a polynomial decay learning rate.')

tf.app.flags.DEFINE_float(
    'label_smoothing', 0.0, 'The amount of label smoothing.')

tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')

tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 2.0,
    'Number of epochs after which learning rate decays.')

tf.app.flags.DEFINE_bool(
    'sync_replicas', False,
    'Whether or not to synchronize the replicas during training.')

tf.app.flags.DEFINE_integer(
    'replicas_to_aggregate', 1,
    'The Number of gradients to collect before updating params.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')




def train(estimator_specs):

     # Check whether data is stored as TF-Records or csv
    is_tfrecord = False

    if FLAGS.eval_metadata.endswith("csv") and FLAGS.train_metadata.endswith("csv"):
        train_metadata = utils.read_csv(FLAGS.train_metadata)
        eval_metadata = utils.read_csv(FLAGS.eval_metadata)
        dataset_len = len(train_metadata)
    else:
        is_tfrecord = True
        train_metadata = utils.list_tfrecord(FLAGS.train_metadata)
        eval_metadata = utils.list_tfrecord(FLAGS.eval_metadata)
        dataset_len = utils.get_dataset_len(train_metadata)

    epochs = int(math.ceil(FLAGS.train_steps /
                           (dataset_len / FLAGS.batch_size)))

    tf.logging.info("Dataset length: {} examples".format(dataset_len))
    tf.logging.info("Epochs to run: {}".format(epochs))

    # learning_rate = _configure_learning_rate(
    #     dataset_len, slim.create_global_step())
    # optimizer = _configure_optimizer(learning_rate)

    params = tf.contrib.training.HParams(
        # learning_rate=FLAGS.learning_rate,
        # beta1=FLAGS.beta1,
        # beta2=FLAGS.beta2,
        # epsilon=FLAGS.epsilon,
        # min_eval_frequency=100,
        # train_steps=FLAGS.train_steps
        # optimizer=optimizer,
        # learning_rate=learning_rate,
        num_samples_per_epoch=dataset_len,
        weight_decay=FLAGS.weight_decay
    )

    run_config = tf.contrib.learn.RunConfig()
    run_config = run_config.replace(model_dir=FLAGS.model_dir)

    model_fn = estimator_specs.get_model_fn(FLAGS.checkpoint_path)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        params=params,
        config=run_config
    )

    preproc_fn_train = estimator_specs.get_preproc_fn(is_training=True)
    preproc_fn_eval = estimator_specs.get_preproc_fn(is_training=False)

    train_input_fn, train_input_hook = estimator_specs.input_fn(
        batch_size=FLAGS.batch_size, metadata=train_metadata, class_dict=estimator_specs.class_dict, is_tfrecord=is_tfrecord, epochs=epochs, image_size=FLAGS.image_size, preproc_fn=preproc_fn_train)
    eval_input_fn, eval_input_hook = estimator_specs.input_fn(
        batch_size=FLAGS.batch_size, metadata=eval_metadata, class_dict=estimator_specs.class_dict, is_tfrecord=is_tfrecord, epochs=1, image_size=FLAGS.image_size, preproc_fn=preproc_fn_eval)

    train_hooks = [train_input_hook]
    if estimator_specs.load_checkpoint_hook is not None:
        train_hooks.append(estimator_specs.load_checkpoint_hook)

    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn, max_steps=FLAGS.train_steps, hooks=train_hooks)
    eval_spec = tf.estimator.EvalSpec(
        input_fn=eval_input_fn, steps=FLAGS.eval_freq, throttle_secs=FLAGS.eval_throttle_secs, hooks=[eval_input_hook])

    tf.estimator.train_and_evaluate(
        estimator=estimator, train_spec=train_spec, eval_spec=eval_spec)
