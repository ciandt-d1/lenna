# -*- coding: utf-8 -*-
from __future__ import division

import math
import tensorflow as tf
import utils

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
tf.app.flags.DEFINE_string(flag_name="network_name",
                           default_value="inception_v4", docstring="Network architecture to use")
tf.app.flags.DEFINE_string(flag_name="endpoint",
                           default_value="Mixed_7d", docstring="Network endpoint name to use")
tf.app.flags.DEFINE_integer(flag_name="batch_size", default_value=1,
                            docstring="Batch size")
tf.app.flags.DEFINE_integer(flag_name="train_steps", default_value=20,
                            docstring="Train steps")
tf.app.flags.DEFINE_integer(flag_name="image_size", default_value=200,
                            docstring="Image size")
tf.app.flags.DEFINE_integer(flag_name="eval_freq", default_value=5,
                            docstring="Frequency to perfom evalutaion")
tf.app.flags.DEFINE_integer(flag_name="eval_throttle_secs", default_value=100,
                            docstring="Evaluation every 'eval_throttle_secs' seconds")
tf.app.flags.DEFINE_float(flag_name="learning_rate",
                          default_value=1e-3, docstring="Learning Rate")
tf.app.flags.DEFINE_float(flag_name="beta1",
                          default_value=0.9, docstring="First order momentum for Adam optimizer")
tf.app.flags.DEFINE_float(flag_name="beta2",
                          default_value=0.999, docstring="Second order momentum for Adam optimizer")
tf.app.flags.DEFINE_float(flag_name="epsilon",
                          default_value=1e-8, docstring="Epsilon to avoid division by zero on Adam optimizer")
tf.app.flags.DEFINE_boolean(flag_name="debug", default_value=False, docstring="Debug mode")


def train(estimator_specs):

    params = tf.contrib.training.HParams(
        learning_rate=FLAGS.learning_rate,
        beta1=FLAGS.beta1,
        beta2=FLAGS.beta2,
        epsilon=FLAGS.epsilon,
        min_eval_frequency=100,
        train_steps=FLAGS.train_steps
    )

    run_config = tf.contrib.learn.RunConfig()
    run_config = run_config.replace(model_dir=FLAGS.model_dir)

    model_fn = estimator_specs.get_model_fn(
        FLAGS.network_name, FLAGS.endpoint, FLAGS.checkpoint_path)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        params=params,
        config=run_config
    )

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

    preproc_fn_train = estimator_specs.get_preproc_fn(
        network_name=FLAGS.network_name, is_training=True)
    preproc_fn_eval = estimator_specs.get_preproc_fn(
        network_name=FLAGS.network_name, is_training=False)

    train_input_fn, train_input_hook = estimator_specs.input_fn(
        batch_size=FLAGS.batch_size, metadata=train_metadata, class_dict=estimator_specs.class_dict, is_tfrecord=is_tfrecord, epochs=epochs, image_size=FLAGS.image_size,preproc_fn=preproc_fn_train)
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
