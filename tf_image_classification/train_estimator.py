# -*- coding: utf-8 -*-
from __future__ import division

import math
import tensorflow as tf
import utils
import json
import os

slim = tf.contrib.slim

tf.logging.set_verbosity(tf.logging.INFO)

# Set default flags for the output directories
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    name='model_dir', default='./models',
    help='Output directory for model and training stats.')
tf.app.flags.DEFINE_string(name="warm_start_ckpt", default=None,
                           help="Checkpoint path to load pre-trained weights")
tf.app.flags.DEFINE_string(
    name="train_metadata", default="",
    help="Trainset metadata")
tf.app.flags.DEFINE_string(
    name="eval_metadata", default="",
    help="Evalset metadata")
tf.app.flags.DEFINE_integer(name="batch_size", default=1,
                            help="Batch size")
tf.app.flags.DEFINE_integer(name="train_steps", default=20,
                            help="Train steps")
tf.app.flags.DEFINE_integer(name="image_size", default=299,
                            help="Image size")
tf.app.flags.DEFINE_integer(name="eval_freq", default=5,
                            help="Frequency to perfom evalutaion")
tf.app.flags.DEFINE_integer(name="eval_throttle_secs", default=120,
                            help="Evaluation every 'eval_throttle_secs' seconds")
tf.app.flags.DEFINE_boolean(name="debug", default=False, help="Debug mode")
tf.app.flags.DEFINE_boolean(
    name="shuffle", default=True, help="Whether or not shuffle the dataset")


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

#####################
# Fine-Tuning Flags #
#####################

tf.app.flags.DEFINE_string(
    'trainable_scopes', None,
    'Comma-separated list of scopes to filter the set of variables to train.'
    'By default, None would train all the variables.')

tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', None,
    'Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint.')

tf.app.flags.DEFINE_string(
    'checkpoint_restore_scopes', None,
    'Comma-separated list of scopes of variables to restore '
    'from a checkpoint.')

#####################
# Checkpoint Flags #
#####################
tf.app.flags.DEFINE_integer(name="save_summary_steps", default=100,
                            help="Save summaries every this many steps")
tf.app.flags.DEFINE_integer(name="save_checkpoints_steps", default=None,
                            help="Save checkpoints every this many steps. Can not be specified with `save_checkpoints_secs`")
tf.app.flags.DEFINE_integer(name="save_checkpoints_secs", default=None,
                            help="Save checkpoints every this many seconds. Can not be specified with save_checkpoints_steps")
tf.app.flags.DEFINE_integer(name="keep_checkpoint_max", default=5,
                            help="The maximum number of recent checkpoint files to keep. -1 to keep every checkpoints")

def train(estimator_specs):
    """Train your model defined by ``estimator_specs``.

    Here is where all the main flow is defined to train your model.
    The following steps take place:

    * Read the dataset (from tf-records or csv) into `tf.data.Dataset <https://www.tensorflow.org/api_docs/python/tf/data/Dataset>`_
    * Check whether the job will be for hyperparameter tuning (ML Engine) or not
    * Create estimator `tf.estimator.Estimator <https://www.tensorflow.org/versions/master/api_docs/python/tf/estimator>`_ from :class:`~tf_image_classification.estimator_specs.EstimatorSpec`
    * Create input functions for both training and evaluation steps
    * Train and Evaluate `tf.estimator.train_and_evaluate <https://www.tensorflow.org/versions/master/api_docs/python/tf/estimator/train_and_evaluate>`_ .

    Args:
        ``estimator_specs`` (:class:`~tf_image_classification.estimator_specs.EstimatorSpec`) : estimator to be trained

    Returns:
        Nothing is returned since the ``ckpt`` and ``summaries`` files are already saved during the training.
    """

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

    params = tf.contrib.training.HParams(
        num_samples_per_epoch=dataset_len,
        weight_decay=FLAGS.weight_decay
    )

    env = json.loads(os.environ.get('TF_CONFIG', '{}'))
    task_data = env.get('task') or {'type': 'master', 'index': 0}
    trial = task_data.get('trial')

    if trial is not None:
        output_dir = os.path.join(FLAGS.model_dir, trial)
        tf.logging.info(
            "Hyperparameter Tuning - Trial {}. model_dir = {}".format(trial, output_dir))
    else:
        output_dir = FLAGS.model_dir

    model_fn = estimator_specs.get_model_fn()

    
    run_config = tf.estimator.RunConfig(
        model_dir=output_dir,        
        save_summary_steps= FLAGS.save_summary_steps,        
        keep_checkpoint_max= None if (FLAGS.keep_checkpoint_max < 0) else FLAGS.keep_checkpoint_max
    )

    if FLAGS.save_checkpoints_steps is None and FLAGS.save_checkpoints_secs is None: 
        save_checkpoints_secs = 600
        save_checkpoints_steps = FLAGS.save_checkpoints_secs
    elif FLAGS.save_checkpoints_steps is not None and FLAGS.save_checkpoints_secs is not None: 
        raise('`save_checkpoints_steps` and `save_checkpoints_secs` cannot be both set.')
    else:
        save_checkpoints_secs = FLAGS.save_checkpoints_secs
        save_checkpoints_steps = FLAGS.save_checkpoints_steps


    run_config = run_config.replace(save_checkpoints_steps=save_checkpoints_steps,save_checkpoints_secs=save_checkpoints_secs)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        params=params,
        config=run_config,
        warm_start_from=tf.train.warm_start(
            ckpt_to_initialize_from=FLAGS.warm_start_ckpt, vars_to_warm_start=FLAGS.checkpoint_restore_scopes)
    )

    preproc_fn_train = estimator_specs.get_preproc_fn(is_training=True)
    preproc_fn_eval = estimator_specs.get_preproc_fn(is_training=False)

    train_input_fn = estimator_specs.input_fn(
        batch_size=FLAGS.batch_size, metadata=train_metadata, class_dict=estimator_specs.class_dict, is_tfrecord=is_tfrecord, epochs=epochs, image_size=FLAGS.image_size, preproc_fn=preproc_fn_train)
    eval_input_fn = estimator_specs.input_fn(
        batch_size=FLAGS.batch_size, metadata=eval_metadata, class_dict=estimator_specs.class_dict, is_tfrecord=is_tfrecord, epochs=1, image_size=FLAGS.image_size, preproc_fn=preproc_fn_eval)

    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn, max_steps=FLAGS.train_steps)
    eval_spec = tf.estimator.EvalSpec(
        input_fn=eval_input_fn, steps=FLAGS.eval_freq, throttle_secs=FLAGS.eval_throttle_secs)

    tf.estimator.train_and_evaluate(
        estimator=estimator, train_spec=train_spec, eval_spec=eval_spec)


def evaluate(estimator_specs):
    """Evaluate your model defined by ``estimator_specs``

    Args:
        ``estimator_specs`` (:class:`~tf_image_classification.estimator_specs.EstimatorSpec`) : estimator to be evaluated

    Returns:
        ``metrics`` (dict): Dictionary of metrics defined on :func:`~tf_image_classification.estimator_specs.EstimatorSpec.metric_ops`
    """

    is_tfrecord = False

    if FLAGS.eval_metadata.endswith("csv"):
        eval_metadata = utils.read_csv(FLAGS.eval_metadata)
        dataset_len = len(eval_metadata)
    else:
        is_tfrecord = True
        eval_metadata = utils.list_tfrecord(FLAGS.eval_metadata)
        dataset_len = utils.get_dataset_len(eval_metadata)

    tf.logging.info("Evalset length: {} examples".format(dataset_len))

    params = tf.contrib.training.HParams(
        num_samples_per_epoch=dataset_len,
        weight_decay=FLAGS.weight_decay
    )

    run_config = tf.estimator.RunConfig()
    run_config = run_config.replace(model_dir=FLAGS.model_dir)

    model_fn = estimator_specs.get_model_fn()

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        params=params,
        config=run_config,
        warm_start_from=FLAGS.warm_start_ckpt
    )

    preproc_fn_eval = estimator_specs.get_preproc_fn(is_training=False)

    eval_input_fn = estimator_specs.input_fn(
        batch_size=FLAGS.batch_size, metadata=eval_metadata, class_dict=estimator_specs.class_dict, is_tfrecord=is_tfrecord, epochs=1, image_size=FLAGS.image_size, preproc_fn=preproc_fn_eval)

    eval_metrics = estimator.evaluate(
        input_fn=eval_input_fn, name="Evaluation")
    return eval_metrics
