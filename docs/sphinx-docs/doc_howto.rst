#########################
HOW-TO
#########################


.. note::
	Lenna was built upon **TensorFlow 1.8.0**

***********
Core files
***********

* `dataset.py <https://github.com/ciandt-d1/lenna/blob/master/tf_image_classification/dataset.py>`_ : Implements dataset manipulation using the `Dataset API <https://www.tensorflow.org/programmers_guide/datasets>`_
* `estimator_specs.py <https://github.com/ciandt-d1/lenna/blob/master/tf_image_classification/estimator_specs.py>`_ : Here it is defined the abstract class you should inherit from in order to create your own estimator.
* `train_estimator.py <https://github.com/ciandt-d1/lenna/blob/master/tf_image_classification/train_estimator.py>`_ : The core of the framework. It contains the main training flow using `Estimator API <https://www.tensorflow.org/programmers_guide/estimators>`_ . It instantiates the model and input functions for later usage during traning and evaluation.
* `utils.py <https://github.com/ciandt-d1/lenna/blob/master/tf_image_classification/utils.py>`_ : It contains some utility functions to perform I/O with Google Cloud Storage and tf-record manipulation

*************
Installation
*************

It's recommended to use the framework as a pip package.
So after downloading the code:

.. code-block:: bash

	cd /path/to/lenna
	python setup sdist
	pip install ./dist/lenna.tar.gz --upgrade

Once installed, all you need to do is to create a class that inherit from :class:`~tf_image_classification.estimator_specs.EstimatorSpec` and implement its abstract methods.


.. note::
    
    Don't worry. We'll publish lenna on `PyPI <https://pypi.org/>`_  as soon as possible.

*************************
Running locally - Example
*************************

.. code-block:: bash

	python myEstimator.py \
            --batch_size 64 \
            --train_steps 10000 \
            --train_metadata tfrecords_path/train* \
            --eval_metadata tfrecords_path/eval* \
            --warm_start_ckpt /path/to/pretrained_ckpt.ckpt \
            --model_dir /path/to/model.ckpt \
            --eval_freq 10 \
            --eval_throttle_secs 30 \
            --learning_rate 0.00001 \
            --batch_size 32


*************************************
Running on Google ML Engine - Example
*************************************

First, you must package your application as a pip package.
Also, you must have a `Google Cloud Platform <https://cloud.google.com/>`_ account.

.. code-block:: bash

	gcloud ml-engine jobs submit training JOB_ID \
            --job-dir=gs://bucket/stagging_folder/ \
            --module-name myEstimatorPkg.myEstimator \
            --packages myEstimator.tar.gz,lenna.tar.gz \
            --region us-east1 \
            --config cloud.yml --  \
            --batch_size 128 \
            --train_steps 1000 \
            --train_metadata gs://bucket/tfrecords/train* \
            --eval_metadata gs://bucket/tfrecords/eval* \
            --warm_start_ckpt gs://bucket/path/to/pretrained_model.ckpt \
            --model_dir gs://bucket/path/to/model.ckpt \
            --eval_freq 10 \
            --eval_throttle_secs 120 \
            --learning_rate 0.00001 \
            --batch_size 32


.. note::

    * All flags are defined on :ref:`flags`.
    * :func:`~tf_image_classification.train_estimator.train` uses the method `train_and_evaluate <https://www.tensorflow.org/api_docs/python/tf/estimator/train_and_evaluate>`_ that runs seamlessly both locally and distributed training, so you **don't need to write a single line of code** to run your model distributed into a ML Engine cluster.
    * For a full example of usage, please read our :ref:`mini-mnist-tutorial`.

.. _flags:

******
FLAGS
******

Lenna uses `TensorFlow Flags <https://www.tensorflow.org/api_docs/python/tf/flags>`_ as argument parser.
One advantage over python standard `ArgumentParser <https://docs.python.org/2/library/argparse.html>`_ is that the flags can be retrieved
throughout any **.py** file within the project.
In the context of the framework, the flags below are can be retrieved by your program.

Example
^^^^^^^

.. code-block:: python

    import tensorflow as tf
    FLAGS = tf.app.flags.FLAGS
    print(FLAGS.learning_rate)

Standard Flags
^^^^^^^^^^^^^^

* `model_dir` : Output directory for model and training stats
    * Default value: **None** 
* `warm_start_ckpt` : Checkpoint to load pre-trained model
    * Default value: **None**
* `train_metadata` : Path to train metadata (**.tfrecord** Only!)
    * Default value: **None**
* `eval_metadata` : Path to eval metadata (**.tfrecord** Only!)
    * Default value: **None**
* `batch_size` : Batch size
    * Default value: **1**
* `train_steps` : Train steps
    * Default value: **20**
* `image_size` : Image size used for image preprocessing, if any.
    * Default value: **299**
* `eval_freq` : How many eval batches to evaluate
    * Default value: **5**
* `eval_throttle_secs` : Evaluation every `eval_throttle_secs` seconds
    * Default value: **120**
* `debug` : Debug mode (does not shuffle dataset)
    * Default value: **False**

Optimizer Flags
^^^^^^^^^^^^^^^

* `weight_decay` : Weight decay for batch norm layers.
    * Defaut value: **0.00004**
* `optimizer` : Name of optimizer
    * Default value: **rmsprop**
    * Possible values: 
	    * `adadelta <https://www.tensorflow.org/api_docs/python/tf/train/AdadeltaOptimizer>`_
	    * `adagrad <https://www.tensorflow.org/api_docs/python/tf/train/AdagradOptimizer>`_
	    * `adam <https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer>`_
	    * `ftrl <https://www.tensorflow.org/api_docs/python/tf/train/FtrlOptimizer>`_
	    * `momentum <https://www.tensorflow.org/api_docs/python/tf/train/MomentumOptimizer>`_ 
	    * `sgd <https://www.tensorflow.org/api_docs/python/tf/train/GradientDescentOptimizer>`_
	    * `rmsprop <https://www.tensorflow.org/api_docs/python/tf/train/RMSPropOptimizer>`_

* `adadelta_rho` : The decay rate for adadelta
    * Default Value: **0.95**
* `adagrad_initial_accumulator_value` : Starting value for the AdaGrad accumulators
    * Default Value: **0.1**
* `adam_beta1` : The exponential decay rate for the 1st moment estimates
    * Default Value: **0.9**
* `adam_beta2` : The exponential decay rate for the 2nd moment estimates
    * Default Value: **0.999**
* `opt_epsilon` : Epsilon term for the optimizer
    * Default value: **1.0**
* `ftrl_learning_rate_power` : The learning rate power for ftrl optimizer
    * Default Value: **-0.5**
* `ftrl_initial_accumulator_value` : Starting value for the FTRL accumulators
    * Default Value: **0.1**
* `ftrl_l1` : The FTRL l1 regularization strength
    * Default Value: **0.0**
* `ftrl_l2` : The FTRL l2 regularization strength
    * Default Value: **0.0**
* `momentum` : Momentum for MomentumOptimizer
    * Default Value: **0.9**
* `rmsprop_momentum` : Momentum for RMSPropOptimizer
    * Default Value: **0.9**
* `rmsprop_decay` : Decay term for RMSProp
    * Default Value: **0.9**


Learning Rate Flags
^^^^^^^^^^^^^^^^^^^^

* `learning_rate_decay_type` : Specifies how the learning rate is decayed.
	* Default Value: **exponential**
	* Possible values:
		* `fixed <https://www.tensorflow.org/versions/master/api_docs/python/tf/constant>`_
		* `exponential <https://www.tensorflow.org/api_docs/python/tf/train/exponential_decay>`_
		* `polynomial <https://www.tensorflow.org/api_docs/python/tf/train/polynomial_decay>`_
   
* `learning_rate` : Initial learning rate
    * Default Value: **0.01**
* `end_learning_rate` : The minimal end learning rate used by a polynomial decay learning rate
    * Default Value: **0.0001**
* `learning_rate_decay_factor` : Learning rate decay factor
    * Default Value: **0.94**
* `label_smoothing` : The amount of label smoothing
    * Default Value: **0.0**
* `num_epochs_per_decay` : Number of epochs after which learning rate decays
    * Default Value: **2.0**
* `sync_replicas` : Whether or not to synchronize the replicas during training
    * Default Value: **False**
* `replicas_to_aggregate` : The Number of gradients to collect before updating params
    * Default Value: **1**


Fine Tuning Flags
^^^^^^^^^^^^^^^^^^

* `trainable_scopes` : Comma-separated list of scopes to train. If `None`, all variables will be trained.
    * Default Value : **None**
* `checkpoint_exclude_scopes` : Comma-separated list of scopes to exclude when loading checkpoint weights. If `None`, restore all variables.
    * Default Value : **None**
* `checkpoint_restore_scopes`: Comma-separated list of scopes of variables to restore from a checkpoint.
	* Default Value : **None**

Checkpoint Flags
^^^^^^^^^^^^^^^^^

* `save_summary_steps` : Save summaries every this many steps
	* Default Value: **100**
                            
* `save_checkpoints_steps` : Save checkpoints every this many steps. Can not be specified with `save_checkpoints_secs`
	* Default Value: **None**
                            
* `save_checkpoints_secs` : Save checkpoints every this many seconds. Can not be specified with save_checkpoints_steps
	* Default Value: **None**
                            
* `keep_checkpoint_max` : The maximum number of recent **ckpt** files to keep. -1 to keep all checkpoints
	* Default Value: **5**

* `export_saved_model` : Whether or not to export saved model
	* Default Value: **True**
