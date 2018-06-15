#########################
How to use the framework
#########################


.. note::
	This framework is built upon **TensorFlow 1.8.0**

***********
Main files
***********

* **dataset.py** : Implements dataset manipulation using the `Dataset API <https://www.tensorflow.org/programmers_guide/datasets>`_
* **estimator_specs.py** : Here it is defined the abstract class you should inherit from in order to create your own estimator.
* **train_estimator.py** : The core of the framework. It contains the main training flow using `Estimator API <https://www.tensorflow.org/programmers_guide/estimators>`_ . It instanciates the model, input and hook functions for later usage during traning.
* **utils.py** : It contains some utility functions to perform IO with Google Cloud Storage and tf-record manipulation

***********
How to use
***********

We do recommend to use the framework as a pip package.
So after downloading the code:


.. code-block:: bash

	cd /path/to/tf_image_classification
	python setup sdist
	pip install ./dist/tf_image_classification.3.0.0.tar.gz --upgrade

Once installed, all you need to do is to create a class that inherit from :class:`~tf_image_classification.estimator_specs.EstimatorSpec` and implement its abstract methods.

*************************
Running locally - Example
*************************

.. code-block:: bash

	python myEstimator.py --batch_size 64 --train_steps 10000 \
	--train_metadata tfrecords_path/train* --eval_metadata tfrecords_path/eval* \
	--checkpoint_path checkpoint_path/pretrained_ckpt.ckpt --model_dir ./models \
	--eval_freq 10 --eval_throttle_secs 30 --learning_rate 0.00001 

*************************************
Running on Google ML Engine - Example
*************************************

First, you must package your application as a pip package.

.. code-block:: bash

	gcloud ml-engine jobs submit training JOB_ID --job-dir=gs://bucket/stagging_folder/ \
	--module-name myEstimatorPkg.myEstimator \
	--packages myEstimator.tar.gz,tf_image_classification-3.0.0.tar.gz,slim-0.1.tar.gz \
	--region us-east1 --config cloud.yml --  --batch_size 128 --train_steps 1000 \
	--train_metadata gs://bucket/tfrecords/train* \ --eval_metadata gs://bucket/tfrecords/eval* \
	--checkpoint_path gs://bucket/pretrained_checkpoints/pretrained_model.ckpt \
	--model_dir gs://bucket/trained-checkpoints/ --eval_freq 10 \
	--eval_throttle_secs 120 --learning_rate 0.00001


.. note::

	:func:`~tf_image_classification.train_estimator.train` uses the method `train_and_evaluate <https://www.tensorflow.org/api_docs/python/tf/estimator/train_and_evaluate>`_ that runs seamlessly both locally and distributed training, so you **don't need to write a single line of code** to run your model distributed into a ML Engine cluster.



******
FLAGS
******

Common
=======

* `model_dir` : Output directory for model and training stats
    * Default value: **None** 
* `checkpoint_path` : Checkpoint to load pre-trained model
    * Default value: **None**
* `train_metadata` : Path to train metadata ( **.csv** or **.tfrecord**)
    * Default value: **None**
* `eval_metadata` : Path to eval metadata ( **.csv** or **.tfrecord**)
    * Default value: **None**
* `batch_size` : Batch size
    * Default value: **1**
* `train_steps` : Train steps
    * Default value: **20**
* `image_size` : Image size for resize on preprocessing
    * Default value: **299**
* `eval_freq` : How many eval batches to evaluate
    * Default value: **5**
* `eval_throttle_secs` : Evaluation every `eval_throttle_secs` seconds
    * Default value: **120**
* `debug` : Debug mode (does not shuffle dataset)
    * Default value: **False**

Optimizers
===========

* `weight_decay` : The weight decay on the model weights (_e.g._ batchnorm layers)
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


Learning rate
==============

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


Fine Tuning
============

* `trainable_scopes` : Comma-separated list of scopes to train. If `None`, all variables will be trained.
    * Default Value : `None`
* `checkpoint_exclude_scopes` : Comma-separated list of scopes to exclude when loading checkpoint weights. If `None`, restore all variables.
    * Default Value : `None`
* `checkpoint_restore_scopes`: Comma-separated list of scopes of variables to restore from a checkpoint.
	* Default Value : `None`

Checkpoint
===========

* `save_summary_steps` : Save summaries every this many steps
	* Default Value: 100
                            
* `save_checkpoints_steps` : Save checkpoints every this many steps. Can not be specified with `save_checkpoints_secs`
	* Default Value: None
                            
* `save_checkpoints_secs` : Save checkpoints every this many seconds. Can not be specified with save_checkpoints_steps
	* Default Value: None
                            
* `keep_checkpoint_max` : The maximum number of recent checkpoint files to keep. -1 to keep every checkpoints
	* Default Value: 5
