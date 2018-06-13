# TensorFlow Image Classification Framework

This framework is built upon TensorFlow 1.8.0

### Hierarchy

- **dataset.py** : Implements dataset manipulation using the [Dataset API](https://www.tensorflow.org/programmers_guide/datasets)
- **estimator_specs.py** : Here it is defined the abstract class you should inherit from in order to create your own estimator.
- **hooks.py** : Contains some [hooks](https://www.tensorflow.org/api_docs/python/tf/train/SessionRunHook) used during training, such as: initialization of batch iterator and checkpoint loading.
- **train_estimator.py** : The core of the framework. It contains the main training flow using [Estimator API](https://www.tensorflow.org/programmers_guide/estimators). It instanciates the model, input and hook functions for later usage during traning.
- **utils.py** : It contains some utility functions to perform IO with Google Cloud Storage and tf-record manipulation

### How to use

We do recommend to use the framework as a pip package. Look [here](https://bitbucket.org/ciandt_it/tf_image_classification/src) to know how to build and install it.
Once installed, all you need to do is to create a class that inherit from `tf_image_classification.estimator_specs.EstimatorSpec` and implement its abstract methods: 

- `get_preproc_fn(self, is_training)`: returns a preprocessing function that will be applied upon each batch
- `get_model_fn(self, checkpoint_path)`: returns a function that builds the graph used for training, evaluation and inference. See [here](https://www.tensorflow.org/api_docs/python/tf/contrib/learn/ModeKeys) for more info.
- `metric_ops(self, labels, predictions)`: returns a dictionary with metrics (_e.g._ accuracy, precision, recall, etc...). For a full list of available metrics, look [here](https://www.tensorflow.org/api_docs/python/tf/metrics)
- `input_fn(self, ...)`: returns a function that provides batches to training procedure. You don't need to overload this method once it's already implemented in the base class and it currently supports tf-records and .csv as metadata.

Use [this example](https://bitbucket.org/ciandt_it/d1-ml-labs/src/master/2017-11-Demographics-Estimation/train_demographics.py?at=master&fileviewer=file-view-default) for demographics estimation as a reference code.
Now I will suppose that you already have written your own estimator, say _myEstimator.py_. Let's take a look how to run it.

### Running locally - Example

All the flags are defined on _train_estimator.py_. 

```bash
python myEstimator.py --batch_size 64 --train_steps 10000 --train_metadata tfrecords_path/train* --eval_metadata tfrecords_path/eval* --checkpoint_path checkpoint_path/pretrained_ckpt.ckpt --model_dir ./models --eval_freq 10 --eval_throttle_secs 30 --learning_rate 0.00001
```
- --checkpoint_path : points to a **.ckpt** file from a pre-trained model, like VGG, Inception, MobileNet and so on. If you don't specify it, your model will be trained from scratch
- --model_dir : where to save training files (checkpoints and summaries for tensorboard)
- --eval_throttle_secs : time interval until perform evaluation

### Running on Google ML Engine

First, you must package your application (see reference code).
```bash
gcloud ml-engine jobs submit training JOB_ID --job-dir=gs://bucket/stagging_folder/ --module-name myEstimatorPkg.myEstimator --packages myEstimator.tar.gz,tf_image_classification-1.3.1.tar.gz,slim-0.1.tar.gz --region us-east1 --config cloud.yml --  --batch_size 128 --train_steps 1000 --train_metadata gs://bucket/tfrecords/train* --eval_metadata gs://bucket/tfrecords/eval* --checkpoint_path gs://bucket/pretrained_checkpoints/pretrained_model.ckpt --model_dir gs://bucket/trained-checkpoints/ --eval_freq 10 --eval_throttle_secs 120 --learning_rate 0.00001
```

OBS: **train_estimator.py** trains the model using the method [`train_and_evaluate`](https://www.tensorflow.org/api_docs/python/tf/estimator/train_and_evaluate) that runs seamlessly both locally and distributed training, so you don't need to write a single line of code to run your model distributed into a ML Engine cluster.

## FLAGS

### Common
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

### Optimizers
* `weight_decay` : The weight decay on the model weights (_e.g._ batchnorm layers)
    * Defaut value: **0.00004**
* `optimizer` : Name of optimizer
    * Default value: **rmsprop**
    * Possible values: [adadelta](https://www.tensorflow.org/api_docs/python/tf/train/AdadeltaOptimizer), [adagrad](https://www.tensorflow.org/api_docs/python/tf/train/AdagradOptimizer), [adam](https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer), [ftrl](https://www.tensorflow.org/api_docs/python/tf/train/FtrlOptimizer), [momentum](https://www.tensorflow.org/api_docs/python/tf/train/MomentumOptimizer), [sgd](https://www.tensorflow.org/api_docs/python/tf/train/GradientDescentOptimizer) or [rmsprop](https://www.tensorflow.org/api_docs/python/tf/train/RMSPropOptimizer)
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


### Learning rate

* `learning_rate_decay_type` : Specifies how the learning rate is decayed. One of [fixed](https://www.tensorflow.org/versions/master/api_docs/python/tf/constant), [exponential](https://www.tensorflow.org/api_docs/python/tf/train/exponential_decay),or [polynomial](https://www.tensorflow.org/api_docs/python/tf/train/polynomial_decay)
    * Default Value: **exponential**
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

### Fine Tuning

* `trainable_scopes` : Comma-separated list of scopes to train. If `None`, all variables will be trained.
    * Default Value: `None`
* `checkpoint_exclude_scopes` : Comma-separated list of scopes to exclude when loading checkpoint weights. If `None`, restore all variables.
    * Default Value: `None`