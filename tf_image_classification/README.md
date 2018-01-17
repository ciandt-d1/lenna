# TensorFlow Image Classification Framework

This framework is built upon TensorFlow 1.4.0

### Hierarchy

- **dataset.py** : Implements dataset handling using [Dataset API](https://www.tensorflow.org/programmers_guide/datasets)
- **estimator_specs.py** : Here it is defined the abstract class you should inherit from in order to create your own estimator.
- **hooks.py** : Contains some [hooks](https://www.tensorflow.org/api_docs/python/tf/train/SessionRunHook) used during training, such as: initialization of batch iterator and checkpoint load.
- **train_estimator.py** : The core of the framework. It contains the main training flow using [Estimator API](https://www.tensorflow.org/programmers_guide/estimators). It instanciate the model, input and hook functions for later usage during traning.
- **utils.py** : It contains some utility functions to perform IO with Google Cloud Storage and tf-record manipulation

### How to use

We do recommend to use the framework as a pip package. Look [here](https://bitbucket.org/ciandt_it/tf_image_classification/src) to know how to build and install it.
Once installed, all you need to do is to create a class that inherit from `tf_image_classification.estimator_specs.EstimatorSpec` and implement its abstract methods: 

- `get_preproc_fn(self, is_training)`: returns a preprocessing function that will be applied upon each batch
- `get_model_fn(self, checkpoint_path)`: returns a function that builds the graph used for training, evaluation and inference. See [here](https://www.tensorflow.org/api_docs/python/tf/contrib/learn/ModeKeys) for mor info.
- `metric_ops(self, labels, predictions)`: returns a dictionary with metrics (_e.g._ accuracy, precision, recall, etc...). For a full list of available metrics, look [here](https://www.tensorflow.org/api_docs/python/tf/metrics)
- `input_fn(self, ...)`: returns a function that provides batches to training procedure. You don't need to overload this method once it's already implemented in the base class and it currently supports tf-records and .csv as metadata.

Use [this example](https://bitbucket.org/ciandt_it/d1-ml-labs/src/master/2017-11-Demographics-Estimation/train_demographics.py?at=master&fileviewer=file-view-default) for demographics estimation as a reference code.
Now I will suppose that you already have written your own estimator, say _myEstimator.py_. Let's take a look how to run it.

### Running locally - Example

All the flags are defined on _train_estimator.py_. 

```bash
python myEstimator.py --batch_size 64 --train_steps 10000 --train_metadata tfrecords_path/train* --eval_metadata tfrecords_path/eval* --checkpoint_path checkpoint_path/pretrained_ckpt.ckpt --model_dir ./models --eval_freq 10 --eval_throttle_secs 30 --learning_rate 0.00001
```
- --checkpoint_path : points to a **.ckpt** file from a pre-trained model, like VGG, Inception, MobileNet and so on.
- --model_dir : where to save training files (checkpoints and summaries for tensorboard)
- --eval_throttle_secs : time interval until perform evaluation

### Running on Google ML Engine

First, you must package your application (explain here how).
```bash
gcloud ml-engine jobs submit training JOB_ID --job-dir=gs://bucket/stagging_folder/ --module-name myEstimatorPkg.myEstimator --packages myEstimator.tar.gz,tf_image_classification-1.3.1.tar.gz,slim-0.1.tar.gz --region us-east1 --config cloud.yml --  --batch_size 128 --train_steps 1000 --train_metadata gs://bucket/tfrecords/train* --eval_metadata gs://bucket/tfrecords/eval* --checkpoint_path gs://bucket/pretrained_checkpoints/pretrained_model.ckpt --model_dir gs://bucket/trained-checkpoints/ --eval_freq 10 --eval_throttle_secs 120 --learning_rate 0.00001
```

OBS: **train_estimator.py** trains the model using the method [`train_and_evaluate`](https://www.tensorflow.org/api_docs/python/tf/estimator/train_and_evaluate) that runs seamlessly both locally and distributed training, so you don't need to write a single line of code to run your model distributed into a ML Engine cluster.


## TODO
- Bring optimizer to framework (today it is app-dependent)
- Study whatelse flags that can be brought here.
- Implement MNIST (or anyother canonical problem) as a reference code.