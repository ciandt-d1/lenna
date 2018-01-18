# Mini MNIST Tutorial

This tutorial aims to provide a reference code using the tf_image_classification framework.
The dataset used is a subset of the MNIST dataset. It is located [here](), and also it is stored on GCS, both [images](gs:///morghulis/mini_mnist/images/) and [tf-records](gs:///morghulis/mini_mnist/tf_records/).
This tutorial comprises dataset preprocessing using DataFlow, creation of your own estimator, training and model post-processing for deployment.
The network used to train was [Inception_V4](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models).
If you want only use a given network architecture and not its pre-trained weights, just don't use the option `--checkpoint_path`.
For this tutorial, it was used the networks define on [TensorFlow Slim](https://github.com/tensorflow/models/tree/master/research/slim/nets).
This tutorial was built using Slim as a pip package. See [here]() how to build and install slim.

OBS: Remember that the GCS paths presented here are just examples. Alter them according to your needs and your project_id on GCP.

## Transform your images and labels into tf-records using Dataflow
The dataset used in this tutorial consists of only 600 images, so you can easily perform this operation locally.
Once dealing with a huge amount of data, it is preferable to preprocess your dataset on cloud. Using DataFlow you can achieve this seamlessly by just adding the flags `--cloud` and `--num_workers`

#### HOW TO RUN

##### Train
###### Cloud
```bash
python dataflow_jpeg_to_tfrecord.py --input_path gs://morghulis/mini_mnist/trainset_gcs.csv --output_path gs://morghulis/mini_mnist/tf_records/train --num_workers 10 --job_name mini-mnist-train --cloud
```
###### Local
```bash
python dataflow_jpeg_to_tfrecord.py --input_path ./dataset/trainset_local.csv --output_path ./dataset/tf_records/train
```

##### Eval
###### Cloud
```bash
python dataflow_jpeg_to_tfrecord.py --input_path gs://morghulis/mini_mnist/evalset_gcs.csv --output_path gs://morghulis/mini_mnist/tf_records/eval --num_workers 10 --job_name mini-mnist-eval --cloud
```

###### Local
```bash
python dataflow_jpeg_to_tfrecord.py --input_path ./dataset/evalset_local.csv --output_path ./dataset/tf_records/eval
```

## Estimator 
Take a look on `MiniMNIST` class on **train_mini_mnist.py**. There it is implemented the following methods from `tf_image_classification.estimator_specs.EstimatorSpec` base class


- `get_preproc_fn(self, is_training)`: returns a preprocessing function that will be applied upon each batch
- `get_model_fn(self, checkpoint_path)`: returns a function that builds the graph used for training, evaluation and inference. See [here](https://www.tensorflow.org/api_docs/python/tf/contrib/learn/ModeKeys) for mor info.
- `metric_ops(self, labels, predictions)`: returns a dictionary with metrics (_e.g._ accuracy, precision, recall, etc...). For a full list of available metrics, look [here](https://www.tensorflow.org/api_docs/python/tf/metrics)
- `input_fn(self, ...)`: returns a function that provides batches to training procedure. Actually, this isn't implemented by `MiniMNIST` once it's already implemented in the base class and it currently supports both tf-records and .csv as metadata.

#### HOW TO RUN


###### Local

```bash
python train_mini_mnist.py --batch_size 1 --train_steps 100 --train_metadata ./dataset/tf_records/train* --eval_metadata ./dataset/tf_records/eval* --checkpoint_path ./checkpoints/inception_v4.ckpt --model_dir ./trained_models --eval_freq 100 --eval_throttle_secs 30 --learning_rate 0.001 --image_size 299
```

OBS: Inception_V4 was trained with 299x299 images, so as the original images are 28x28, it is necessary to specify the flag `--image_size`

###### ML Engine

Before running the training on ML Engine, you must package your project first. You can do this by running the following command
```bash
python setup.py sdist
```

You'll se that a directory `dist` is created and it contains your project package as a **tar.gz** file.
As the code depends both of the _tf_image_classification framework_ and _slim_, their packages needed to be generated.
In our case, they are already on GCS, so we won't need to package them, but just make a reference when submiting the job.
However, you can also make references for these packages locally.

- _tf_image_classifier_ : **gs://tfcf/releases/tf_image_classification-1.3.1.tar.gz**
- _slim_ : **gs://morghulis/libs/object-detection-1.4.1/slim-0.1.tar.gz**

```bash
JOB_ID="MINI_MNIST_${USER}_$(date +%Y%m%d_%H%M%S)"

gcloud ml-engine jobs submit training ${JOB_ID} --job-dir=gs://mini_mnist/${JOB_ID} --module-name mini_mnist.train_mini_mnist --packages dist/mini_mnist-0.1.tar.gz,gs://tfcf/releases/tf_image_classification-1.3.1.tar.gz,gs://morghulis/libs/object-detection-1.4.1/slim-0.1.tar.gz --region us-east1 --config ./cloud.yml --  --batch_size 32 --train_steps 1000 --train_metadata gs://mini_mnist/tf_records/train* --eval_metadata gs://mini_mnist/tf_records/eval* --checkpoint_path gs://mini_mnist/pretrained_ckpt/inception_v4.ckpt --model_dir gs://mini_mnist/trained_models/${JOB_ID} --eval_freq 10 --eval_throttle_secs 120 --learning_rate 0.00001 --image_size 299
```
On **cloud.yml** it is defined the cluster specifications
```yaml
trainingInput:
  runtimeVersion: "1.4"
  scaleTier: CUSTOM
  masterType: standard_gpu
  workerCount: 5
  workerType: standard_gpu
  parameterServerCount: 3
  parameterServerType: standard
```

## Rename input and output tensors
Up to this time, the model trained both locally or distributed following the aforementioned steps using TF-1.4.0 cannot be frozen directly, because some weird ops appears on the graph regarding batchnorm layers and an error is raised when you try to use it.
A workaround is currently implement on **rename_nodes.py**. Please, take a look on it.
What it's basically done is to create the graph from the source code (and not by **.meta** file), load the checkpoint and save it again. Yeah, just that. The good point of this approach is that you can rename your input tensor for easy usage on deployment, so you don't have to spend minutes searching it on the graph itself.

#### HOW TO RUN

```bash
python rename_nodes.py --checkpoint_path ./trained_models/model.ckpt-100 --output_checkpoint_path ./trained_models/renamed_model/renamed_model.ckpt --image_size 299
```
## Post-processing
For a detailed explanation of how post-process your model, take a look on these [utility functions](https://bitbucket.org/ciandt_it/tf_image_classification/src/master/tf_image_classification/utils/README.md?at=master&fileviewer=file-view-default).
Below, it is presented the code to perfom the operation on our example. 
Run the commands below from utils directory.

## Freeze graph
```bash
python freeze_graph.py --model_dir ../../example/mini_mnist/trained_models/renamed_model/ --output_tensors prediction --output_pb ../../example/mini_mnist/trained_models/frozen_model.pb
```

## Prune useless nodes
```bash
python optimize_for_inference.py --input ../../example/mini_mnist/trained_models/renamed_model/frozen_model.pb --output ../../example/mini_mnist/trained_models/renamed_model/opt_frozen_model.pb --input_names input_image --output_names prediction
```

## Quantization (optional)
```bash
python quantize_graph.py  --input ../../example/mini_mnist/trained_models/renamed_model/opt_frozen_model.pb --output ../../example/mini_mnist/trained_models/renamed_model/quantized_model.pb --output_node_names prediction  --print_nodes --mode eightbit --logtostderr
```

## Usage