Utility functions for deployment
==================================

The following functions will help you to deploy your models.


Freeze graph
#############

As you probably may know, TensorFlow saves basically four files when training a model:

* **checkpoint**
* **model.ckpt.data-00000-of-00001**
* **model.ckpt.index**
* **model.ckpt.meta**

It may be awkward to deal with all of these files for deployment and it is preferrible to deal with only one that contains both graph and weights.

You can do that by freezing the graph with **freeze_graph.py**. What it basically does is to transform all the graph variables to constants and dump a single **.pb** file.

You can both embed this function to your code or use it as an utility after your training procedure

Example
^^^^^^^^

.. code-block:: bash

	python freeze_graph.py --model_dir /path/to/ckpt/ --output_tensors tensor_list --output_pb /path/to/model.pb

.. autofunction:: tf_image_classification.deploy_utils.freeze_graph.freeze_graph


Visualize graph
################

If you want to visualize your graph using Tensorboard, you first need to generate the events files.

For this, use **pb_viewer.py**

You can both embed this function to your code or use it as an utility after freezing your model.

Example
^^^^^^^^

.. code-block:: bash

	python pb_viewer.py --input_graph_pb /path/to/model.pb --output_events_file /path/to/event_files

.. autofunction:: tf_image_classification.deploy_utils.pb_viewer.generate_events