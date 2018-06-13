# Utility functions
The following functions will help you to deploy your models.

### Freeze graph
As you may know, TensorFlow saves basically four files:
- checkpoint 
- model.ckpt.data-00000-of-00001
- model.ckpt.index
- model.ckpt.meta

It may be awkward to deal with all of these files for deployment and it is preferrible to deal with only one that contains both graph and weights.
You can do that by freezing the graph with **freeze_graph.py**. What it basically does is to transform all the graph variables to constants and dump a single **.pb** file.

##### How to use

```bash
python freeze_graph.py --model_dir model_dir --output_tensors output_tensors --output_pb frozen_model.pb
```
- --model_dir: directory where  **.ckpt** is stored
- --output_tensors: comma separated list of tensors to be the model outputs (_e.g._: out1,out2,out3 - No spaces)
- --output_pb: name of the output frozen **.pb** model.

### Optimize for inference 
On deployment mode, there may be some useless nodes on the graph, such as nodes for optimization, loss and accuracy calculations.
You can prune the useless nodes using **optimize_for_inference.py**

##### How to use
```bash
python optimize_for_inference.py --input frozen_model.pb --output opt_frozen_model.pb --input_names input_tensors --output_names output_tensors
```

- --input: input frozen **.pb** model
- --output: output optimized **.pb** model
- --input_names: comma separated list of tensors to be the model inputs (_e.g._: in1,in2,in3 - No spaces)
- --output_names: comma separated list of tensors to be the model outputs (_e.g._: out1,out2,out3 - No spaces)

### Quantization
Once your model is frozen and optimized (optional step), your may want to deploy it on a hardware-limited device (_e.g._: mobile phone).
One of the things you can do is to quantize your graph, that is, transform your 32bits floating-point weights into 8bits fixed-points weights.
This saves you lots of computational steps and disk size.
For more [look here](https://www.tensorflow.org/performance/quantization).

##### How to use
```bash
python quantize_graph.py  --input frozen_model.pb --output quantized_model.pb --output_node_names output_tensors  --print_nodes --mode eightbit --logtostderr
```

- --input: input frozen **.pb** model
- --output: output quantized **.pb** model
- --output_node_names: comma separated list of tensors to be the model outputs (_e.g._: out1,out2,out3 - No spaces)
- --print_nodes: log tensonrs
- --mode: always use _eightbit_ option in order to convert your weights to 8bits fixed-point weights;
- --logtostderr: log everything else

### Visualize graph
If you want to check your graph using tensorboard, you first need to generate the events files.
For this, use **pb_viewer.py**

##### How to use

```bash
python pb_viewer.py --input_graph_pb frozen_model.pb --output_events_path event_path
```

Then
```bash
tensorboard --logdir event_path
```