import os
import argparse
import tensorflow as tf


def freeze_graph(model_dir, output_tensors, output_pb):
    """Extract the sub graph defined by the output nodes and convert 
    all its variables into constant 
    Args:
        model_dir: the root folder containing the checkpoint state file
        output_tensors: a string, containing all the output node's names, 
                            comma separated
    """
    if not tf.gfile.Exists(model_dir):
        raise AssertionError(
            "Export directory doesn't exists. Please specify an export "
            "directory: %s" % model_dir)

    if not output_tensors:
        tf.loogin.error("You need to supply the name of a node to --output_tensors.")
        return -1

    # Retrievecheckpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path
    
    # Clear original devices from graph
    clear_devices = True

    with tf.Session(graph=tf.Graph()) as sess:
        # Import graph from .meta file
        saver = tf.train.import_meta_graph(
            input_checkpoint + '.meta', clear_devices=clear_devices)
        
        # and restore weights
        saver.restore(sess, input_checkpoint)

        # Convert variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,         
            tf.get_default_graph().as_graph_def(),            
            output_tensors.split(","),
            variable_names_blacklist=['global_step']
        )

        # Dump frozen model
        with tf.gfile.GFile(output_pb, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        tf.logging.info("%d ops in the final graph." % len(output_graph_def.node))

    #return output_graph_def


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="",
                        help="Model folder to export")
    parser.add_argument("--output_tensors", type=str, default="",
                        help="The name of the output nodes, comma separated.")
    parser.add_argument("--output_pb", type=str,
                        default="/frozen_model.pb", help="Output pb file")
    args = parser.parse_args()

    freeze_graph(args.model_dir, args.output_tensors, args.output_pb)
