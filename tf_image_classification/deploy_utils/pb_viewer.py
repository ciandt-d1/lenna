# -*- coding: utf-8 -*-
"""
.. module:: deploy_utils
   :platform: Unix
   :synopsis: Utilities to help you deploy your models

.. moduleauthor:: Rodrigo Pereira <rodrigofp@ciandt.com>

"""

import sys
import argparse
import tensorflow as tf
from tensorflow.python.platform import gfile


def generate_events(pb_file, output_events_file):
    """ Read a frozen model and write the events for Tensorboard

        Args:
            ``pb_file`` (str): **.pb** frozen model file
            
            ``output_events_file`` (str): output file to save events

    """
    
    with tf.Session() as sess:    
        with gfile.FastGFile(pb_file, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            g_in = tf.import_graph_def(graph_def)        
        train_writer = tf.summary.FileWriter(output_events_file)
        train_writer.add_graph(sess.graph)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_graph_pb", required=True)
    parser.add_argument("-o", "--output_events_file", required=True)
    args = vars(parser.parse_args())

    generate_events(args['input_graph_pb'],args['output_events_file'])

