# -*- coding: utf-8 -*-

import tensorflow as tf


class IteratorInitializerHook(tf.train.SessionRunHook):
    """Hook to initialise data iterator after Session is created."""

    def __init__(self):
        super(IteratorInitializerHook, self).__init__()
        self.iterator_initializer_func = None

    def after_create_session(self, session, coord):
        tf.logging.info("Initialize batch iterator")
        self.iterator_initializer_func(session)


class LoadCheckpointHook(tf.train.SessionRunHook):
    """Hook to initialise data iterator after Session is created."""

    def __init__(self):
        super(LoadCheckpointHook, self).__init__()
        self.load_checkpoint_initializer_func = None

    def after_create_session(self, session, coord):
        if self.load_checkpoint_initializer_func is not None:
            tf.logging.info("Loading weights from checkpoint")
            self.load_checkpoint_initializer_func(session)
