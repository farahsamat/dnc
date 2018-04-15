import numpy as np
import tensorflow as tf
from dnc.controller import BaseController


class RecurrentController(BaseController):

    def network_vars(self):
        self.lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(256)
        self.state = self.lstm_cell.zero_state(self.batch_size, tf.float32)

    def network_op(self, X, state):
        X = tf.convert_to_tensor(X)
        return self.lstm_cell(X, state)

    def get_state(self):
        return self.state

    def update_state(self, new_state):
        return tf.no_op()