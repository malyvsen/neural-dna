import numpy as np
import tensorflow as tf



class Weights:
    def __init__(self, initializer, depth):
        self.initializer = initializer
        self.depth = depth


    def __call__(self, num_inputs, num_outputs):
        x = tf.Variable(self.initializer(shape=(num_inputs, self.depth)))
        y = tf.Variable(self.initializer(shape=(self.depth, num_outputs)))
        return tf.matmul(x, y)



class Biases:
    def __init__(self, initializer, depth):
        self.initializer = initializer
        self.depth = depth


    def __call__(self, num_units):
        x = tf.Variable(self.initializer(shape=(1, self.depth)))
        y = tf.Variable(self.initializer(shape=(self.depth, num_units)))
        return tf.reshape(tf.matmul(x, y), shape=(num_units,))
