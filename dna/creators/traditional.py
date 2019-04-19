import tensorflow as tf



class Weights:
    def __init__(self, initializer):
        self.initializer = initializer


    def __call__(self, num_inputs, num_outputs, representation_id):
        return tf.Variable(self.initializer(shape=(num_inputs, num_outputs)))



class Biases:
    def __init__(self, initializer):
        self.initializer = initializer


    def __call__(self, num_units, representation_id):
        return tf.Variable(self.initializer(shape=(num_units,)))
