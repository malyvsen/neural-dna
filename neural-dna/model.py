import tensorflow as tf
from architecture import Weights, Biases, Activation



class Model:
    def __init__(self, inputs, architecture):
        self.inputs = inputs
        self.representations = [self.inputs]
        for element in architecture:
            if isinstance(element, Weights):
                weights = element.creator(num_inputs=self.representations[-1].shape[1]._value, num_outputs=element.num_units, representation_id=len(self.representations))
                self.representations.append(tf.matmul(self.representations[-1], weights))
            elif isinstance(element, Biases):
                biases = element.creator(num_units=self.representations[-1].shape[1]._value, representation_id=len(self.representations))
                self.representations.append(self.representations[-1] + biases)
            elif isinstance(element, Activation):
                self.representations.append(element.function(self.representations[-1]))
            else:
                raise TypeError('unrecognized architecture element type')
        self.outputs = self.representations[-1]
