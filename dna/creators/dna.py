import numpy as np
import tensorflow as tf
from model import Model



class Weights:
    def __init__(self, dna_architecture):
        self.dna_architecture = dna_architecture


    def __call__(self, num_inputs, num_outputs, representation_id):
        dna_inputs = []
        for input_id in np.linspace(0, 1, num_inputs):
            for output_id in np.linspace(0, 1, num_outputs):
                dna_inputs.append([input_id, output_id, representation_id])
        dna_inputs = tf.Variable(initial_value=dna_inputs, trainable=False) # a single "batch" of positions
        result = Model(dna_inputs, self.dna_architecture).outputs
        result = tf.reshape(result, (num_inputs, num_outputs))
        result /= (num_inputs * num_outputs) ** .5 # makes norm of vectors in matmul(?, weights) less sensitive to shape changes
        return result



class Biases:
    def __init__(self, dna_architecture):
        self.dna_architecture = dna_architecture


    def __call__(self, num_units, representation_id):
        dna_inputs = []
        for unit_id in np.linspace(0, 1, num_units):
            dna_inputs.append([unit_id, representation_id])
        dna_inputs = tf.Variable(initial_value=dna_inputs, trainable=False) # a single "batch" of positions
        result = Model(dna_inputs, self.dna_architecture).outputs
        result = tf.reshape(result, (num_units,))
        result /= num_units ** .5 # makes norm of vectors in (? + biases) less sensitive to shape changes
        return result
