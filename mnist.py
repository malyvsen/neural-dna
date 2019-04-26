#%% imports
import tensorflow as tf
import numpy as np
from tqdm import trange
import dna
import utils


#%% model definitions
traditional_weights_creator = dna.creators.traditional.Weights(tf.initializers.glorot_uniform())
traditional_biases_creator = dna.creators.traditional.Biases(tf.initializers.constant(0.0))

dna_weights_creator = dna.creators.dna.Weights(tf.initializers.glorot_uniform(), depth=4)
dna_biases_creator = dna.creators.dna.Biases(tf.initializers.glorot_uniform(), depth=4)

traditional_architecture = dna.architecture.generate([10], traditional_weights_creator, traditional_biases_creator)
dna_architecture = dna.architecture.generate([10], dna_weights_creator, dna_biases_creator)

mnist_inputs = tf.placeholder(shape=(None, 28, 28), dtype=tf.float32)
mnist_reshaped = tf.reshape(mnist_inputs, (-1, 28*28))
labels = tf.placeholder(shape=(None,), dtype=tf.int32)
labels_one_hot = tf.one_hot(labels, 10)

def add_loss(model):
    model.loss = tf.losses.softmax_cross_entropy(labels_one_hot, model.outputs)

traditional_model = dna.Model(mnist_reshaped, traditional_architecture)
add_loss(traditional_model)
traditional_model.optimizer = tf.train.AdamOptimizer(1e-1).minimize(traditional_model.loss)

dna_model = dna.Model(mnist_reshaped, dna_architecture)
add_loss(dna_model)
dna_model.optimizer = tf.train.AdamOptimizer(1e-1).minimize(dna_model.loss)


#%% dataset-related
mnist = tf.keras.datasets.mnist.load_data()[0] # only training set


#%% training
session = tf.Session()
session.run(tf.global_variables_initializer())

def train(model, num_epochs, batch_size):
    num_steps = int(np.ceil(len(mnist[0]) * num_epochs / batch_size))
    for step in trange(num_steps):
        batch = utils.random_batch(data=mnist[0], labels=mnist[1], size=batch_size)
        session.run(model.optimizer, feed_dict={mnist_inputs: batch[0], labels: batch[1]})

train(traditional_model, num_epochs=1, batch_size=16)
train(dna_model, num_epochs=1, batch_size=16)


#%% evaluation
def show(model, title):
    weight_values = session.run(model.weights[0])
    weight_images = np.reshape(weight_values, (28, 28, 10))
    weight_images = np.moveaxis(weight_images, -1, 0)
    utils.show(weight_images, save_as=f'results/{title}.png', title=title)

show(traditional_model, 'traditional')
show(dna_model, 'dna')
