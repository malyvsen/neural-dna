#%% imports
import tensorflow as tf
import dna
import utils


#%% model definitions
autoencoder_transformations = [50, tf.nn.tanh, 50, tf.nn.tanh, 2, 50, tf.nn.tanh, 50, tf.nn.tanh, 28*28, tf.nn.relu]

traditional_weights_creator = dna.creators.traditional.Weights(tf.initializers.glorot_uniform())
traditional_biases_creator = dna.creators.traditional.Biases(tf.initializers.constant(0.0))

traditional_architecture = dna.architecture.generate(autoencoder_transformations, traditional_weights_creator, traditional_biases_creator)
dna_weights_architecture = dna.architecture.generate([256, tf.nn.relu, 32, tf.nn.relu, 256, tf.nn.relu, 1], traditional_weights_creator, traditional_biases_creator)
dna_biases_architecture = dna.architecture.generate([256, tf.nn.relu, 32, tf.nn.relu, 1], traditional_weights_creator, traditional_biases_creator)
dna_architecture = dna.architecture.generate(autoencoder_transformations, dna.creators.dna.Weights(dna_weights_architecture), dna.creators.dna.Biases(dna_biases_architecture))

mnist_inputs = tf.placeholder(shape=(None, 28, 28), dtype=tf.float32)
mnist_reshaped = tf.reshape(mnist_inputs, (-1, 28*28))

def add_loss(model):
    model.outputs_reshaped = tf.reshape(model.outputs, (-1, 28, 28))
    model.loss = tf.reduce_mean(tf.square(model.outputs_reshaped - mnist_inputs))

traditional_model = dna.Model(mnist_reshaped, traditional_architecture)
add_loss(traditional_model)
traditional_model.optimizer = tf.train.AdamOptimizer(1e-3).minimize(traditional_model.loss)

dna_model = dna.Model(mnist_reshaped, dna_architecture)
add_loss(dna_model)
dna_model.optimizer = tf.train.AdamOptimizer(1e-3).minimize(dna_model.loss)


#%% dataset-related
mnist = tf.keras.datasets.mnist.load_data()[0][0] # only images from training set


#%% training
session = tf.Session()
session.run(tf.global_variables_initializer())

for step in range(1024):
    batch = utils.random_batch(mnist, 16)
    session.run(traditional_model.optimizer, feed_dict={mnist_inputs: batch})
    session.run(dna_model.optimizer, feed_dict={mnist_inputs: batch})
    if step % 64 == 0:
        originals = utils.random_batch(mnist, 4)
        utils.show(originals, title='originals')
        traditional_fakes = session.run(traditional_model.outputs_reshaped, feed_dict={mnist_inputs: originals})
        utils.show(traditional_fakes, title='traditional fakes')
        dna_fakes = session.run(dna_model.outputs_reshaped, feed_dict={mnist_inputs: originals})
        utils.show(dna_fakes, title='dna fakes')
