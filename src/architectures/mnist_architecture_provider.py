import tensorflow as tf


class MnistArchitectureProvider:

    def encoder_builder(self, x, z_dim: int):
        hidden_layer_neurons_count = 200
        hidden_layers_count = 3
        h = x
        h = tf.layers.flatten(h)
        for i in range(hidden_layers_count):
            h = tf.layers.dense(h, hidden_layer_neurons_count, activation=tf.nn.relu, name=f'Encoder_{i}')

        return tf.layers.dense(h, z_dim, name=f'Encoder_output')

    def decoder_builder(self, z, x_dim: int):
        hidden_layer_neurons_count = 200
        hidden_layers_count = 3
        h = z
        for i in range(hidden_layers_count):
            h = tf.layers.dense(h, hidden_layer_neurons_count, activation=tf.nn.relu, name=f'Decoder_{i}')

        h = tf.layers.dense(h, units=28*28, activation=tf.nn.sigmoid, name='Output')
        h = tf.reshape(h, [-1, 28, 28])
        return h
