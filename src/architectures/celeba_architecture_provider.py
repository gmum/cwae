import tensorflow as tf


class CelebaArchitectureProvider:

    def encoder_builder(self, x, z_dim: int):
        h = x
        for i, filter_count in enumerate([32, 32, 64, 64]):
            h = tf.layers.conv2d(x, kernel_size=(4, 4), strides=(2, 2), padding='same', activation=tf.nn.relu, 
                                 filters=filter_count, name=f'Encoder_Conv_{i}')
        h = tf.layers.flatten(h, name='Encoder_Flatten')
        h = tf.layers.dense(h, units=1024, activation=tf.nn.relu, name='Encoder_FC_0')
        h = tf.layers.dense(h, units=256, activation=tf.nn.relu, name='Encoder_FC_1')
        return tf.layers.dense(h, z_dim, name=f'Encoder_output')

    def decoder_builder(self, z, x_dim: int):
        h = tf.layers.dense(z, units=256, activation=tf.nn.relu, name='Decoder_FC_0')
        h = tf.layers.dense(z, units=1024, activation=tf.nn.relu, name='Decoder_FC_1')
        h = tf.reshape(h, [-1, 4, 4, 64])

        for filter_count in [64, 32, 32, 3]:
            h = tf.layers.conv2d_transpose(h, kernel_size=(4, 4), strides=(2, 2), activation=tf.nn.relu, 
                                           filters=filter_count, padding='same', name='Decoder_Deconv_1')
        return h
