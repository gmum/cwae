import tensorflow as tf


class Cifar10ArchitectureProvider:

    def encoder_builder(self, x, z_dim: int):
        h = tf.layers.conv2d(x, kernel_size=(2, 2), activation=tf.nn.relu, filters=3, name='Encoder_Conv_0')
        h = tf.layers.conv2d(h, kernel_size=(2, 2), strides=(2, 2), activation=tf.nn.relu, filters=32, 
                             name='Encoder_Conv_1')
        h = tf.layers.conv2d(h, kernel_size=(2, 2), activation=tf.nn.relu, filters=32, name='Encoder_Conv_2')
        h = tf.layers.conv2d(h, kernel_size=(2, 2), activation=tf.nn.relu, filters=32, name='Encoder_Conv_3')
        h = tf.layers.flatten(h, name='Encoder_Flatten')
        h = tf.layers.dense(h, units=128, activation=tf.nn.relu, name='Encoder_FC')
        return tf.layers.dense(h, z_dim, name=f'Encoder_output')

    def decoder_builder(self, z, x_dim: int):
        h = tf.layers.dense(z, units=128, activation=tf.nn.relu, name='Decoder_FC_0')
        h = tf.layers.dense(h, units=8192, activation=tf.nn.relu, name='Decoder_FC_1')
        h = tf.reshape(h, [-1, 16, 16, 32])
        h = tf.layers.conv2d_transpose(h, kernel_size=(2, 2), padding='same', activation=tf.nn.relu, filters=32, 
                                       name='Decoder_Deconv_1')
        h = tf.layers.conv2d_transpose(h, kernel_size=(2, 2), padding='same', activation=tf.nn.relu, filters=32, 
                                       name='Decoder_Deconv_2')
        h = tf.layers.conv2d_transpose(h, kernel_size=(3, 3), strides=(2, 2), filters=32, activation=tf.nn.relu, 
                                       name='Decoder_Deconv_3')
        h = tf.layers.conv2d(h, kernel_size=(2, 2), filters=3, activation=tf.nn.sigmoid, name='Decoder_Conv_1')
        return h
